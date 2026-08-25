import os
import time
import subprocess
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple
import yt_dlp
from yt_dlp.utils import sanitize_filename

logger = logging.getLogger("downloader_api")

app = FastAPI(title="Clippyme Downloader Microservice")
bgutil_process = None

class DownloadRequest(BaseModel):
    url: str
    output_dir: str
    cookies_file_path: Optional[str] = None

# We'll adapt the original _FORMAT_LADDER and attempts from download.py
_FORMAT_LADDER = (
    'bestvideo[vcodec^=avc1][height>=720][height<=1080][ext=mp4]+bestaudio[ext=m4a]/'
    'bestvideo[height>=720][height<=1080]+bestaudio/'
    'bestvideo[height<=1080]+bestaudio/'
    'best'
)

SOURCE_INFO_FILENAME = "source_info.json"

def _resolve_cookies_path(cookies_file_path):
    if cookies_file_path and os.path.exists(cookies_file_path):
        return os.path.abspath(cookies_file_path)
    repo_root_cookies = os.path.join(os.path.dirname(__file__), "..", "..", "..", "cookies.txt")
    if os.path.exists(repo_root_cookies):
        return os.path.abspath(repo_root_cookies)
    env_cookies = os.environ.get("YOUTUBE_COOKIES")
    if env_cookies:
        env_path = os.path.join("data", "cookies_env.txt")
        os.makedirs(os.path.dirname(env_path) or ".", exist_ok=True)
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_cookies)
        return os.path.abspath(env_path)
    return None

def _write_source_info(output_dir, info):
    try:
        from clippyme.domain.banner import suggest_banner
        channel_url = info.get("channel_url") or info.get("uploader_url")
        webpage_url = info.get("webpage_url") or info.get("original_url")
        uploader_id = info.get("uploader_id") or info.get("channel_id")
        banner = suggest_banner(channel_url or webpage_url or "", channel_hint=uploader_id)
        data = {
            "uploader_id": uploader_id,
            "channel_url": channel_url,
            "webpage_url": webpage_url,
            "banner": banner,
        }
        tmp = os.path.join(output_dir, SOURCE_INFO_FILENAME + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(output_dir, SOURCE_INFO_FILENAME))
    except Exception as exc:
        logger.warning(f"source_info capture skipped: {exc}")

def _extractor_args_for(attempt: str):
    if not attempt or attempt.lower() == "default":
        return None
    clients = [c.strip() for c in attempt.split("+") if c.strip()]
    return {"youtube": {"player_client": clients}}

def _format_bytes(b):
    if not b or b <= 0:
        return ""
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.1f} MB"

def _format_speed(s):
    if not s or s <= 0:
        return ""
    if s < 1024 * 1024:
        return f"{s / 1024:.1f} KB/s"
    return f"{s / (1024 * 1024):.1f} MB/s"

def _format_eta(secs):
    if not secs or secs < 0:
        return ""
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _get_progress_hook(output_dir: str):
    last_update = [0]
    def hook(d):
        if d.get('status') == 'downloading':
            now = time.time()
            if now - last_update[0] < 0.5:
                return
            last_update[0] = now
            
            downloaded = d.get('downloaded_bytes') or 0
            total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
            percent = 0
            if total > 0:
                percent = int(downloaded * 100 / total)
            elif '_percent_str' in d:
                try:
                    p_str = d['_percent_str'].strip().replace('%', '')
                    percent = int(float(p_str))
                except Exception:
                    pass
            
            speed_val = d.get('speed')
            speed_str = (d.get('_speed_str') or "").strip() or _format_speed(speed_val)
            eta_val = d.get('eta')
            eta_str = (d.get('_eta_str') or "").strip() or _format_eta(eta_val)
            downloaded_str = _format_bytes(downloaded)
            total_str = _format_bytes(total)
            
            try:
                from clippyme.domain.runtime_state import RuntimeState
                rs = RuntimeState(output_dir)
                rs.data["download_percent"] = max(0, min(100, percent))
                if speed_str:
                    rs.data["download_speed"] = speed_str
                if eta_str:
                    rs.data["download_eta"] = eta_str
                if downloaded_str and total_str:
                    rs.data["download_bytes"] = f"{downloaded_str} / {total_str}"
                rs.save()
            except Exception as e:
                logger.warning(f"Error saving download runtime state: {e}")
    return hook


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _is_safe_output_dir(target_dir: str) -> bool:
    if not target_dir:
        return False
    abs_target = os.path.abspath(target_dir)
    allowed_roots = [
        REPO_ROOT,
        os.path.abspath(os.getenv("OUTPUT_DIR", "output")),
        os.path.abspath("data"),
        os.path.abspath("tmp"),
    ]
    for root in allowed_roots:
        try:
            if os.path.commonpath([abs_target, root]) == root:
                return True
        except ValueError:
            continue
    return False


@app.post("/download")
def download_video(req: DownloadRequest):
    if not _is_safe_output_dir(req.output_dir):
        raise HTTPException(status_code=400, detail="Invalid or unauthorized output_dir")
    step_start_time = time.time()
    cookies_path = _resolve_cookies_path(req.cookies_file_path)
    
    attempts = [
        ("default", False, False),
        ("web_safari", False, False),
        ("default", False, True),
        ("web_safari", False, True),
    ]
    if cookies_path:
        attempts.extend([
            ("default", True, True),
            ("web_safari", True, True),
        ])
    attempts.extend([
        ("android_vr", False, False),
        ("android", False, False),
        ("ios", False, False),
        ("mweb", False, False),
    ])

    base_ydl_opts = {
        'format': _FORMAT_LADDER,
        'merge_output_format': 'mp4',
        'quiet': False,
        'verbose': True,
        'no_warnings': False,
        'socket_timeout': 30,
        'retries': 10,
        'fragment_retries': 10,
        'http_chunk_size': 10485760,
        'cachedir': False,
        'remote_components': ['ejs:github'],
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
        },
    }

    last_error = None
    for i, (client_name, use_cookies, use_po_token) in enumerate(attempts, 1):
        extractor_args = _extractor_args_for(client_name)
        active_cookiefile = cookies_path if (use_cookies and cookies_path) else None
        attempt_opts = {
            **base_ydl_opts,
            'cookiefile': active_cookiefile,
        }
        
        if use_po_token:
            if not extractor_args:
                extractor_args = {}
            extractor_args['youtubepot-bgutilhttp'] = {'base_url': ['http://127.0.0.1:4416']}
            
        if extractor_args:
            attempt_opts['extractor_args'] = extractor_args

        try:
            with yt_dlp.YoutubeDL(attempt_opts) as ydl:
                info = ydl.extract_info(req.url, download=False)
                video_title = info.get('title', 'remote_video')
                sanitized_title = sanitize_filename(video_title)
                _write_source_info(req.output_dir, info)

            output_template = os.path.join(req.output_dir, f'{sanitized_title}.%(ext)s')
            expected_file = os.path.join(req.output_dir, f'{sanitized_title}.mp4')
            if os.path.exists(expected_file):
                os.remove(expected_file)

            ydl_opts = {
                **attempt_opts,
                'format': _FORMAT_LADDER,
                'outtmpl': output_template,
                'merge_output_format': 'mp4',
                'overwrites': True,
                'progress_hooks': [_get_progress_hook(req.output_dir)],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([req.url])

            downloaded_file = os.path.join(req.output_dir, f'{sanitized_title}.mp4')
            if not os.path.isfile(downloaded_file):
                mp4_candidates = [
                    os.path.join(req.output_dir, f)
                    for f in os.listdir(req.output_dir)
                    if f.endswith('.mp4') and not f.startswith('clip_') and not f.startswith('source_clip_')
                ]
                if mp4_candidates:
                    downloaded_file = max(mp4_candidates, key=os.path.getmtime)
            if not os.path.isfile(downloaded_file):
                raise FileNotFoundError("yt-dlp completed without producing an MP4 file")

            return {"downloaded_file": downloaded_file, "sanitized_title": sanitized_title}

        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {i} failed: {e}")
            from clippyme.pipeline.download import classify_download_error
            # If it's a fatal error, don't retry
            err_msg = str(e)
            if classify_download_error(err_msg) == "fatal":
                logger.error(f"Fatal error encountered: {err_msg}")
                raise HTTPException(status_code=400, detail=f"Fatal download error: {err_msg}")
            # Else continue to next attempt

    raise HTTPException(status_code=500, detail=f"All download attempts failed. Last error: {last_error}")

bgutil_process = None

@app.on_event("startup")
def startup_event():
    global bgutil_process
    repo_dir = os.path.join(REPO_ROOT, "data", "bgutil-ytdlp-pot-provider")
    server_dir = os.path.join(repo_dir, "server")
    if not os.path.exists(repo_dir):
        logger.info("Cloning bgutil-ytdlp-pot-provider...")
        subprocess.run(["git", "clone", "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git", repo_dir], check=True)
        subprocess.run(["npm", "install"], shell=True, cwd=server_dir, check=True)
    elif not os.path.exists(os.path.join(server_dir, "node_modules")):
        logger.info("Installing bgutil-ytdlp-pot-provider dependencies...")
        subprocess.run(["npm", "install"], shell=True, cwd=server_dir, check=True)
    else:
        logger.info("bgutil-ytdlp-pot-provider already exists.")
        logger.info("Running npm install in bgutil-ytdlp-pot-provider...")
        subprocess.run(["npm", "install"], cwd=server_dir, shell=True, check=True)
    
    logger.info("Compiling typescript (npx tsc)...")
    subprocess.run(["npx", "tsc"], cwd=server_dir, shell=True, check=True)
    logger.info("bgutil-ytdlp-pot-provider build complete.")
    
    logger.info("Starting bgutil-ytdlp-pot-provider HTTP server...")
    bgutil_process = subprocess.Popen(["node", "build/main.js"], cwd=server_dir, shell=False)
    time.sleep(2) # Wait for node to bind to port 4416

@app.on_event("shutdown")
def shutdown_event():
    global bgutil_process
    if bgutil_process:
        logger.info("Stopping bgutil-ytdlp-pot-provider server...")
        bgutil_process.terminate()
        bgutil_process.wait()
    logger.info("Downloader Microservice stopping.")
