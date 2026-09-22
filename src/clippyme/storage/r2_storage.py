"""Cloudflare R2 / S3-compatible cloud object storage service.

Provides pure-Python standard-library S3 SigV4 uploads with zero external
dependencies (no boto3 required on host, though boto3 can be used if present).
Stores clips in Cloudflare R2 (zero egress fees) and returns public CDN URLs.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger("clippyme")


def get_r2_config() -> Dict[str, Any]:
    """Load R2 configuration from environment variables."""
    enabled = os.getenv("R2_ENABLED", "0").lower() in ("1", "true", "yes")
    account_id = os.getenv("R2_ACCOUNT_ID", "").strip()
    access_key_id = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    bucket_name = os.getenv("R2_BUCKET_NAME", "").strip()
    public_domain = os.getenv("R2_PUBLIC_DOMAIN", "").strip().rstrip("/")
    endpoint_url = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip("/")
    region = os.getenv("R2_REGION", "auto").strip() or "auto"
    auto_cleanup = os.getenv("R2_AUTO_CLEANUP", "0").lower() in ("1", "true", "yes")

    if not endpoint_url and account_id:
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    return {
        "enabled": enabled,
        "account_id": account_id,
        "access_key_id": access_key_id,
        "secret_access_key": secret_access_key,
        "bucket_name": bucket_name,
        "public_domain": public_domain,
        "endpoint_url": endpoint_url,
        "region": region,
        "auto_cleanup": auto_cleanup,
    }


def is_r2_configured(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if R2 is enabled and required credentials are set."""
    c = cfg or get_r2_config()
    return bool(
        c.get("enabled")
        and c.get("access_key_id")
        and c.get("secret_access_key")
        and c.get("bucket_name")
        and c.get("endpoint_url")
    )


def sign_v4_headers(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload_bytes: bytes,
    access_key: str,
    secret_key: str,
    region: str,
    service: str = "s3",
    timestamp: Optional[datetime.datetime] = None,
) -> Dict[str, str]:
    """Generate AWS Signature Version 4 headers for an HTTP request."""
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path or "/"

    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    now = timestamp or datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    headers_to_sign: Dict[str, str] = {
        "host": host,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }
    for k, v in headers.items():
        headers_to_sign[k.lower()] = v.strip()

    sorted_keys = sorted(headers_to_sign.keys())
    canonical_headers = "".join(f"{k}:{headers_to_sign[k]}\n" for k in sorted_keys)
    signed_headers = ";".join(sorted_keys)

    canonical_request = (
        f"{method}\n"
        f"{path}\n"
        f"{parsed.query}\n"
        f"{canonical_headers}\n"
        f"{signed_headers}\n"
        f"{payload_hash}"
    )

    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    canonical_req_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = f"{algorithm}\n{amz_date}\n{credential_scope}\n{canonical_req_hash}"

    def _sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = _sign(b"AWS4" + secret_key.encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    auth_header = (
        f"{algorithm} Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    result = dict(headers)
    result["Host"] = host
    result["x-amz-date"] = amz_date
    result["x-amz-content-sha256"] = payload_hash
    result["Authorization"] = auth_header
    return result


def upload_bytes_to_r2(
    data: bytes,
    remote_key: str,
    content_type: str = "video/mp4",
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Upload raw bytes to R2 and return the public or direct object URL."""
    c = cfg or get_r2_config()
    if not is_r2_configured(c):
        raise ValueError("Cloudflare R2 is not configured or enabled")

    remote_key = remote_key.lstrip("/")
    endpoint = c["endpoint_url"].rstrip("/")
    bucket = c["bucket_name"]
    url = f"{endpoint}/{bucket}/{remote_key}"

    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(data)),
    }
    signed_headers = sign_v4_headers(
        method="PUT",
        url=url,
        headers=headers,
        payload_bytes=data,
        access_key=c["access_key_id"],
        secret_key=c["secret_access_key"],
        region=c["region"],
    )

    req = urllib.request.Request(url, data=data, headers=signed_headers, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = getattr(resp, "status", 200)
            if status >= 300:
                raise RuntimeError(f"R2 upload failed with status {status}")
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"R2 upload failed with HTTP {err.code}: {body}") from err

    if c.get("public_domain"):
        return f"{c['public_domain']}/{remote_key}"
    return url


def upload_file_to_r2(
    local_path: str,
    remote_key: str,
    content_type: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Upload a local file to R2."""
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    if not content_type:
        guessed, _ = mimetypes.guess_type(local_path)
        content_type = guessed or "application/octet-stream"

    with open(local_path, "rb") as f:
        data = f.read()

    return upload_bytes_to_r2(data, remote_key, content_type=content_type, cfg=cfg)


def get_remote_clip_url(
    job_id: str,
    filename: str,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """Check if the clip was already uploaded to R2 and return its remote URL."""
    if not output_dir or not os.path.isdir(output_dir):
        return None

    cache_path = os.path.join(output_dir, "r2_urls.json")
    if not os.path.isfile(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            return mapping.get(filename)
    except Exception as exc:
        logger.debug("Failed reading r2_urls.json in %s: %s", output_dir, exc)
        return None


def get_clip_url(
    job_id: str,
    filename: str,
    output_dir: Optional[str] = None,
) -> str:
    """Return public R2 CDN URL if uploaded, else local /videos/{job_id}/{filename}."""
    remote = get_remote_clip_url(job_id, filename, output_dir)
    if remote:
        return remote
    return f"/videos/{job_id}/{filename}"


def sync_job_clips_to_r2(
    job_id: str,
    output_dir: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Upload all user-facing clips and media in output_dir to R2."""
    c = cfg or get_r2_config()
    if not is_r2_configured(c):
        return {}

    if not os.path.isdir(output_dir):
        return {}

    cache_path = os.path.join(output_dir, "r2_urls.json")
    uploaded: Dict[str, str] = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                uploaded = json.load(f)
        except Exception:
            uploaded = {}

    patterns = ["clip_*.mp4", "composed_*.mp4", "thumb_*.jpg", "cover_*.jpg"]
    files_to_sync: list[str] = []
    for pat in patterns:
        files_to_sync.extend(glob.glob(os.path.join(output_dir, pat)))

    updated = False
    for local_file in sorted(files_to_sync):
        leaf = os.path.basename(local_file)
        if leaf.startswith("source_"):
            continue
        if leaf in uploaded:
            continue
        if not os.path.isfile(local_file) or os.path.getsize(local_file) == 0:
            continue

        remote_key = f"clips/{job_id}/{leaf}"
        try:
            url = upload_file_to_r2(local_file, remote_key, cfg=c)
            uploaded[leaf] = url
            updated = True
            logger.info("Uploaded %s to R2: %s", leaf, url)

            if c.get("auto_cleanup"):
                try:
                    os.remove(local_file)
                    logger.info("Auto-cleaned local file: %s", leaf)
                except OSError as exc:
                    logger.warning("Could not auto-clean %s: %s", local_file, exc)
        except Exception as exc:
            logger.error("Failed uploading %s to R2: %s", leaf, exc)

    if updated or not os.path.isfile(cache_path):
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(uploaded, f, indent=2)
        except Exception as exc:
            logger.warning("Could not persist r2_urls.json in %s: %s", output_dir, exc)

    return uploaded
