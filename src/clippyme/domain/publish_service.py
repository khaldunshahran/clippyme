"""Publish-to-Zernio flow (moved out of app.py — thin-handler rule).

Owns: optional compose-first pass, upload-path resolution (fresh compose →
existing composed file → base clip), the blocking Zernio upload run off the
event loop, and the ZernioError → ClippyMeError mapping (preserving the
response-body snippet the frontend parses for per-platform 429 daily-limit
failures). Receives the request as a plain dict so this module never imports
``api.schemas``.
"""
import asyncio
import logging
import os
from datetime import datetime, timezone

from clippyme.domain.clip_resolve import ResolvedClip
from clippyme.domain.compose import compose_layers
from clippyme.domain.errors import ClippyMeError, NotFoundError, ValidationError
from clippyme.domain.job_artifacts import record_clip_publish

logger = logging.getLogger("clippyme")


async def publish_clip_flow(*, job_id: str, clip_index: int,
                            resolved: ResolvedClip, req: dict,
                            zernio_cfg: dict) -> dict:
    """Compose (optionally) and upload one clip to Zernio.

    ``req`` is ``PublishRequest.model_dump()``; ``resolved`` comes from
    ``resolve_clip(..., require_file=False)`` — the base clip may be absent
    when a composed file exists on disk.
    """
    api_key = zernio_cfg.get("api_key")
    if not api_key:
        raise ValidationError("Zernio API key not configured")

    from clippyme.domain.job_artifacts import is_clip_verified_published
    if is_clip_verified_published(resolved.clip_info):
        raise ClippyMeError("Clip already published", status_code=409)

    from clippyme.domain.clip_resolve import composed_clip_basename
    job_dir = resolved.job_dir
    base_clip = resolved.clip_path
    upload_path = base_clip
    composed_path = os.path.join(job_dir, composed_clip_basename(resolved.clip_info, clip_index))

    toggles = req.get("toggles")
    logger.info(
        "publish_clip_flow: job=%s clip=%d compose_first=%s toggles=%s has_hook_params=%s has_sub_params=%s",
        job_id, clip_index, req.get("compose_first"),
        list((toggles or {}).keys()),
        bool(req.get("hook_params")),
        bool(req.get("subtitle_params")),
    )

    if req.get("compose_first") and toggles:
        try:
            composed_filename = await compose_layers(
                base_clip=base_clip,
                job_dir=job_dir,
                clip_index=clip_index,
                metadata=resolved.metadata,
                clip_info=resolved.clip_info,
                toggles=toggles,
                hook_params=req.get("hook_params") or {},
                subtitle_params=req.get("subtitle_params") or {},
                logo_params=req.get("logo_params") or {},
                grade_params=req.get("grade_params") or {},
                banner_params=req.get("banner_params") or {},
                drop_ranges=req.get("drop_ranges"),
            )
            upload_path = os.path.join(job_dir, composed_filename)
        except ClippyMeError:
            raise
        except Exception as e:
            logger.error("publish: compose_layers failed for %s/%d: %s", job_id, clip_index, e)
            raise ClippyMeError(f"Compose before publish failed: {e}", status_code=500)
    elif os.path.exists(composed_path):
        upload_path = composed_path

    if not os.path.exists(upload_path):
        raise NotFoundError(f"Clip file not found: {upload_path}")

    # Phase 1B (OpusClip-level upgrade): gate publish on the FINAL file that
    # will actually be uploaded. QA used to inspect only the pre-compose
    # render; a broken compose (truncated encode, lost audio, blank burn)
    # could still ship. A critical finding here blocks the upload with a
    # clear error instead of uploading a broken export. Warnings stay
    # advisory. This runs BEFORE the Zernio upload call below and is outside
    # its try/except, so the raise is never swallowed.
    from clippyme.pipeline.media_qa import inspect_clip
    try:
        _qa_end = float((resolved.clip_info or {}).get("end", 0))
        _qa_start = float((resolved.clip_info or {}).get("start", 0))
        _qa_expected = _qa_end - _qa_start if _qa_end > _qa_start else None
    except (TypeError, ValueError):
        _qa_expected = None
    _qa_verdict = await asyncio.to_thread(
        inspect_clip,
        upload_path,
        expected_duration=_qa_expected,
        expected_aspect=None,  # aspect is render-gated; structural checks don't need it
        smartcut_applied=False,
    )
    if _qa_verdict.get("critical"):
        _qa_detail = "; ".join(_qa_verdict.get("issues") or [])
        logger.error(
            "publish: final-file QA BLOCKED upload for %s/%d: %s",
            job_id, clip_index, _qa_detail,
        )
        raise ClippyMeError(
            f"Publish blocked: final clip failed QA ({_qa_detail})",
            status_code=500,
        )
    if _qa_verdict.get("warnings"):
        logger.warning(
            "publish: final-file QA warnings for %s/%d: %s",
            job_id, clip_index, "; ".join(_qa_verdict["warnings"]),
        )

    # Resolve / generate thumbnail if requested
    thumbnail_path = req.get("thumbnail_path")
    if req.get("generate_ai_thumbnail") and not thumbnail_path:
        from clippyme.storage.config_store import load_persistent_config
        cfg = load_persistent_config() or {}
        gemini_key = os.environ.get("GEMINI_API_KEY") or cfg.get("GEMINI_API_KEY")
        if gemini_key:
            start = resolved.clip_info.get("start", 0)
            end = resolved.clip_info.get("end", 0)
            transcript = resolved.metadata.get("transcript") or {}
            from clippyme.domain.smartcut import clip_transcript_segments
            segments = clip_transcript_segments(transcript, start, end)
            clip_transcript_text = " ".join(s.get("text", "") for s in segments if s.get("text"))

            cover_candidate = os.path.splitext(base_clip)[0] + "_cover.jpg"
            face_img = cover_candidate if os.path.exists(cover_candidate) else None

            from clippyme.studio.youtube_studio import generate_youtube_thumbnail
            out_thumb_dir = os.path.join(job_dir, "thumbnails")
            resolved_title = req.get("title") or resolved.clip_info.get("title", "") or f"Clip {clip_index + 1}"
            try:
                thumbnail_path = await asyncio.to_thread(
                    generate_youtube_thumbnail,
                    title=resolved_title,
                    output_dir=out_thumb_dir,
                    api_key=gemini_key,
                    face_image_path=face_img,
                    bg_image_path=None,
                    extra_prompt=req.get("thumbnail_prompt") or "",
                    video_context=clip_transcript_text[:1500],
                    aspect_ratio=req.get("thumbnail_aspect_ratio", "9:16"),
                    model=req.get("thumbnail_model"),
                )
                logger.info("publish_clip_flow: generated AI thumbnail -> %s", thumbnail_path)
            except Exception as exc:
                logger.warning("publish: AI thumbnail generation failed: %s", exc)
                thumbnail_path = None

    # Phase 3D: A/B title testing (default off). When the request carries no
    # explicit title and the experiment is enabled, resolve the variant to
    # ship deterministically from the clip's stored title_variants. Guarded:
    # the experiment must never break publishing.
    ab_assignment = None
    try:
        from clippyme.domain.title_ab_testing import (
            ab_testing_enabled, resolve_title_for_post)
        if not req.get("title") and ab_testing_enabled():
            ab_key = (f"{job_id}:{clip_index}:{req.get('platforms')}:"
                      f"{req.get('scheduled_for') or 'now'}")
            ab_assignment = resolve_title_for_post(
                resolved.clip_info, assignment_key=ab_key)
            logger.info("publish: A/B title variant %s for %s/%d",
                        ab_assignment.get("variant"), job_id, clip_index)
    except Exception as exc:  # noqa: BLE001 - experiment is advisory only
        logger.warning("publish: A/B title resolution failed: %s", exc)
        ab_assignment = None

    # Run the publish in a worker thread (presign + PUT + create are blocking)
    from clippyme.integrations.social_publisher import publish_clip, ZernioError
    try:
        result = await asyncio.to_thread(
            publish_clip,
            api_key=api_key,
            clip_path=upload_path,
            title=req.get("title") or (ab_assignment or {}).get("title")
            or resolved.clip_info.get("title", "")[:100] or f"Clip {clip_index + 1}",
            caption=req.get("caption") or "",
            platform_targets=req.get("platforms"),
            schedule_mode=req.get("schedule_mode"),
            scheduled_for=req.get("scheduled_for"),
            # Phase 1E: the ultimate fallback is env-configurable
            # (ZERNIO_DEFAULT_TZ) instead of a hardcoded Europe/Rome, so a
            # US-oriented campaign is one env var away from correct timing.
            # Precedence: request > zernio config store > env > Europe/Rome.
            timezone=req.get("timezone") or zernio_cfg.get("timezone")
            or os.environ.get("ZERNIO_DEFAULT_TZ", "Europe/Rome"),
            tiktok_settings=req.get("tiktok_settings"),
            start_date=req.get("start_date"),
            thumbnail_path=thumbnail_path,
        )
    except ValueError as e:
        raise ValidationError(str(e))
    except ZernioError as e:
        logger.error("publish: Zernio error: %s (status=%s body=%s)",
                     e, e.status_code, (e.body or "")[:200])
        # Include the full response body (truncated) in the error detail so
        # the frontend can parse per-platform failures like the Zernio
        # "Daily limit reached" 429 and skip the exhausted platform for the
        # rest of a batch publish run.
        body_snippet = (e.body or "")[:500]
        detail_msg = f"Zernio API error: {e}"
        if body_snippet:
            detail_msg = f"{detail_msg} | body={body_snippet}"
        raise ClippyMeError(
            detail_msg,
            status_code=502 if e.status_code is None else e.status_code,
        )
    except Exception as exc:
        logger.exception("publish: unexpected error")
        try:
            from clippyme.domain.watchdog import on_api_error
            asyncio.create_task(on_api_error(f"/api/publish/{job_id}/{clip_index}", str(exc), 500, context={"job_id": job_id, "clip_index": clip_index}))
        except Exception:
            pass
        raise ClippyMeError("Publish failed", status_code=500)

    # Best-effort: the publish already succeeded, so a metadata-write hiccup
    # here must not fail the response — just leave the history badge stale.
    try:
        await asyncio.to_thread(
            record_clip_publish,
            job_id,
            clip_index,
            os.path.dirname(job_dir),
            {
                "platforms": req.get("platforms"),
                "post_id": result.get("post_id"),
                "scheduled_for": result.get("scheduled_for"),
                "at": datetime.now(timezone.utc).isoformat(),
            },
        )
        from clippyme.domain.analytics_service import record_published_clip as record_analytics
        await asyncio.to_thread(
            record_analytics,
            job_id,
            clip_index,
            resolved.clip_info,
            result,
            req.get("platforms"),
        )
        # Phase 3D: attribute the shipped A/B title variant to the analytics
        # record (best-effort; never fails the publish response).
        if ab_assignment and ab_assignment.get("enabled"):
            try:
                from clippyme.domain.title_ab_testing import (
                    record_title_variant_assignment)
                await asyncio.to_thread(
                    record_title_variant_assignment,
                    job_id,
                    clip_index,
                    result.get("post_id"),
                    ab_assignment,
                    platform=(req.get("platforms") or [None])[0],
                )
            except Exception as e:
                logger.warning("publish: A/B attribution failed: %s", e)
    except Exception as e:
        logger.warning("publish: failed to persist publish record for %s/%d: %s", job_id, clip_index, e)

    try:
        from clippyme.storage.config_store import load_persistent_config
        cfg = await asyncio.to_thread(load_persistent_config) or {}
        auto_cleanup = bool(cfg.get("AUTO_CLEANUP_PUBLISHED", True))
        if req.get("delete_after_publish") or auto_cleanup:
            from clippyme.domain.job_artifacts import delete_clip_artifacts, cleanup_published_job
            output_root = os.path.dirname(job_dir)
            await asyncio.to_thread(
                delete_clip_artifacts,
                job_id,
                resolved.clip_info,
                base_clip,
                upload_path,
                output_root
            )
            await asyncio.to_thread(
                cleanup_published_job,
                job_id,
                output_root,
                clip_index
            )
            logger.info("publish: successfully deleted post-publish artifacts for %s/%d", job_id, clip_index)
    except Exception as e:
        logger.warning("publish: failed to run post-publish artifact cleanup for %s/%d: %s", job_id, clip_index, e)

    return {"success": True, **result}
