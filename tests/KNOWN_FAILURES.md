# Known test failures (do not "fix" by weakening the test)

Baseline (Slice 1, commit `425cc04`, laptop run 2026-10-06):
**1347 passed / 1 skipped / 7 failed.** The 7 below fail identically on the
base commit and are untouched by Slice 1. Slice 2 must not increase this
count; a fix that resolves any of them should remove it from this list and
note the fixing commit.

## 1. `tests/domain/test_encode.py::test_video_args_shape`
## 2. `tests/domain/test_encode.py::test_video_args_overrides_and_toggles`
**Cause:** test isolation, not product code. The laptop sets the machine-level
env var `CLIPPYME_VIDEO_ENCODER=nvenc`; the tests assume the libx264 default
and clear `CLIPPYME_X264_CRF`/`CLIPPYME_X264_PRESET` via monkeypatch but never
clear `CLIPPYME_VIDEO_ENCODER`. They pass in CI (var unset).
**Recommended fix (test-only):** add
`monkeypatch.delenv("CLIPPYME_VIDEO_ENCODER", raising=False)` to both tests.

## 3. `tests/pipeline/test_run_ops.py::test_cut_command_shape_and_precision`
## 4. `tests/pipeline/test_run_ops.py::test_cut_command_uses_shared_x264_settings`
**Cause:** code/test drift, not environment. `build_cut_command` was changed
to stream-copy (`-c:v copy -c:a copy`, no re-encode); the tests still assert
the old re-encode contract (`-vsync cfr`, shared x264 args in the argv). The
function's docstring is stale too (still claims "Shared x264 settings").
**Recommended fix:** update the two tests (and the docstring) to the
stream-copy contract — or deliberately reintroduce the re-encode.

## 5. `tests/pipeline/test_run_ops.py::test_vfr_normalization_command_uses_shared_encode_policy`
**Cause:** same env isolation bug as 1-2 (`CLIPPYME_VIDEO_ENCODER=nvenc`).
**Recommended fix (test-only):** clear the var via monkeypatch in the test.

## 6. `tests/api/test_config_routes.py::test_cross_site_request_rejected`
## 7. `tests/api/test_security.py::test_require_trusted_request_cross_site_beats_trusted_origin`
**Cause:** policy superseded, not a regression. Commit `a8cd0f9` ("trust
allow-listed Origin before Sec-Fetch-Site rejection") deliberately changed
`require_trusted_config_request` so an allow-listed `Origin` is trusted
regardless of `Sec-Fetch-Site`; the tests still encode the pre-`a8cd0f9`
policy ("cross-site beats trusted origin"). The current behavior is
documented in the function's docstring.
**Needs a product decision:** either update the two tests to the `a8cd0f9`
policy or revert the policy. Do not "fix" by deleting the tests.
