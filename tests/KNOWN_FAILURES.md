# Known test failures (do not "fix" by weakening the test)

## Current state: ZERO known failures

Full suite (laptop run 2026-10-07, after the follow-up batch):
**1366 passed / 1 skipped / 0 failed.** All 7 baseline failures below are
resolved; the file is kept as a resolution log. A new failure must either be
fixed or get an entry here with cause + recommended fix before the next tag.

## Resolved failures (were failing at baseline 1347/1/7, Slice 1 commit 425cc04)

### 1-2. `tests/domain/test_encode.py::test_video_args_shape` + `::test_video_args_overrides_and_toggles`
**Cause:** test isolation, not product code. The laptop sets the machine-level
env var `CLIPPYME_VIDEO_ENCODER=nvenc`; the tests assumed the libx264 default.
**Fixed by:** autouse `_pin_video_encoder` fixture in `tests/conftest.py`
(forces `CLIPPYME_VIDEO_ENCODER=libx264` unless a test overrides it).

### 3. `tests/pipeline/test_run_ops.py::test_vfr_normalization_command_uses_shared_encode_policy`
**Cause:** same env isolation bug as 1-2.
**Fixed by:** the same `_pin_video_encoder` fixture.

### 4. `tests/pipeline/test_run_ops.py::test_cut_command_shape_and_precision`
### 5. `tests/pipeline/test_run_ops.py::test_cut_command_uses_shared_x264_settings`
**Cause:** code/test drift. `build_cut_command` is a stream copy
(`-c:v copy -c:a copy`); the tests asserted the old re-encode contract and
the docstring still claimed x264/CRF18.
**Fixed by:** updated both tests to the stream-copy contract (renamed the
second to `test_cut_command_is_stream_copy_not_reencode`), fixed the
docstring, and added `test_stream_copy_cut_keyframe_granularity`, which cuts
a real file and measures the keyframe offset empirically (see the pre-Slice-2
finding on GOP-sized caption offset in the commit notes).

### 6. `tests/api/test_config_routes.py::test_cross_site_request_rejected`
### 7. `tests/api/test_security.py::test_require_trusted_request_cross_site_beats_trusted_origin`
**Cause:** policy superseded, not a regression. Commit `a8cd0f9` ("trust
allow-listed Origin before Sec-Fetch-Site rejection") deliberately changed
`require_trusted_config_request`; the tests encoded the pre-`a8cd0f9` policy.
**Fixed by:** user decision 2026-10-07 (KEEP the new policy) — both tests now
assert the new contract: allow-listed Origin passes despite cross-site
Sec-Fetch-Site; unlisted Origin rejected; cross-site with no Origin rejected.
