"""Shared pytest configuration.

Some test modules import ``clippyme.pipeline.main`` (and transitively
``ultralytics`` / ``cv2`` / ``mediapipe``) at module load time. Those modules
are marked ``@pytest.mark.integration`` so they only run in the Docker image
that ships the heavy CV/ML runtime — but a marker cannot stop a *collection*
import error: on a host without those wheels, pytest fails to import the module
before the marker is ever consulted, which breaks even ``-m "not integration"``.

We skip collecting those modules when the heavy runtime is unavailable so the
host suite (``pytest -m "not integration"``) stays green. In Docker, where the
wheels are present, the modules collect and run normally.
"""
import importlib.util

# Heavy deps that the pipeline integration tests import at module load.
_HEAVY_DEPS = ("ultralytics", "cv2", "mediapipe", "scenedetect")


def _heavy_runtime_available() -> bool:
    if not all(importlib.util.find_spec(dep) is not None for dep in _HEAVY_DEPS):
        return False
    # Presence isn't enough: main.py uses the legacy ``mediapipe.solutions`` API
    # (pinned 0.10.14). Newer host wheels (e.g. 0.10.35, the only one with a
    # Windows build) drop that attribute, so collecting the pipeline tests would
    # crash on import. Treat a wrong-API mediapipe as "runtime unavailable" so
    # the host suite stays green; the Docker image (0.10.14) passes this check.
    try:
        import mediapipe  # noqa: PLC0415
        return hasattr(mediapipe, "solutions")
    except Exception:
        return False


# When the runtime is missing, don't even try to collect the pipeline tests
# that import it at the top level.
collect_ignore_glob = []
if not _heavy_runtime_available():
    collect_ignore_glob.append("pipeline/test_main_*.py")


def pytest_addoption(parser):
    parser.addoption(
        "--generate-golden", action="store_true", default=False,
        help="regenerate golden-frame reference PNGs "
             "(CI environment only; committed references must come from CI)")


import pytest as _pytest  # noqa: E402


@_pytest.fixture
def generate_golden(request):
    return request.config.getoption("--generate-golden")


@_pytest.fixture(autouse=True)
def _pin_video_encoder(monkeypatch):
    """Hermetic encoder default for the host suite.

    The dev laptop sets ``CLIPPYME_VIDEO_ENCODER=nvenc`` machine-wide; tests
    asserting libx264-shaped encode args must not depend on machine env. Pin
    libx264 unless a test overrides it explicitly via monkeypatch.
    """
    monkeypatch.setenv("CLIPPYME_VIDEO_ENCODER", "libx264")


@_pytest.fixture(autouse=True)
def _auth_dev_bypass_default(monkeypatch):
    """Default the API test-suite to the explicit dev auth bypass.

    Auth hardening made "no auth config" fail closed (401). The bulk of the
    suite exercises API behavior, not auth, so tests run with
    AUTH_DISABLED_DEV=1 (local default admin) unless a test explicitly
    configures auth (AUTH_ENABLED=1 + SUPABASE_URL + mocked JWKS), which takes
    precedence over the bypass for JWT flows.
    """
    monkeypatch.setenv("AUTH_DISABLED_DEV", "1")
    # Ensure no ambient production/auth env leaks into the suite.
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("ADMIN_USER_IDS", raising=False)
    monkeypatch.delenv("CLIPPYME_API_TOKEN", raising=False)
