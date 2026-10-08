"""Task 13 pipeline tests: shared ``framing_impl`` fixture."""

import pytest


@pytest.fixture
def framing_impl():
    """The Task 13 domain interface module (``clippyme.domain.project_framing``)."""
    try:
        import clippyme.domain.project_framing as impl
    except ImportError:
        pytest.skip("clippyme.domain.project_framing is not present")
    return impl
