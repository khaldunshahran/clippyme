from clippyme.integrations.dubbing import SUPPORTED_LANGUAGES, DubbingError


def test_supported_languages_contains_essentials():
    assert "en" in SUPPORTED_LANGUAGES
    assert "es" in SUPPORTED_LANGUAGES
    assert "it" in SUPPORTED_LANGUAGES
    assert "fr" in SUPPORTED_LANGUAGES
    assert "de" in SUPPORTED_LANGUAGES
    assert len(SUPPORTED_LANGUAGES) >= 30
