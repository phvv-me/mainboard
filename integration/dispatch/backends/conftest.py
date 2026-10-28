import pytest


@pytest.fixture(autouse=True)
def keys(monkeypatch) -> None:
    """Test keys for every cloud whose API is recorded here, so no request needs a real account."""
    for name in (
        "RUNPOD_API_KEY",
        "LAMBDA_API_KEY",
        "DIGITAL_OCEAN_API_KEY",
        "VERDA_CLIENT_ID",
        "VERDA_CLIENT_SECRET",
        "VULTR_API_KEY",
        "HOTAISLE_API_KEY",
        "HOTAISLE_TEAM_HANDLE",
        "SHADEFORM_API_KEY",
        "HYPERSTACK_API_KEY",
    ):
        monkeypatch.setenv(name, "test-key")
