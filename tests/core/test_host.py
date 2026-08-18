import pytest

from mainboard.core.host import current_platform, platform_selectors


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", "linux-64"),
        ("Linux", "aarch64", "linux-aarch64"),
        ("Darwin", "arm64", "osx-arm64"),
        ("Darwin", "x86_64", "osx-64"),
        ("Windows", "AMD64", "win-64"),
    ],
)
def test_platform_mapping(
    monkeypatch: pytest.MonkeyPatch, *, system: str, machine: str, expected: str
) -> None:
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("platform.machine", lambda: machine)
    assert current_platform() == expected


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [
        ("linux-64", ("linux-64", "linux", "unix")),
        ("osx-arm64", ("osx-arm64", "osx", "unix")),
        ("linux", ("linux", "unix")),
        ("win-64", ("win-64", "win")),
    ],
)
def test_overlay_selectors_cover_a_platform_and_the_families_above_it(
    platform_name: str, expected: tuple[str, ...]
) -> None:
    assert platform_selectors(platform_name) == expected
