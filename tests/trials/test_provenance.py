from importlib.metadata import PackageNotFoundError

import pytest

from mainboard.trials import provenance


def test_a_package_whose_every_distribution_is_gone_reads_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distribution index can name a package the environment no longer carries."""

    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(provenance, "version", missing)
    monkeypatch.setattr(provenance, "packages_distributions", lambda: {"ghost": ["gone", "away"]})
    assert provenance.installed("ghost") == "absent"
