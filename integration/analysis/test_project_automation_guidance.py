"""Onboarding credits the documented task token, including locked environment selection."""

from pathlib import Path

import pytest

from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.rules.general.project.onboarding import OnboardingReadiness


@pytest.mark.parametrize(
    ("command", "documented"),
    [
        ("mainboard run setup", True),
        ("mainboard run --locked setup", True),
        ("mainboard run --frozen setup", True),
        ("mainboard run --no-install --locked setup", True),
        ("mainboard run --as-is setup", True),
        ("mainboard run setup-extra", False),
        ("mainboard run --locked setup.extra", False),
        ("mainboard run --unknown setup", False),
        ("other-mainboard run setup", False),
        ("mainboard run test", False),
    ],
)
def test_onboarding_matches_the_exact_documented_task(
    tmp_path: Path, command: str, documented: bool
) -> None:
    load_catalog()
    (tmp_path / "mainboard.toml").write_text('[tasks]\nsetup = "mainboard binary sync"\n')
    (tmp_path / "README.md").write_text(f"Run `{command}` to prepare this checkout.\n")
    with Lint(tmp_path, (OnboardingReadiness,), fixing=False) as lint:
        found = lint.findings(lint.facts(tmp_path))
    rows = found[OnboardingReadiness.id]
    assert len(rows) == 1
    expected = 20.0 if documented else 0.0
    assert f"onboarding readiness is {expected}" in rows[0][3]
