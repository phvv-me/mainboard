"""The environment verbs behave as pixi's do.

`install` solves a lock the manifest moved past and then installs, `--locked` refuses such a lock
(what every host runs), `--frozen` installs it as it stands, `lock` solves without installing and
`--check` says when it had to, and `run --as-is` touches neither. Requirement edits keep a
table-style requirement's other fields.
"""

from pathlib import Path

import pytest

from mainboard.core.errors import MissionError
from mainboard.deps.editing import ManifestText
from mainboard.engines.compile.policy import LockPolicy

_MANIFEST = '[workspace]\nname = "it"\n\n[deps]\npython = ">=3.14"\n'


@pytest.fixture
def python_only(workspace: Path) -> Path:
    """A workspace declaring Python alone, the smallest environment that solves."""
    (workspace / "mb.toml").write_text(_MANIFEST, encoding="utf-8")
    return workspace


def test_install_solves_a_stale_lock_and_locked_refuses_it(mb, python_only: Path) -> None:
    assert mb("install").code == 0
    assert (python_only / "mb.lock").is_file()
    assert mb("lock", "--check").code == 0

    assert mb("add", "--frozen", "--lang", "python", "six>=1.15").code == 0
    refused = mb("install", "--locked")
    assert refused.code == 1 and "was not solved from the manifest" in refused.said

    assert mb("lock", "--check").code == 1
    assert "six" in (python_only / "mb.lock").read_text(encoding="utf-8")
    assert mb("lock", "--check").code == 0
    assert mb("install", "--frozen").code == 0


def test_as_is_runs_what_is_installed_without_solving(mb, python_only: Path) -> None:
    assert mb("install").code == 0
    lock = (python_only / "mb.lock").read_bytes()
    (python_only / "mb.toml").write_text(_MANIFEST + '\n[python.deps]\nsix = ">=1.15"\n')

    ran = mb("run", "--as-is", "python", "-c", "print('ran as it is')")

    assert ran.code == 0 and "ran as it is" in ran.out
    assert (python_only / "mb.lock").read_bytes() == lock


def test_a_failed_solve_leaves_the_manifest_and_lock_as_they_were(mb, python_only: Path) -> None:
    assert mb("install").code == 0
    kept = {name: (python_only / name).read_bytes() for name in ("mb.toml", "mb.lock")}

    refused = mb("add", "--no-install", "--lang", "python", "six>=999")

    assert refused.code == 1
    assert {name: (python_only / name).read_bytes() for name in kept} == kept
    assert mb("lock", "--check").code == 0


def test_upgrade_leaves_a_member_named_by_version(mb, python_only: Path) -> None:
    (python_only / "member").mkdir()
    (python_only / "member/pyproject.toml").write_text(
        '[project]\nname = "member"\nversion = "0.1"\n', encoding="utf-8"
    )
    member = '[python.deps]\nmember = "*"\n'
    manifest = _MANIFEST.replace("\n\n", '\nmembers = ["member"]\n\n') + member
    (python_only / "mb.toml").write_text(manifest, encoding="utf-8")

    assert mb("upgrade", "member", "--no-install").code == 0
    assert (python_only / "mb.toml").read_text(encoding="utf-8") == manifest


def test_locked_and_frozen_exclude_each_other() -> None:
    with pytest.raises(MissionError, match="exclude each other"):
        LockPolicy.of(locked=True, frozen=True)
    assert LockPolicy.of(locked=True).flag == "--locked"
    assert LockPolicy.of().flag == ""


def test_raising_a_table_requirement_keeps_its_other_fields() -> None:
    manifest = ManifestText(
        '[python.deps]\ntorch = { version = "==2.14.0", index = "https://example.invalid" }\n'
        'local = { path = "packages/local", editable = true }\n'
    )

    manifest.put(("python", "deps"), "torch", spec=">=2.15")

    assert manifest.constraint(("python", "deps"), "torch") == ">=2.15"
    assert 'index = "https://example.invalid"' in manifest.text()
    assert manifest.versioned(("python", "deps"), "torch")
    assert not manifest.versioned(("python", "deps"), "local")


@pytest.mark.parametrize(
    ("spec", "comment_at"),
    [(">=2.5.12", 33), (">=2.4.6,<2.5,!=2.4.9", 33), (">=2.4.6,<2.5,!=2.4.9,!=2.4.10", 41)],
)
def test_a_raised_requirement_keeps_its_comment_column(spec: str, comment_at: int) -> None:
    """The comment stays in its column, one space past the value once the value outgrows it."""
    manifest = ManifestText("""[deps]
numpy  = ">=2.4.6,<2.5"          # why capped
torch  = { version = "==2.14" }  # why pinned
""")

    manifest.put(("deps",), "numpy", spec=spec)
    manifest.put(("deps",), "torch", spec=">=2.15")

    numpy, torch = manifest.text().splitlines()[1:]
    assert numpy.startswith("numpy  = ") and numpy.index("#") == comment_at
    assert torch.index("#") == 33
