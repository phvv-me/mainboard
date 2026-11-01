"""A fix the project's proof rejects is undone before anything is written; the fixes it passes
are written. The proof here insists one file keeps its `print`, the way a test would fail on a
fix that changed what a program does. What is written is as tidy as it was found, by the project's
own Ruff configuration, and a fix that renames a folder writes the files under it."""

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard.analysis.commands.gate import Options, gate
from mainboard.analysis.engine.fixing import Fixpoint
from mainboard.analysis.engine.lint import Lint, Selection, fix, load_catalog
from mainboard.analysis.rules.general.dependencies.modules import NonAsciiSourcePath
from mainboard.analysis.rules.python.libraries.logging import TextAroundTheLogger

_CHECK = "import pathlib, sys; sys.exit('print(' not in pathlib.Path('spared.py').read_text())"
# A line the default 88 columns would wrap, and the project's own 120 keeps whole.
_LONG = "kept whole, a line that Ruff's default width would wrap and the project's width keeps"


def _printing(root: Path, *, config: str = "") -> None:
    """A project whose `kept.py` and `spared.py` each print, under `[tool.mcmr]` plus `config`."""
    _written(root, {"pyproject.toml": f"[tool.mcmr]\n{config}"})
    _written(root, {f"{name}.py": f'print("{name}")\n' for name in ("kept", "spared")})


def _written(root: Path, files: Mapping[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


def test_a_fix_the_proof_rejects_is_undone_and_the_rest_written(tmp_path: Path) -> None:
    load_catalog()
    _printing(tmp_path, config=f"proof = [{sys.executable!r}, '-c', {_CHECK!r}]\n")

    with Lint(tmp_path, (TextAroundTheLogger,)) as lint:
        fixer = Fixpoint(lint, tmp_path, ("safe",), proof=lint.project.proof)
        report = list(fixer.run(write=True))

    kept, spared = (
        (tmp_path / name).read_text(encoding="utf-8") for name in ("kept.py", "spared.py")
    )
    assert "logger.result" in kept and spared == 'print("spared")\n', report
    # `refactor gate --fix` reads what the run refused.
    assert any(line.startswith("proof: undid") and "spared.py" in line for line in report)
    assert not fixer.refusals


def test_a_gate_that_applies_a_safe_fix_writes_it_and_names_the_change(
    tmp_path: Path, capsys
) -> None:
    _printing(tmp_path)

    code = gate(
        Path("kept.py"), root=tmp_path, options=Options(fix=True, rule=(TextAroundTheLogger.id,))
    )

    shown = capsys.readouterr()
    assert code == 0 and "logger.result" in (tmp_path / "kept.py").read_text(encoding="utf-8")
    assert "kept.py:1 PY-LOGG0002 rewrote 1 span" in shown.out + shown.err


@pytest.mark.parametrize(
    ("source", "fixed"),
    [
        pytest.param(
            f'import sys\n\nARGS = sys.argv\nprint("{_LONG}")\n',
            f'import sys\n\nfrom mb import logger\n\nARGS = sys.argv\nlogger.result("{_LONG}")\n',
            id="a-tidy-file-is-settled-and-formatted-by-its-own-configuration",
        ),
        pytest.param(
            'import sys\nARGS=sys.argv\nprint("spared")\n',
            'import sys\nfrom mb import logger\nARGS=sys.argv\nlogger.result("spared")\n',
            id="an-untidy-file-keeps-the-shape-it-was-found-in",
        ),
    ],
)
def test_a_kept_fix_settles_imports_and_formats_a_file_only_as_far_as_it_was_tidy(
    tmp_path: Path, source: str, fixed: str
) -> None:
    load_catalog()
    config = "[tool.mcmr]\n\n[tool.ruff]\nline-length = 120\n"
    _written(tmp_path, {"pyproject.toml": config, "module.py": source})

    with Lint(tmp_path, (TextAroundTheLogger,)) as lint:
        report = list(Fixpoint(lint, tmp_path, ("safe",)).run(write=True))

    assert (tmp_path / "module.py").read_text(encoding="utf-8") == fixed, report


def test_a_fix_that_renames_a_populated_folder_writes_the_files_under_it(tmp_path: Path) -> None:
    names = ["__init__.py", "mod.py", "sub/__init__.py", "sub/deep.py"]
    _written(tmp_path, {"pyproject.toml": "[tool.mcmr]\n"})
    _written(tmp_path, {f"café/{name}": "X = 1\n" for name in names})

    fix(tmp_path, selection=Selection(rule=(NonAsciiSourcePath.id,)), write=True, unsafe=True)

    written = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.py"))
    assert written == [f"cafe/{name}" for name in names]
