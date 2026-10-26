"""A fix the project's proof rejects is undone before anything is written; the fixes it passes
are written. The proof here insists one file keeps its `print`, the way a test would fail on a
fix that changed what a program does."""

import sys
from pathlib import Path

from mainboard.analysis.commands.gate import gate
from mainboard.analysis.engine.fixing import Fixpoint
from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.rules.python.libraries.logging import TextAroundTheLogger

_CHECK = "import pathlib, sys; sys.exit('print(' not in pathlib.Path('spared.py').read_text())"


def _printing(root: Path, *, config: str = "") -> None:
    """A project whose `kept.py` and `spared.py` each print, under `[tool.mcmr]` plus `config`."""
    (root / "pyproject.toml").write_text(f"[tool.mcmr]\n{config}", encoding="utf-8")
    for name in ("kept", "spared"):
        (root / f"{name}.py").write_text(f'print("{name}")\n', encoding="utf-8")


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

    code = gate(Path("kept.py"), root=tmp_path, fix=True, rule=(TextAroundTheLogger.id,))

    shown = capsys.readouterr()
    assert code == 0 and "logger.result" in (tmp_path / "kept.py").read_text(encoding="utf-8")
    assert "kept.py:1 PY-LOGG0002 rewrote 1 span" in shown.out + shown.err
