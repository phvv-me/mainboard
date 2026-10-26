"""A fix the project's proof rejects is undone before anything is written; the fixes it passes
are written. The proof here insists one file keeps its `print`, the way a test would fail on a
fix that changed what a program does."""

import sys
from pathlib import Path

from mainboard.analysis.engine.fixing import Fixpoint
from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.rules.python.libraries.logging import TextAroundTheLogger

_CHECK = "import pathlib, sys; sys.exit('print(' not in pathlib.Path('spared.py').read_text())"


def test_a_fix_the_proof_rejects_is_undone_and_the_rest_written(tmp_path: Path) -> None:
    load_catalog()
    (tmp_path / "pyproject.toml").write_text(
        f"[tool.mcmr]\nproof = [{sys.executable!r}, '-c', {_CHECK!r}]\n", encoding="utf-8"
    )
    (tmp_path / "kept.py").write_text('print("kept")\n', encoding="utf-8")
    (tmp_path / "spared.py").write_text('print("spared")\n', encoding="utf-8")

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
