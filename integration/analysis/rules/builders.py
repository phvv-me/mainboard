from collections.abc import Mapping
from pathlib import Path

from mainboard.analysis.engine.lint import Selection, fix, load_catalog

from ..engine.builders import written


def fixed(
    root: Path, files: Mapping[str, str], *, rules: tuple[str, ...], config: str = "[tool.mcmr]\n"
) -> dict[str, str]:
    """Fix a project of `files` with `rules` (the fixes marked for review too), as `lint fix` does.

    The project holds a `pyproject.toml` with `config`. Answers every file's text afterwards.
    """
    written(root, {"pyproject.toml": config, **files})
    load_catalog()
    fix(root, selection=Selection(rule=rules), write=True, unsafe=True)
    return {name: (root / name).read_bytes().decode() for name in files}
