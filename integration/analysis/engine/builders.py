from collections.abc import Mapping
from pathlib import Path


def written(root: Path, files: Mapping[str, str]) -> Path:
    """Write `files` (each path to its text) under `root`, making the folders; answers `root`."""
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root
