"""Every rule module on disk is loaded, namespace portions included: a package without an
initializer once left most of the catalog unregistered while every check still ran green."""

import sys
from pathlib import Path

import mainboard.analysis.rules as rules
from mainboard.analysis.engine.lint import load_catalog


def test_every_rule_file_is_imported_by_the_catalog() -> None:
    load_catalog()
    home = Path(next(iter(rules.__path__)))
    files = [
        ".".join(["mainboard.analysis.rules", *path.relative_to(home).with_suffix("").parts])
        for path in home.rglob("*.py")
        if path.name != "__init__.py"
    ]

    assert files
    assert [name for name in files if name not in sys.modules] == []
