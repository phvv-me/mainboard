"""Every group and command of the CLI answers `--help`: parsing, imports and help render."""

import pytest

from mainboard.cli import build


def paths(node=None, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Every command path the CLI declares, groups included, walked from the root."""
    node = node or build()
    found: list[tuple[str, ...]] = []
    for name, sub in sorted(getattr(node, "_commands", {}).items()):
        if name.startswith("-"):
            continue
        found.append((*prefix, name))
        found += paths(sub, (*prefix, name))
    return found


@pytest.mark.parametrize("path", paths(), ids=" ".join)
def test_every_command_answers_help(mb, path: tuple[str, ...]) -> None:
    ran = mb(*path, "--help")
    assert ran.code == 0, ran.said
    assert "Usage:" in ran.out
