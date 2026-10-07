"""Every group and command of the CLI answers `--help`, documents every parameter, and every
command the docs name exists: help that says nothing and docs naming a removed verb both slipped
through before (no parameter description reached `--help` until 2026-09-29)."""

import io
import re
from pathlib import Path

import pytest
from rich.console import Console

from mainboard.cli import build

_PACKAGE = Path(__file__).parents[1]
# The package's README, and the workspace's agent skills when the package sits in its monorepo.
_DOCS = [
    doc
    for doc in (
        _PACKAGE / "README.md",
        _PACKAGE.parents[1] / "AGENTS.md",
        *sorted(_PACKAGE.parents[1].glob(".agents/skills/*/SKILL.md")),
    )
    if doc.is_file()
]
# A call as a doc spells it in code: a shell line in a fenced block (`$ ` optional) or an inline
# code span, `mb` or `mainboard` then the words of a command path.
_WORDS = r"(?:mb|mainboard) ((?:[a-z][a-z-]*)(?: [a-z][a-z-]*)*)"
_SHELL_LINE = re.compile(rf"^\s*(?:\$ )?{_WORDS}")
_SPAN = re.compile(rf"`{_WORDS}")


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


@pytest.mark.parametrize("path", paths(), ids=" ".join)
def test_every_parameter_says_what_it_is(path: tuple[str, ...]) -> None:
    rendered = io.StringIO()
    build().help_print(list(path), console=Console(file=rendered, width=4000, color_system=None))
    # A parameter row is `  --name: <description> [default: ...]`; a bare one says nothing.
    silent = re.findall(r"^  (\S+): (?:\[default|$)", rendered.getvalue(), re.MULTILINE)
    assert not silent, f"{' '.join(path)}: no description for {silent}"


def test_every_command_the_docs_name_exists() -> None:
    known = {" ".join(path) for path in paths()}
    missing = set()
    for doc in _DOCS:
        fenced = False
        for line in doc.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            calls = _SPAN.findall(line)
            if fenced and (shell := _SHELL_LINE.match(line)):
                calls.append(shell.group(1))
            for call in calls:
                words = call.split()
                # The longest known command the words start with; a tool's own argument
                # (`mb run pytest`) follows a known command and is not a command itself.
                if not any(" ".join(words[:n]) in known for n in range(1, len(words) + 1)):
                    missing.add(f"{doc.name}: mb {call}")
    assert not missing, sorted(missing)
