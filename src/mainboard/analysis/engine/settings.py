"""The settings Ruff, ty, pyrefly and mypy read for a project: the files they come from and the
places those files name, which a mirror (mirror.py) must carry for the tools to read them as they
do in place."""

import configparser
import os
import re
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from .facts import Setting

# The files each checker reads its settings from, in the order it prefers them within one folder.
# A `pyproject.toml` or a `setup.cfg` counts only with the checker's own table.
_PYPROJECT = "pyproject.toml"
_FILES = {
    "ruff": (".ruff.toml", "ruff.toml", _PYPROJECT),
    "ty": ("ty.toml", _PYPROJECT),
    "pyrefly": ("pyrefly.toml", _PYPROJECT),
    "mypy": ("mypy.ini", ".mypy.ini", _PYPROJECT, "setup.cfg"),
}
# How an INI value lists several places (`mypy_path = src:stubs`).
_LISTED = re.compile(r"[:,\s]+")


class Settings:
    """The settings the checkers read for a project, found as they find them.

    Each checker reads the nearest file that configures it, the project's own or one above it, and
    Ruff also the files that one extends. A setting names a place by a relative path: a search
    root such as `..`, the stubs in `../../stubs`. Only Ruff's file may lie above the project's
    parent, as it always could; the others are looked for no higher than Ruff's.

    root: the lowest folder holding the project, those files and the places they name; the
        project's parent at least.
    outside: the files lying outside the project.
    named: the places they name that exist outside the project, other than those files, shortest
        path first.
    """

    @staticmethod
    def _chain(folders: Sequence[Path], tool: str) -> Iterator[tuple[Path, Mapping[str, Setting]]]:
        """The files `tool` reads, nearest first, each with its table.

        Ruff's end with the files that one extends.
        """
        found = Settings._nearest(folders, tool)
        seen: set[Path] = set()
        while found is not None and found[0] not in seen:
            seen.add(found[0])
            yield found
            found = Settings._extension(*found) if tool == "ruff" else None

    @staticmethod
    def _extension(
        file: Path, table: Mapping[str, Setting]
    ) -> tuple[Path, Mapping[str, Setting]] | None:
        """The file Ruff's `table` of `file` extends, with its table; None for none or no file."""
        target = table.get("extend")
        extended = (file.parent / target).resolve() if isinstance(target, str) else None
        if extended is None or not extended.is_file():
            return None
        return extended, Settings._table(extended, "ruff") or {}

    @staticmethod
    def _ini(text: str, *, tool: str) -> Mapping[str, Setting] | None:
        """The sections of `tool` in INI `text` (`mypy`, `mypy-pkg`); None for none or no INI.

        Each value lists its places.
        """
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except configparser.Error:
            return None
        sections = {
            name: {key: _LISTED.split(value) for key, value in parser[name].items()}
            for name in parser.sections()
            if name == tool or name.startswith(f"{tool}-")
        }
        return sections or None

    @staticmethod
    def _nearest(folders: Sequence[Path], tool: str) -> tuple[Path, Mapping[str, Setting]] | None:
        """The file in the first of `folders` that configures `tool`, with its table for it."""
        files = (folder / name for folder in folders for name in _FILES[tool])
        return next(
            (
                (file, table)
                for file in files
                if (table := Settings._table(file, tool)) is not None
            ),
            None,
        )

    @staticmethod
    def _places(folder: Path, table: Mapping[str, Setting]) -> Iterator[Path]:
        """The existing places the strings of `table` name relative to `folder`."""
        for text in Settings._strings(table):
            if text and not Path(text).is_absolute():
                place = Path(os.path.normpath(folder / text))
                if os.path.exists(place):
                    yield place

    @staticmethod
    def _strings(setting: Setting) -> Iterator[str]:
        """Every string in `setting`, however deep."""
        pending = [setting]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                yield item
            elif isinstance(item, Mapping):
                pending.extend(item.values())
            elif isinstance(item, Sequence):
                pending.extend(item)

    @staticmethod
    def _table(file: Path, tool: str) -> Mapping[str, Setting] | None:
        """What `file` holds for `tool`; None for a file not there, not valid or without it.

        A TOML file named for the tool is all its table, a `pyproject.toml` has it under
        `[tool.<tool>]`.
        """
        try:
            text = file.read_text(encoding="utf-8")
        except OSError:
            return None
        if file.suffix != ".toml":
            return Settings._ini(text, tool=tool)
        return Settings._toml(file, text, tool=tool)

    @staticmethod
    def _toml(file: Path, text: str, *, tool: str) -> Mapping[str, Setting] | None:
        """What TOML `text` of `file` holds for `tool`, None for invalid text or none."""
        try:
            table = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return None
        return table.get("tool", {}).get(tool) if file.name == _PYPROJECT else table

    def __init__(self, project: Path) -> None:
        above = (project, *project.parents)
        ruff = list(Settings._chain(above, "ruff"))
        reached = Path(os.path.commonpath([project, *(file.parent for file, _ in ruff)]))
        floor = project.parent if reached == project else reached
        within = [folder for folder in above if folder.is_relative_to(floor)]
        read = [
            *ruff,
            *(
                found
                for tool in _FILES
                if tool != "ruff"
                for found in Settings._chain(within, tool)
            ),
        ]
        self.outside = list(
            dict.fromkeys(file for file, _ in read if not file.is_relative_to(project))
        )
        self.named = sorted(
            {
                place
                for file, table in read
                for place in Settings._places(file.parent, table)
                if not place.is_relative_to(project) and place not in self.outside
            },
            key=lambda place: (len(place.parts), place),
        )
        self.root = Path(os.path.commonpath([floor, *self.named]))
