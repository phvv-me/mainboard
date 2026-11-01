from collections.abc import Mapping, Sequence

from hypothesis import strategies as st

from .modules.builders import module_of

# A move of definitions: the module's path, the names that leave it and the destination's path.
type Move = tuple[str, Sequence[str], str]
type Row = dict[str, str | list[str] | list[dict[str, str]] | None]


def row(sources: Mapping[str, str], move: Move) -> Row:
    """The `move-definitions` row of `move` over the Python files of `sources`, no importers.

    The destination's text is None when no file of `sources` stands there.
    """
    path, names, destination = move
    return {
        "operation": "move-definitions",
        "path": path,
        "module": module_of(path),
        "content": sources[path],
        "names": list(names),
        "destination": destination,
        "destination_module": module_of(destination),
        "destination_content": sources.get(destination),
        "importers": [],
        "reached": [],
    }


@st.composite
def enum_modules(draw: st.DrawFn) -> dict[str, str]:
    """Modules `src/pkg/m0.py`, `m1.py`, ..., each defining the enum `C0`, `C1`, ... it names.

    An enum reads `StrEnum` and `auto`, and some also `Path`, so the imports a move carries differ.
    """
    count = draw(st.integers(min_value=1, max_value=4))
    pathed = draw(st.lists(st.booleans(), min_size=count, max_size=count))
    return {
        f"src/pkg/m{index}.py": (
            "from enum import StrEnum, auto\n"
            + ("from pathlib import Path\n" if path else "")
            + f"\n\nclass C{index}(StrEnum):\n    A = auto()\n"
            + ("    HOME = Path\n" if path else "")
        )
        for index, path in enumerate(pathed)
    }
