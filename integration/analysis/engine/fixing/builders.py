from collections.abc import Sequence

from mainboard.analysis.engine.fixing.components import Applied


def applied(
    change: str, *, files: Sequence[str], moves: Sequence[tuple[str, str]] = ()
) -> Applied:
    """A kept fix that told `change` and changed `files`, moving each of `moves` (old, new)."""
    return Applied(
        rule="PY-TEST0015",
        path=files[0],
        line=1,
        change=change,
        files=frozenset(files),
        moves=tuple(moves),
    )


def module(*names: str) -> str:
    """A test module with a passing test for each of `names`."""
    return "".join(f"def {name}():\n    assert True\n\n\n" for name in names).rstrip("\n") + "\n"
