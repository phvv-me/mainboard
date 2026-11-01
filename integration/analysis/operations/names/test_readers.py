"""`name_readers` is the one row of everything that reads top-level names of a module from outside:
the `from ... import` statements taking them, the reads through a name bound to the module, the
reads spelling the module in full, and whether some module imports every name of it."""

import pytest
from patos import FrozenModel

from .builders import facts_cursor

type Record = dict[str, str | int | list[str]]

_TAKING, _READING, _SPELLING, _SKIPPING = (f"src/pkg/{name}.py" for name in "bcde")
_SOURCES = {
    "src/pkg/a.py": "_hidden = 1\nother = 2\n",
    _TAKING: "from .a import _hidden\n\nVALUE = _hidden\n",
    _READING: "from . import a\n\nVALUE = a._hidden\n",
    _SPELLING: "import pkg.a\n\nVALUE = pkg.a._hidden\n",
    _SKIPPING: "from .a import other\n",
    "src/pkg/f.py": "from .a import *\n",
}


class Readers(FrozenModel):
    """The columns of a `name_readers` row; a list no reader filled is NULL."""

    importers: list[Record] | None
    reached: list[Record] | None
    dotted: list[Record] | None
    starred: bool


def _readers(sources: dict[str, str], names: list[str]) -> Readers:
    """The `name_readers` row of `names` of `pkg.a` over the facts of a scratch project."""
    with facts_cursor(sources) as cursor:
        result = cursor.execute("SELECT * FROM name_readers('pkg.a', ?)", [names])
        columns = [column for column, *_ in result.description]
        return Readers(**dict(zip(columns, result.fetchone() or [], strict=True)))


def test_the_row_lists_each_kind_of_reader_of_a_name_with_its_file() -> None:
    row = _readers(_SOURCES, ["_hidden"])

    assert [(each["path"], each["content"]) for each in row.importers or []] == [
        (_TAKING, _SOURCES[_TAKING])
    ]
    assert [(each["path"], each["name"]) for each in row.reached or []] == [(_READING, "_hidden")]
    assert [(each["path"], each["name"]) for each in row.dotted or []] == [(_SPELLING, "_hidden")]
    assert row.starred is True


def test_a_read_ends_where_the_name_it_reads_ends() -> None:
    row = _readers(_SOURCES, ["_hidden"])

    reads = [*(row.reached or []), *(row.dotted or [])]

    ends = [(each["end_line"], each["end_column"]) for each in reads]
    assert ends == [(3, len("VALUE = a._hidden")), (3, len("VALUE = pkg.a._hidden"))]


@pytest.mark.parametrize(
    ("names", "importers"),
    [
        pytest.param(["_hidden", "other"], [_TAKING, _SKIPPING], id="two-names"),
        pytest.param(["other"], [_SKIPPING], id="a-name-one-module-takes"),
    ],
)
def test_the_row_lists_the_readers_of_every_name_asked_for(
    names: list[str], importers: list[str]
) -> None:
    row = _readers(_SOURCES, names)

    assert [each["path"] for each in row.importers or []] == importers


def test_a_module_nobody_reads_has_a_row_with_no_readers() -> None:
    row = _readers({"src/pkg/a.py": _SOURCES["src/pkg/a.py"]}, ["_hidden"])

    assert (row.importers, row.reached, row.dotted, row.starred) == (None, None, None, False)
