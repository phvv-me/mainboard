"""`Statement.renamed` imports a name under a new name by replacing the name alone, so layout,
comments and an `as` alias stay, and `Statement.binds` tells whether the importing module's own
name follows the rename."""

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .builders import rewritten, statement_at

_PLAIN = "from .a import _hidden\n"


@pytest.mark.parametrize(
    ("source", "line", "after"),
    [
        pytest.param(_PLAIN, 1, "from .a import hidden\n", id="a-single-name"),
        pytest.param(
            "from .a import other, _hidden\n",
            1,
            "from .a import other, hidden\n",
            id="among-others",
        ),
        pytest.param(
            "from .a import _hidden as h\n", 1, "from .a import hidden as h\n", id="under-an-alias"
        ),
        pytest.param(
            "from .a import _hidden as _hidden\n",
            1,
            "from .a import hidden as _hidden\n",
            id="under-an-alias-of-the-same-name",
        ),
        pytest.param(
            "from .a import (\n    other,  # why\n    _hidden,\n)\n",
            1,
            "from .a import (\n    other,  # why\n    hidden,\n)\n",
            id="over-lines-with-a-comment",
        ),
        pytest.param(
            "if X:\n    from .a import other, _hidden\n",
            2,
            "if X:\n    from .a import other, hidden\n",
            id="in-an-indented-statement",
        ),
        pytest.param(
            "x = 1; from .a import _hidden\n",
            1,
            "x = 1; from .a import hidden\n",
            id="after-another-statement-on-its-line",
        ),
    ],
)
def test_the_imported_name_is_replaced_and_the_rest_of_the_statement_stays(
    source: str, line: int, after: str
) -> None:
    edits = statement_at(source, line).renamed("_hidden", new="hidden")

    assert edits is not None
    assert rewritten(source, edits) == after


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("from .a import other\n", id="a-statement-importing-other-names"),
        pytest.param("from .a import _hidden_too\n", id="a-name-that-only-starts-with-it"),
    ],
)
def test_a_statement_that_does_not_import_the_name_has_nothing_to_rename(source: str) -> None:
    assert statement_at(source).renamed("_hidden", new="hidden") is None


@pytest.mark.parametrize(
    ("source", "binds"),
    [
        pytest.param(_PLAIN, True, id="the-name-itself"),
        pytest.param("from .a import other, _hidden\n", True, id="the-name-among-others"),
        pytest.param("from .a import _hidden as h\n", False, id="the-name-under-another-name"),
        pytest.param("from .a import _hidden as _hidden\n", False, id="the-name-under-itself"),
        pytest.param("from .a import other\n", False, id="another-name"),
    ],
)
def test_a_statement_binds_a_name_when_it_imports_it_with_no_alias(
    source: str, binds: bool
) -> None:
    assert statement_at(source).binds("_hidden") is binds


@st.composite
def _imports(draw: st.DrawFn) -> tuple[list[str], bool, str]:
    """The names of a `from ... import`, whether it wraps them, and the comment on each line.

    The name `{}` is the one renamed, and some names come under an alias.
    """
    others = draw(st.lists(st.sampled_from(["alpha", "beta", "gamma"]), unique=True, max_size=3))
    names = draw(st.permutations([*others, "{}"]))
    aliased = draw(st.lists(st.booleans(), min_size=len(names), max_size=len(names)))
    spelled = [
        f"{name} as {name.strip('{}')}x" if alias else name
        for name, alias in zip(names, aliased, strict=True)
    ]
    return spelled, draw(st.booleans()), draw(st.sampled_from(["", "  # note"]))


def _written(names: Sequence[str], *, whole: str, wrapped: bool, note: str) -> str:
    """The statement importing `names`, with `{}` read as `whole`, on one line or wrapped."""
    spelled = [name.replace("{}", whole) for name in names]
    if not wrapped:
        return f"from .a import {', '.join(spelled)}\n"
    return "from .a import (\n" + "".join(f"    {name},{note}\n" for name in spelled) + ")\n"


@settings(max_examples=100)
@given(parts=_imports())
def test_renaming_any_layout_of_an_import_changes_the_name_and_nothing_else(
    parts: tuple[list[str], bool, str],
) -> None:
    names, wrapped, note = parts
    source = _written(names, whole="_hidden", wrapped=wrapped, note=note)
    after = _written(names, whole="hidden", wrapped=wrapped, note=note)

    edits = statement_at(source).renamed("_hidden", new="hidden")

    assert edits is not None
    assert rewritten(source, edits) == after
