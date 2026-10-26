import tomllib
from collections.abc import Mapping

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.deps import ManifestText

type _Path = tuple[str, ...]
type _Entry = tuple[int, str, tuple[int, str] | None]
type _Case = tuple[str, _Path, str, str]

_PATHS: list[_Path] = [
    ("deps",),
    ("python", "deps"),
    ("on", "linux", "deps"),
    ("on", "linux", "python", "deps"),
    ("on", "unix", "deps"),
    ("dev", "python", "deps"),
    ("envs", "x", "python", "deps"),
]
_NAMES = ["numpy", "torch", "six", "httpx", "pkg_a"]
_SPECS = st.sampled_from(
    [
        ">=1",
        ">=2.15",
        ">=2.5.12",
        ">=2.4.6,<2.5,!=2.4.9",
        ">=2.4.6,<2.5,!=2.4.9,!=2.4.10",
        "==2.0",
        "*",
    ]
)
_ENTRIES = st.tuples(
    st.integers(0, 4),
    st.sampled_from(
        [
            '">=2.4.6,<2.5"',
            '"==1.9.0"',
            '"*"',
            '{ version = "==2.14.0", index = "https://example.invalid" }',
            '{ path = "packages/local", editable = true }',
            '{ version = "0.65.1", url = "https://example.invalid/lint.zip", sha256 = "00" }',
        ]
    ),
    st.none()
    | st.tuples(st.integers(1, 12), st.sampled_from(["why", "why capped", "byte-stable"])),
)
_TABLES = st.dictionaries(
    st.sampled_from(_PATHS),
    st.dictionaries(st.sampled_from(_NAMES), _ENTRIES, min_size=1, max_size=3),
    min_size=1,
    max_size=4,
)
_TABLE_STYLE = """[python.deps]
torch = { version = "==2.14.0", index = "https://example.invalid" }
local = { path = "packages/local", editable = true }
lint = { version = "0.65.1", url = "https://example.invalid/lint.zip", sha256 = "00" }
stable = "==1.9.0"  # byte-stable
"""
_SPLIT = """[on.linux.deps]
cuda = ">=13.0"

[on.unix.deps]
git = ">=2"

[on.linux.python.deps]
torch = ">=2.14, <3"  # why
"""
_TOGETHER = """[on.linux.deps]
cuda = ">=13.0"

[on.linux.python.deps]
torch = ">=2.14, <3"  # why

[on.unix.deps]
git = ">=2"
"""
_COLUMN = """[deps]
numpy  = ">=2.4.6,<2.5"          # why capped
torch  = { version = "==2.14" }  # why pinned
"""
_HEADED = _COLUMN + '# the next table\n\n[python.deps]\nsix = ">=1"\n'
_APPENDED = _COLUMN + 'httpx  = ">=0.28"\n# the next table\n\n[python.deps]\nsix = ">=1"\n'


def _text(tables: Mapping[_Path, Mapping[str, _Entry]]) -> str:
    """The manifest text declaring each entry of each table, comments and alignment as drawn."""
    return "\n".join(
        f"[{'.'.join(path)}]\n"
        + "".join(
            f"{name}{' ' * gap} = {value}"
            + (f"{' ' * comment[0]}# {comment[1]}" if comment else "")
            + "\n"
            for name, (gap, value, comment) in entries.items()
        )
        for path, entries in tables.items()
    )


@st.composite
def _raisable(draw: st.DrawFn) -> _Case:
    """A manifest, one requirement it declares and a constraint to raise it to."""
    tables = draw(_TABLES)
    path = draw(st.sampled_from(list(tables)))
    return _text(tables), path, draw(st.sampled_from(list(tables[path]))), draw(_SPECS)


@st.composite
def _declarable(draw: st.DrawFn) -> _Case:
    """A manifest whose tables sit together, a requirement it lacks and a constraint for it."""
    tables = dict(sorted(draw(_TABLES).items()))
    path = draw(st.sampled_from(_PATHS))
    name = draw(st.sampled_from([name for name in _NAMES if name not in tables.get(path, {})]))
    return _text(tables), path, name, draw(_SPECS)


def _declared(text: str, path: _Path, name: str) -> str | dict[str, str | bool]:
    """What `name` is declared as in the table at `path`, read by a TOML parser."""
    table = tomllib.loads(text)
    for key in path:
        table = table[key]
    return table[name]


def _has_only_a_new_value(*, before: str, after: str) -> bool:
    """Whether `after` is `before` with another value, its comment in its column.

    The comment stays put until the value outgrows it, then follows one space past the value.
    """
    if not after.startswith(before.partition("=")[0] + "="):
        return False
    comment = before.find("#")
    return comment < 0 or after.index("#") == max(
        comment, len(after.partition("#")[0].rstrip()) + 1
    )


@settings(max_examples=300)
@example(case=(_TABLE_STYLE, ("python", "deps"), "torch", ">=2.15"))
@example(case=(_SPLIT, ("on", "linux", "python", "deps"), "torch", ">=2.15, <3"))
@example(case=(_COLUMN, ("deps",), "numpy", ">=2.5.12"))
@example(case=(_COLUMN, ("deps",), "numpy", ">=2.4.6,<2.5,!=2.4.9"))
@example(case=(_COLUMN, ("deps",), "numpy", ">=2.4.6,<2.5,!=2.4.9,!=2.4.10"))
@example(case=(_COLUMN, ("deps",), "torch", ">=2.15"))
@given(case=_raisable())
def test_raising_a_requirement_changes_its_version_and_nothing_else_of_the_manifest(
    case: _Case,
) -> None:
    text, path, name, spec = case
    manifest = ManifestText(text)
    manifest.put(path, name, spec=spec)
    after = manifest.text()
    moved = [
        (old, new)
        for old, new in zip(text.splitlines(), after.splitlines(), strict=True)
        if old != new
    ]
    before = _declared(text, path, name)

    assert manifest.constraint(path, name) == spec
    assert _declared(after, path, name) == (
        {**before, "version": spec} if isinstance(before, dict) else spec
    )
    assert len(moved) <= 1 and all(
        _has_only_a_new_value(before=old, after=new) for old, new in moved
    )


@settings(max_examples=300)
@example(case=(_TOGETHER, ("on", "unix", "python", "deps"), "httpx", ">=0.28"))
@example(case=(_SPLIT, ("on", "linux", "dev", "deps"), "httpx", ">=0.28"))
@given(case=_declarable())
def test_dropping_a_requirement_undoes_the_put_that_declared_it(case: _Case) -> None:
    text, path, name, spec = case
    manifest = ManifestText(text)
    manifest.put(path, name, spec=spec)

    assert manifest.declares(path, name) and manifest.constraint(path, name) == spec
    manifest.drop(path, name)
    assert manifest.text() == text and not manifest.declares(path, name)


def test_dropping_the_last_requirement_of_a_table_under_interleaved_parents_removes_it() -> None:
    manifest = ManifestText(_SPLIT)

    manifest.drop(("on", "linux", "python", "deps"), "torch")

    assert tomllib.loads(manifest.text()) == {
        "on": {"linux": {"deps": {"cuda": ">=13.0"}}, "unix": {"deps": {"git": ">=2"}}}
    }


@pytest.mark.parametrize(
    ("text", "path", "expected"),
    [
        pytest.param(
            _HEADED,
            ("deps",),
            _APPENDED,
            id="beneath-the-last-requirement-padded-to-the-column",
        ),
        pytest.param(
            '[workspace]\nname = "it"\n',
            ("on", "linux", "python", "deps"),
            '[workspace]\nname = "it"\n\n[on.linux.python.deps]\nhttpx = ">=0.28"\n',
            id="in-a-table-of-its-own-written-as-one-heading",
        ),
    ],
)
def test_a_new_requirement_lands_where_its_neighbours_would_have_it(
    text: str, path: _Path, expected: str
) -> None:
    manifest = ManifestText(text)

    manifest.put(path, "httpx", spec=">=0.28")

    assert manifest.text() == expected


@pytest.mark.parametrize(
    ("declaration", "constraint", "versioned"),
    [
        pytest.param('">=2.4.6,<2.5"', ">=2.4.6,<2.5", True, id="a-range"),
        pytest.param('"*"', "*", True, id="any-release"),
        pytest.param('"==1.*"', "==1.*", True, id="a-wildcard-is-not-an-exact-pin"),
        pytest.param('"==1.9.0"', "==1.9.0", False, id="an-exact-pin"),
        pytest.param('{ version = ">=2" }', ">=2", True, id="a-table-with-a-range"),
        pytest.param(
            '{ version = "==2.14.0" }', "==2.14.0", False, id="a-table-with-an-exact-pin"
        ),
        pytest.param(
            '{ version = ">=2", index = "https://example.invalid" }',
            ">=2",
            True,
            id="a-table-with-an-index-still-has-a-range",
        ),
        pytest.param(
            '{ version = "0.65.1", url = "https://example.invalid/lint.zip", sha256 = "00" }',
            "0.65.1",
            False,
            id="a-url-source-names-its-release-itself",
        ),
        pytest.param(
            '{ path = "packages/local", editable = true }',
            '{ path = "packages/local", editable = true }',
            False,
            id="a-path-source",
        ),
        pytest.param(
            '{ git = "https://example.invalid/r.git", rev = "abc" }',
            '{ git = "https://example.invalid/r.git", rev = "abc" }',
            False,
            id="a-git-source",
        ),
    ],
)
def test_only_a_requirement_naming_a_range_is_versioned_for_raising(
    declaration: str, constraint: str, versioned: bool
) -> None:
    manifest = ManifestText(f"[python.deps]\nx = {declaration}\n")
    path = ("python", "deps")

    assert manifest.constraint(path, "x") == constraint
    assert manifest.versioned(path, "x") is versioned
    assert not manifest.versioned(path, "absent") and not manifest.declares(path, "absent")


@pytest.mark.parametrize(
    "declaration", ['python = "scalar"', 'python = ["array"]', "python = { deps = {} }"]
)
@pytest.mark.parametrize("create", [False, True])
def test_an_invalid_requirement_parent_is_refused_without_mutation(
    declaration: str, create: bool
) -> None:
    manifest = ManifestText(declaration + "\n")
    before = manifest.text()

    with pytest.raises(MissionError, match=r"\[python\] is not a table of requirements"):
        manifest.table(("python", "deps"), create=create)

    assert manifest.text() == before
