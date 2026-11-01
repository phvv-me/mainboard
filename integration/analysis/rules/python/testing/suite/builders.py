from collections.abc import Sequence

_HYPOTHESIS = """\
from hypothesis import given
from hypothesis import strategies as st


"""
_RANGE = """\
import pytest


@pytest.mark.parametrize('n', range(20))
def test_range(n):
    assert n >= 0
"""
_RANGE_FIXED = (
    _HYPOTHESIS
    + """\
@given(n=st.integers(0, 19))
def test_range(n):
    assert n >= 0
"""
)
_BASIC = """\
import pytest


@pytest.mark.parametrize('n', range(20))
def test_one(n):
    assert n >= 0


@pytest.mark.parametrize('size', range(5, 30))
@pytest.mark.slow
def test_two(size):
    assert size
"""
_BASIC_FIXED = (
    "import pytest\n"
    + _HYPOTHESIS
    + """\
@given(n=st.integers(0, 19))
def test_one(n):
    assert n >= 0


@given(size=st.integers(5, 29))
@pytest.mark.slow
def test_two(size):
    assert size
"""
)
_SPAN = """\
import pytest


class TestSpan:
    @pytest.mark.parametrize('size', range(5, 30))
    def test_size(self, size):
        assert size
"""
_SPAN_FIXED = (
    _HYPOTHESIS
    + """\
class TestSpan:
    @given(size=st.integers(5, 29))
    def test_size(self, size):
        assert size
"""
)
_OTHER = """\
@given(st.integers())
def test_other(x):
    assert x == x


"""
_PRESENT = "import pytest\n" + _HYPOTHESIS + _OTHER + _RANGE.split("\n\n\n", 1)[1]
_PRESENT_FIXED = _HYPOTHESIS + _OTHER + _RANGE_FIXED.split("\n\n\n", 1)[1]
_IMPORTED = _RANGE.replace("import pytest", "from pytest import mark").replace(
    "@pytest.mark", "@mark"
)
_TUPLE = _RANGE.replace("'n'", "'a,b'").replace("test_range(n)", "test_range(a, b)")
_IDS = _RANGE.replace("range(20))", "range(20), ids=str)")
_NOTED = _RANGE.replace(
    "('n', range(20))",
    """(
    'n',
    range(20),  # every value
)""",
)
_NAMES = _RANGE.replace("\n\n\n@", "\n\nst = 3\n\n\n@")
_FIXTURE = _RANGE.replace("test_range(n)", "test_range(tmp_path, n)")
_SMALL = _RANGE.replace("range(20)", "range(5)")
_MIXED = (
    _RANGE
    + "\n\n"
    + _RANGE.split("\n\n\n", 1)[1].replace("test_range(n)", "test_bad(tmp_path, n)")
)


def wide_ranges() -> dict[str, tuple[str, str]]:
    """The files the PY-TEST0014 fix reads, by project path, each with the text it must leave.

    A file the fix must refuse is its own expectation.
    """
    return {
        "test_basic.py": (_BASIC, _BASIC_FIXED),
        "test_method.py": (_SPAN, _SPAN_FIXED),
        "test_present.py": (_PRESENT, _PRESENT_FIXED),
        "test_carriage_returns.py": (
            _RANGE.replace("\n", "\r\n"),
            _RANGE_FIXED.replace("\n", "\r\n"),
        ),
        "test_imported.py": (_IMPORTED, _RANGE_FIXED),
        "test_tuple.py": (_TUPLE, _TUPLE),
        "test_ids.py": (_IDS, _IDS),
        "test_noted.py": (_NOTED, _NOTED),
        "test_names.py": (_NAMES, _NAMES),
        "test_fixture.py": (_FIXTURE, _FIXTURE),
        "test_small.py": (_SMALL, _SMALL),
        "test_mixed.py": (_MIXED, _MIXED),
    }


def ranged(bounds: Sequence[tuple[int, int]]) -> str:
    """A module of one parametrized test per (low, high) range."""
    tests = [
        f"@pytest.mark.parametrize('n', range({low}, {high}))\n"
        f"def test_{place}(n):\n    assert n is not None\n"
        for place, (low, high) in enumerate(bounds)
    ]
    return "import pytest\n\n\n" + "\n\n".join(tests)


def drawn(bounds: Sequence[tuple[int, int]]) -> str:
    """The module `ranged` becomes: one Hypothesis integer per range, ends included."""
    tests = [
        f"@given(n=st.integers({low}, {high - 1}))\n"
        f"def test_{place}(n):\n    assert n is not None\n"
        for place, (low, high) in enumerate(bounds)
    ]
    return _HYPOTHESIS + "\n\n".join(tests)


_ALIAS = """\
def test_write(tmpdir):
    p = tmpdir.join('x')
    p.write('data')
    assert p.read() == 'data'
    assert p.strpath
    assert p.exists()
"""
_ALIAS_FIXED = """\
def test_write(tmp_path):
    p = tmp_path / 'x'
    p.write_text('data')
    assert p.read_text() == 'data'
    assert str(p)
    assert p.exists()
"""
_CHAIN = """\
def test_chain(tmpdir):
    tmpdir.join('y').write('z')
    assert tmpdir.join('y').read() == 'z'
    assert str(tmpdir.join('v')) and tmpdir.strpath
    assert tmpdir.join('w').strpath
"""
_CHAIN_FIXED = """\
def test_chain(tmp_path):
    (tmp_path / 'y').write_text('z')
    assert (tmp_path / 'y').read_text() == 'z'
    assert str(tmp_path / 'v') and str(tmp_path)
    assert str(tmp_path / 'w')
"""
_CRLF = "def test_crlf(tmpdir):\n    assert str(tmpdir) and tmpdir.strpath\n"
_CRLF_FIXED = "def test_crlf(tmp_path):\n    assert str(tmp_path) and str(tmp_path)\n"
_TWO = """\
def test_one(tmpdir):
    assert tmpdir.strpath


def test_two(tmpdir):
    q = tmpdir.join('q')
    q.write('1')
"""
_TWO_FIXED = """\
def test_one(tmp_path):
    assert str(tmp_path)


def test_two(tmp_path):
    q = tmp_path / 'q'
    q.write_text('1')
"""
_CLOSURE = """\
def test_closure(tmpdir):
    def helper():
        return str(tmpdir)

    assert helper()
"""
_FIXTURE_HOME = """\
import pytest


@pytest.fixture
def workdir(tmpdir):
    return str(tmpdir)


def test_it(workdir):
    assert workdir
"""
_METHOD_DIR = """\
class TestDir:
    def test_it(self, tmpdir):
        assert str(tmpdir)
"""
_ENSURE = "def test_ensure(tmpdir):\n    tmpdir.ensure('x')\n"
_BARE = "def test_bare(tmpdir):\n    helper(tmpdir)\n"
_REMOVE = """\
def test_remove(tmpdir):
    p = tmpdir.join('x')
    p.remove()
"""
_FACTORY = """\
def test_dir(tmpdir):
    assert str(tmpdir)


def test_factory(tmpdir_factory):
    assert tmpdir_factory
"""
_NAMED = "def test_named(tmpdir, tmp_path):\n    assert str(tmpdir) and tmp_path\n"
_ANNOTATED = "def test_annotated(tmpdir: Local):\n    assert str(tmpdir)\n"
_TWO_ARGUMENTS = "def test_two_arguments(tmpdir):\n    assert str(tmpdir.join('a', 'b'))\n"
_SUM = "def test_sum(tmpdir, name):\n    assert str(tmpdir.join('a' + name))\n"
_MODE = "def test_mode(tmpdir):\n    tmpdir.join('a').write(b'x', 'wb')\n"
_RETURNED = """\
import pytest


@pytest.fixture
def workdir(tmpdir):
    return tmpdir.join('w')


def test_it(workdir):
    assert workdir
"""
_REASSIGNED = """\
def test_reassigned(tmpdir):
    p = tmpdir.join('x')
    p = other()
    p.write('d')
"""


def legacy_tmpdirs() -> dict[str, tuple[str, str]]:
    """The files the PY-TEST0002 fix reads, by project path, each with the text it must leave.

    A file the fix must refuse is its own expectation.
    """
    return {
        "test_alias.py": (_ALIAS, _ALIAS_FIXED),
        "test_chain.py": (_CHAIN, _CHAIN_FIXED),
        "test_carriage_returns.py": (
            _CRLF.replace("\n", "\r\n"),
            _CRLF_FIXED.replace("\n", "\r\n"),
        ),
        "test_two.py": (_TWO, _TWO_FIXED),
        "test_closure.py": (_CLOSURE, _CLOSURE.replace("tmpdir", "tmp_path")),
        "test_fixture_home.py": (_FIXTURE_HOME, _FIXTURE_HOME.replace("tmpdir", "tmp_path")),
        "test_method_dir.py": (_METHOD_DIR, _METHOD_DIR.replace("tmpdir", "tmp_path")),
        "test_ensure.py": (_ENSURE, _ENSURE),
        "test_bare.py": (_BARE, _BARE),
        "test_remove.py": (_REMOVE, _REMOVE),
        "test_factory.py": (_FACTORY, _FACTORY),
        "test_named.py": (_NAMED, _NAMED),
        "test_annotated.py": (_ANNOTATED, _ANNOTATED),
        "test_two_arguments.py": (_TWO_ARGUMENTS, _TWO_ARGUMENTS),
        "test_sum.py": (_SUM, _SUM),
        "test_mode.py": (_MODE, _MODE),
        "test_returned.py": (_RETURNED, _RETURNED),
        "test_reassigned.py": (_REASSIGNED, _REASSIGNED),
    }


def legacy_chain(names: Sequence[str]) -> str:
    """A test writing a file for each name through the py.path API of `tmpdir`."""
    lines = [f"    tmpdir.join('{name}').write('{name}')\n" for name in names]
    return "def test_files(tmpdir):\n" + "".join(lines) + "    assert tmpdir.strpath\n"


def modern_chain(names: Sequence[str]) -> str:
    """The test `legacy_chain` becomes, through `tmp_path`."""
    lines = [f"    (tmp_path / '{name}').write_text('{name}')\n" for name in names]
    return "def test_files(tmp_path):\n" + "".join(lines) + "    assert str(tmp_path)\n"
