"""What a kept fix says it did to the ids of the tests pytest collects: the ids its change line
renames or deletes, and the files it moved. Every id it does not name stays, and the ones it gives
to tests are the ones it promises."""

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard.analysis.engine.fixing.components import Remap

from .builders import applied

_WORD = st.from_regex(r"[a-z]{2,6}", fullmatch=True)
_NAMES = st.lists(_WORD, min_size=2, max_size=6, unique=True)


@pytest.mark.parametrize(
    ("change", "ids"),
    [
        pytest.param(
            "parametrized t.py::test_f_one -> t.py::test_f[one], t.py::K::test_f_two -> "
            "t.py::K::test_f[two]",
            (
                ("t.py::test_f_one", "t.py::test_f[one]"),
                ("t.py::K::test_f_two", "t.py::K::test_f[two]"),
            ),
            id="a-merge-names-each-test-and-the-case-it-became",
        ),
        pytest.param(
            "parametrized test_f_one -> test_f[one], test_f_two -> test_f[two]",
            (("test_f_one", "test_f[one]"), ("test_f_two", "test_f[two]")),
            id="a-merge-names-each-test-by-its-function-and-the-case-it-became",
        ),
        pytest.param(
            "removed the duplicates t.py::test_b -> gone, t.py::K::test_c -> gone, "
            "kept t.py::test_a",
            (("t.py::test_b", None), ("t.py::K::test_c", None)),
            id="a-deletion-names-each-test-it-removes-and-not-the-one-it-kept",
        ),
        pytest.param(
            "turned the 1 wide ranges of `testfunctionfact:t.py` into Hypothesis integers: "
            "t.py::K::test_size[*] -> t.py::K::test_size",
            (("t.py::K::test_size[*]", "t.py::K::test_size"),),
            id="every-case-of-a-test-becoming-one-test",
        ),
        pytest.param(
            "rewrote 1 span (text around the logger)", (), id="a-change-that-names-no-test"
        ),
        pytest.param("moved a -> b", (), id="an-arrow-between-things-that-are-no-test-ids"),
        pytest.param("renamed latest -> newest", (), id="an-arrow-after-a-word-ending-in-test"),
    ],
)
def test_a_change_line_names_the_ids_it_renames_and_the_tests_it_deletes(
    change: str, ids: tuple[tuple[str, str | None], ...]
) -> None:
    remap = Remap.of(applied(change, files=["t.py"]))

    assert remap.ids == ids


def test_a_bare_name_speaks_of_the_files_of_the_fix_and_a_filed_id_of_its_own_file() -> None:
    filed = Remap.of(applied("removed u.py::test_b -> gone", files=["t.py", "u.py"]))
    bare = Remap.of(applied("parametrized test_f_one -> test_f[one]", files=["t.py", "u.py"]))

    assert (filed.files, bare.files) == ({"u.py"}, {"t.py", "u.py"})


@settings(max_examples=30)
@given(names=_NAMES)
def test_a_name_with_no_file_is_read_in_every_module_and_class_of_the_files_the_fix_changed(
    names: Sequence[str],
) -> None:
    holders = ("t.py", "t.py::K", "u.py")
    before = {f"{holder}::test_f_{name}" for holder in holders for name in names}
    change = ", ".join(f"test_f_{name} -> test_f[{name}]" for name in names)

    after = Remap.of(applied(change, files=["t.py"])).apply(before)

    merged = {f"{holder}::test_f[{name}]" for holder in holders[:2] for name in names}
    assert after == merged | {f"u.py::test_f_{name}" for name in names}


@settings(max_examples=30)
@given(names=_NAMES, others=_NAMES)
def test_a_merge_gives_each_test_one_case_and_leaves_the_other_ids_as_they_were(
    names: Sequence[str], others: Sequence[str]
) -> None:
    kept = {f"t.py::test_other_{name}" for name in others}
    before = {f"t.py::test_f_{name}" for name in names} | kept
    change = ", ".join(f"t.py::test_f_{name} -> t.py::test_f[{name}]" for name in names)

    after = Remap.of(applied(change, files=["t.py"])).apply(before)

    assert after == {f"t.py::test_f[{name}]" for name in names} | kept


@settings(max_examples=30)
@given(names=_NAMES, others=_NAMES)
def test_a_merge_promises_the_cases_it_names_and_no_id_it_leaves_as_it_was(
    names: Sequence[str], others: Sequence[str]
) -> None:
    kept = {f"t.py::test_other_{name}" for name in others}
    before = {f"t.py::test_f_{name}" for name in names} | kept
    change = ", ".join(f"t.py::test_f_{name} -> t.py::test_f[{name}]" for name in names)

    promised = Remap.of(applied(change, files=["t.py"])).promised(before)

    assert promised == {f"t.py::test_f[{name}]" for name in names}


@pytest.mark.parametrize(
    ("change", "moves", "before", "promised"),
    [
        pytest.param(
            "removed t.py::test_b -> gone",
            (),
            {"t.py::test_a", "t.py::test_b", "t.py::test_b[x]"},
            set(),
            id="a-deleted-test-promises-nothing",
        ),
        pytest.param(
            "turned t.py::test_size[*] -> t.py::test_size",
            (),
            {"t.py::test_size[a]", "t.py::test_size[b]", "t.py::test_other"},
            {"t.py::test_size"},
            id="every-case-of-a-test-becoming-one-test-promises-the-one-test",
        ),
        pytest.param(
            "moved a file",
            (("pkg", "lib"),),
            {"pkg/t.py::test_a[x]"},
            set(),
            id="a-moved-file-promises-nothing-of-its-ids",
        ),
        pytest.param(
            "parametrized t.py::test_f_one -> t.py::test_f[one]",
            (("t.py", "u.py"),),
            {"t.py::test_f_one", "t.py::test_g"},
            {"u.py::test_f[one]"},
            id="a-promised-id-follows-the-file-the-fix-moved",
        ),
    ],
)
def test_a_change_line_promises_only_the_ids_it_gives_to_tests(
    change: str,
    moves: tuple[tuple[str, str], ...],
    before: set[str],
    promised: set[str],
) -> None:
    remap = Remap.of(applied(change, files=["t.py", "pkg", "lib"], moves=moves))

    assert remap.promised(before) == promised


@pytest.mark.parametrize(
    ("node", "function"),
    [
        pytest.param("t.py::test_f", "t.py::test_f", id="a-test-with-no-case"),
        pytest.param("t.py::K::test_f[one]", "t.py::K::test_f", id="a-case-of-a-method"),
        pytest.param("t.py::test_f[a::b[c]]", "t.py::test_f", id="a-case-id-holding-brackets"),
    ],
)
def test_the_function_of_an_id_is_the_id_without_its_case(node: str, function: str) -> None:
    assert Remap.function(node) == function


@settings(max_examples=30)
@given(cases=st.lists(_WORD, min_size=1, max_size=4, unique=True), stem=_WORD)
def test_a_deleted_test_takes_its_cases_and_no_test_whose_name_starts_like_it(
    cases: Sequence[str], stem: str
) -> None:
    gone = f"t.py::test_{stem}"
    beside = f"t.py::test_{stem}_more"
    before = {gone, beside, *(f"{gone}[{case}]" for case in cases)}

    after = Remap.of(applied(f"removed {gone} -> gone", files=["t.py"])).apply(before)

    assert after == {beside}


@settings(max_examples=30)
@given(cases=st.lists(_WORD, min_size=1, max_size=8, unique=True), stem=_WORD)
def test_every_case_of_a_test_becomes_the_one_test_and_a_test_named_like_it_stays(
    cases: Sequence[str], stem: str
) -> None:
    wide = f"t.py::test_{stem}"
    beside = f"t.py::test_{stem}_more[x]"
    before = {beside, *(f"{wide}[{case}]" for case in cases)}

    after = Remap.of(applied(f"turned {wide}[*] -> {wide}", files=["t.py"])).apply(before)

    assert after == {wide, beside}


@pytest.mark.parametrize(
    ("moves", "before", "after"),
    [
        pytest.param(
            (("pkg/t.py", "lib/t.py"),),
            {"pkg/t.py::test_a", "pkg/u.py::test_b"},
            {"lib/t.py::test_a", "pkg/u.py::test_b"},
            id="a-file-the-fix-moved",
        ),
        pytest.param(
            (("pkg", "lib"),),
            {"pkg/t.py::K::test_a[x]", "pkg2/t.py::test_a"},
            {"lib/t.py::K::test_a[x]", "pkg2/t.py::test_a"},
            id="a-folder-the-fix-moved-and-a-folder-named-like-it",
        ),
    ],
)
def test_the_ids_of_a_moved_file_follow_it_and_the_others_stay(
    moves: tuple[tuple[str, str], ...], before: set[str], after: set[str]
) -> None:
    remap = Remap.of(applied("moved a file", files=["pkg", "lib"], moves=moves))

    assert remap.apply(before) == after
