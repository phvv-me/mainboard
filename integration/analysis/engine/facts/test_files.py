"""A `Listing` keeps what git says of a project while its fact base is derived, so the providers,
tools and rules that ask what the project holds cost one `git ls-files` pair and one
`git check-attr`. It lives only for the derivation: a fix writes new files between one fact base
and the next, so outside a listing, and after one exits, every call asks git again. A listing
opened inside another for the same root joins it, so a verb that writes nothing lists once for
its whole run."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.analysis.engine.facts import Listing, listed, project_files

from ....builders import git
from ..builders import written

_ONE_LISTING = ["ls-files", "ls-files", "check-attr"]
_THREADS = 8
_NAMES = st.lists(
    st.sampled_from(["a.py", "b/c.py", "b/d/e.txt", "f", ".hidden"]), unique=True, max_size=5
)


def test_inside_a_listing_a_project_is_listed_once_whoever_asks(
    repository: Path, git_verbs: Callable[[], list[str]]
) -> None:
    with Listing(repository):
        listed(repository)
        listed(repository)
        project_files(repository)

    assert git_verbs() == _ONE_LISTING


def test_a_listing_opened_inside_another_joins_it_and_leaves_it_open_when_it_exits(
    repository: Path, git_verbs: Callable[[], list[str]]
) -> None:
    with Listing(repository) as outer:
        with Listing(repository) as inner:
            listed(repository)
        after = Listing.of(repository)
        listed(repository)

    assert (inner is outer, after is outer, Listing.of(repository), git_verbs()) == (
        True,
        True,
        None,
        _ONE_LISTING,
    )


def test_inside_a_listing_each_caller_gets_a_copy_it_may_change(repository: Path) -> None:
    with Listing(repository):
        listed(repository).append("scribbled")
        project_files(repository).clear()
        asked = (listed(repository), project_files(repository))

    assert asked == (["README", "a/x.py"], ["README", "a/x.py"])


def test_inside_a_listing_threads_asking_together_share_one_git_run(
    repository: Path, git_verbs: Callable[[], list[str]]
) -> None:
    with Listing(repository), ThreadPoolExecutor(_THREADS) as pool:
        asked = list(pool.map(lambda _: listed(repository), range(_THREADS)))

    assert asked == [["README", "a/x.py"]] * _THREADS
    assert git_verbs() == _ONE_LISTING


def test_a_root_spelled_through_its_parent_is_served_from_the_listing(
    repository: Path, git_verbs: Callable[[], list[str]]
) -> None:
    root = repository / "a"
    with Listing(root):
        listed(root)
        spelled = listed(root / ".." / "a")

    assert spelled == ["x.py"]
    assert git_verbs() == _ONE_LISTING


def test_a_file_written_outside_any_listing_is_seen_by_the_next_call(repository: Path) -> None:
    before = listed(repository)

    written(repository, {"later.py": ""})

    assert listed(repository) == [*before, "later.py"]


def test_a_file_written_after_a_listing_exits_is_seen_by_the_next_call(repository: Path) -> None:
    with Listing(repository):
        before = listed(repository)

    written(repository, {"later.py": ""})

    assert listed(repository) == [*before, "later.py"]


def test_a_listing_that_raises_still_closes(repository: Path) -> None:
    with pytest.raises(RuntimeError), Listing(repository):
        raise RuntimeError

    assert Listing.of(repository) is None


@settings(max_examples=25)
@example(names=[])
@example(names=["b/c.py", "b/d/e.txt", ".hidden"])
@given(names=_NAMES)
def test_a_listing_answers_what_git_lists_outside_it(names: list[str]) -> None:
    with TemporaryDirectory() as folder:
        root = written(Path(folder), dict.fromkeys(names, ""))
        git(root, "init", "--quiet")
        outside = listed(root)
        with Listing(root):
            inside = listed(root)

    assert inside == outside
    assert sorted(outside) == sorted(names)
