from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError, Project
from mainboard.core.membership import Membership

from ..strategies import WORDS

_MANIFEST = Project().manifest


def _project(directory: Path, marker: str) -> Path:
    """`directory` made a project by holding `marker`, answering it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / marker).write_text("", encoding="utf-8")
    return directory


def test_members_are_the_matched_projects_no_exclusion_leaves_out(tmp_path: Path) -> None:
    """A glob picks only the directories holding a manifest or a pyproject, never the root."""
    _project(tmp_path, _MANIFEST)
    _project(tmp_path / "packages" / "lib", "pyproject.toml")
    _project(tmp_path / "packages" / "retired", "pyproject.toml")
    _project(tmp_path / "research" / "head", _MANIFEST)
    (tmp_path / "research" / "notes").mkdir()
    (tmp_path / "research" / "README.md").write_text("", encoding="utf-8")
    membership = Membership(tmp_path, ["packages/*", "research/*", "!packages/retired"], _MANIFEST)

    assert membership.directories() == ["packages/lib", "research/head"]
    assert "." not in Membership(tmp_path, ["**"], _MANIFEST).directories()
    assert membership.claims(tmp_path / "research" / "head")
    assert not membership.claims(tmp_path / "packages" / "retired")
    assert not membership.claims(tmp_path / "research" / "notes")
    assert not membership.claims(tmp_path.parent)


@given(names=st.lists(WORDS, min_size=1, max_size=4, unique=True))
def test_every_listed_directory_claims_itself(names: list[str]) -> None:
    """What `directories` lists, `claims` agrees with, whatever the names."""
    root = Path(__file__).parent
    membership = Membership(root, names, _MANIFEST)
    assert all(membership.claims(root / path) for path in membership.directories())


def test_declared_reads_only_the_workspace_members_and_refuses_broken_toml(
    tmp_path: Path,
) -> None:
    (tmp_path / _MANIFEST).write_text('[workspace]\nmembers = ["a"]\n', encoding="utf-8")
    assert Membership.declared(tmp_path, _MANIFEST).included == ["a"]
    (tmp_path / _MANIFEST).write_text("[tasks]\n", encoding="utf-8")
    assert Membership.declared(tmp_path, _MANIFEST).included == []
    (tmp_path / _MANIFEST).write_text("members = [", encoding="utf-8")
    with pytest.raises(MissionError, match="not valid TOML"):
        Membership.declared(tmp_path, _MANIFEST)


def test_inside_a_member_the_workspace_composing_it_is_the_root(tmp_path: Path) -> None:
    """The way cargo finds its workspace: a claimed member defers, an unclaimed one stands."""
    project = Project()
    (tmp_path / _MANIFEST).write_text('[workspace]\nmembers = ["research/*"]\n', "utf-8")
    member = _project(tmp_path / "research" / "head", _MANIFEST)
    stray = _project(tmp_path / "scratch" / "tool", _MANIFEST)
    (member / "src").mkdir()

    assert project.find_root(member / "src") == tmp_path
    assert project.find_root(stray) == stray
    (tmp_path / _MANIFEST).write_text("[tasks]\n", encoding="utf-8")
    assert project.find_root(member) == member
