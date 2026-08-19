import pytest

from mainboard import MissionError
from mainboard.deps import ManifestText

_DEV_PYTHON = ("dev", "python", "deps")
_PYTHON = ("python", "deps")


def test_a_new_entry_lands_with_the_table_it_joins(text: str) -> None:
    """The comment introducing the next table stays above that table, not above the entry."""
    edited = ManifestText(text)
    edited.put(_DEV_PYTHON, "tqdm", ">=4.70.0, <5")
    written = edited.text().splitlines()
    at = written.index('tqdm      = ">=4.70.0, <5"')
    assert written[at - 1].startswith("pytest")
    assert (
        written[at + 2]
        == "# runtime-keyed toolchains, the table name matches the package in [deps]"
    )


def test_a_new_entry_lines_its_value_up_with_the_rest(text: str) -> None:
    """Alignment is read off the table rather than assumed, so a column stays a column."""
    edited = ManifestText(text)
    edited.put(_PYTHON, "tqdm", ">=4")
    written = edited.text()
    columns = {
        line.index("=") for line in written.splitlines() if line.startswith(("torch", "tqdm"))
    }
    assert len(columns) == 1


def test_adding_then_dropping_leaves_the_file_byte_identical(text: str) -> None:
    """The round trip is the proof that nothing but the one line was ever touched."""
    added = ManifestText(text)
    added.put(_DEV_PYTHON, "tqdm", ">=4.70.0, <5")
    restored = ManifestText(added.text())
    restored.drop(_DEV_PYTHON, "tqdm")
    assert restored.text() == text


def test_replacing_a_requirement_moves_only_its_value(text: str) -> None:
    """An entry already there keeps its own alignment, since only the constraint moved."""
    edited = ManifestText(text)
    edited.put(_PYTHON, "torch", ">=3.0")
    assert 'torch     = ">=3.0"' in edited.text()


def test_a_table_the_manifest_never_had_is_written_as_one_heading(text: str) -> None:
    """A new ecosystem reads as `[rust.deps]`, the way every other table is written."""
    edited = ManifestText(text)
    edited.put(("rust", "deps"), "ripgrep", ">=14, <15")
    written = edited.text()
    assert "[rust.deps]" in written
    assert 'ripgrep = ">=14, <15"' in written
    assert "\n[rust]\n" not in written


@pytest.mark.parametrize(
    ("name", "expected"),
    [("torch", ">=2.9"), ("lab-core", '{ path = "packages/lab-core", editable = true }')],
)
def test_constraint_reports_a_version_or_the_source_standing_in_for_one(
    text: str, name: str, expected: str
) -> None:
    """A requirement carrying a source has no range, so it answers with how it is written."""
    assert ManifestText(text).constraint(_PYTHON, name) == expected


@pytest.mark.parametrize(
    ("path", "name", "present"),
    [
        (_PYTHON, "torch", True),
        (_PYTHON, "absent", False),
        (("rust", "deps"), "torch", False),
        (("workspace", "name"), "torch", False),
    ],
)
def test_declares_answers_for_a_missing_key_table_or_branch(
    text: str, path: tuple[str, ...], name: str, present: bool
) -> None:
    """Asking about a table that is not there is a question with an answer, not a failure."""
    assert ManifestText(text).declares(path, name) is present


def test_reaching_a_table_the_manifest_lacks_names_the_heading_it_wanted(text: str) -> None:
    """The refusal spells the table so a caller reads which one was missing."""
    with pytest.raises(MissionError, match=r"\[rust\] is not in this manifest"):
        ManifestText(text).table(("rust", "deps"))


def test_reaching_through_something_that_is_not_a_table_refuses(text: str) -> None:
    """A key path landing on a value rather than a table is a mistake worth naming."""
    with pytest.raises(MissionError, match="not a table of requirements"):
        ManifestText(text).table(("workspace", "name"))
