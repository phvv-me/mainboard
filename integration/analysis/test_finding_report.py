"""Compact reports keep every value of a finding, and only what is new reaches the reader.

`Seen` counts a finding by its rule, its message with the varying values held out, and its line's
text (in prose, the names it quotes), never by its path: a file or block that moves, a paragraph
that reflows or a count that changes brings the same findings along, and only what was added is
new.
"""

from collections.abc import Mapping
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.analysis.commands.gate import Options, gate
from mainboard.analysis.engine.lint.seen import Seen
from mainboard.analysis.findings import Report
from mainboard.analysis.rules.python.libraries.logging import TextAroundTheLogger
from mainboard.git import Git

# Where findings may sit: a Python line anchors by its text, a markdown line by the names quoted.
_FILES = ("a.py", "pkg/b.py", "c.md", "pkg/deep/d.py")
_LOGGING = (TextAroundTheLogger.id,)

type Layout = dict[str, list[tuple[str, str, int]]]


def _committed(root: Path, files: Mapping[str, str]) -> Git:
    """A repository at `root` holding `files` in one commit."""
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    git = Git(root)
    git.out("init", "--quiet")
    git.out("add", "-A")
    git.out("-c", "user.name=it", "-c", "user.email=it@example.invalid", "commit", "-qm", "one")
    return git


@st.composite
def _layouts(draw: st.DrawFn, names: list[tuple[str, str]]) -> Layout:
    """`names` (rule, name) spread over `_FILES` in any order, each with a count of its own."""
    layout: Layout = {}
    for rule, name in draw(st.permutations(names)):
        file = draw(st.sampled_from(_FILES))
        layout.setdefault(file, []).append((rule, name, draw(st.integers(1, 99))))
    return layout


@st.composite
def _moves(draw: st.DrawFn) -> tuple[Layout, Layout, int]:
    """Findings laid out, then laid out again elsewhere and recounted, with some added."""
    rules = draw(st.lists(st.sampled_from(["R1", "R2"]), min_size=1, max_size=8))
    names = [(rule, f"name_{i}") for i, rule in enumerate(rules)]
    added = [("R1", f"added_{i}") for i in range(draw(st.integers(0, 3)))]
    return draw(_layouts(names)), draw(_layouts(names + added)), len(added)


def _laid(root: Path, layout: Layout, *, wrap: str) -> Report:
    """Write `layout` under `root`, removing the files it leaves out, and report its findings.

    A markdown line is wrapped by `wrap`, as a reflowed paragraph rewraps it.
    """
    for file in _FILES:
        (root / file).unlink(missing_ok=True)
    found: dict[str, list[tuple[str, int, int, str]]] = {"R1": [], "R2": []}
    for file, findings in layout.items():
        lines = [
            f"{wrap} {name} {wrap}" if file.endswith(".md") else name for _, name, _ in findings
        ]
        (root / file).parent.mkdir(parents=True, exist_ok=True)
        (root / file).write_text("\n".join(lines) + "\n", encoding="utf-8")
        for line, (rule, name, count) in enumerate(findings, 1):
            found[rule].append((file, line, 0, f"`{name}` holds {count} statements"))
    return Report(found=found)


@settings(max_examples=60)
@given(move=_moves())
@example(move=({"a.py": [("R1", "name_0", 41)]}, {"pkg/b.py": [("R1", "name_0", 42)]}, 0))
@example(
    move=(
        {"c.md": [("R2", "name_0", 3)]},
        {"c.md": [("R2", "name_0", 3)], "a.py": [("R1", "added_0", 1)]},
        1,
    )
)
def test_findings_that_move_reflow_or_recount_stay_seen_and_only_added_ones_are_new(
    tmp_path_factory: pytest.TempPathFactory, move: tuple[Layout, Layout, int]
) -> None:
    root = tmp_path_factory.mktemp("seen")
    before, after, added = move
    Seen(root).fresh(_laid(root, before, wrap="first"), [*before])

    fresh = Seen(root).fresh(_laid(root, after, wrap="then rewrapped"), [*after])

    assert fresh.total == added


def test_a_file_moved_while_other_files_are_checked_is_matched_when_it_is_next_checked(
    tmp_path: Path,
) -> None:
    rows = [("old.py", 1, 0, "`moved_line` is long"), ("old.py", 2, 0, "`other_line` is long")]
    (tmp_path / "old.py").write_text("moved_line\nother_line\n", encoding="utf-8")
    (tmp_path / "else.py").write_text("", encoding="utf-8")
    Seen(tmp_path).fresh(Report(found={"R1": rows}), ["old.py", "else.py"])
    (tmp_path / "old.py").rename(tmp_path / "new.py")
    Seen(tmp_path).fresh(Report(found={"R1": []}), ["else.py"])

    fresh = Seen(tmp_path).fresh(
        Report(found={"R1": [("new.py", *r[1:]) for r in rows]}), ["new.py"]
    )

    assert fresh.total == 0


def test_a_file_another_session_creates_is_recorded_by_a_check_and_never_reported(
    tmp_path: Path,
) -> None:
    _committed(tmp_path, {"README": "one\n"})
    Seen(tmp_path).fresh(Report(found={"R1": []}), ["README"])
    (tmp_path / "theirs.py").write_text("their_line\n", encoding="utf-8")
    theirs = Report(found={"R1": [("theirs.py", 1, 0, "`their_line` is long")]})

    first, second = (Seen(tmp_path).fresh(theirs, files) for files in (["README"], ["theirs.py"]))

    assert (first.total, second.total) == (0, 0)


def test_a_gate_after_git_mv_reports_only_the_line_the_moved_file_gained(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    git = _committed(tmp_path, {"pyproject.toml": "[tool.mcmr]\n", "old.py": 'print("one")\n'})
    gate(Path("old.py"), root=tmp_path, options=Options(rule=_LOGGING, new=True))
    git.out("mv", "old.py", "new.py")
    (tmp_path / "new.py").write_text('print("one")\nprint("two")\n', encoding="utf-8")

    code = gate(Path("new.py"), root=tmp_path, options=Options(rule=_LOGGING, new=True))

    shown = capsys.readouterr().out
    assert code == 1 and " new.py 2" in shown and " new.py 1" not in shown


@pytest.mark.parametrize(
    "url",
    [
        "https://developer.apple.com/videos/play/wwdc2026/274/",
        "https://example.com/27/274?count=3#section-2",
        "http://localhost:8080/123",
    ],
)
def test_grouped_findings_preserve_links_and_variable_values(url: str) -> None:
    report = Report(
        found={
            "SW-MIG0006": [
                ("B.swift", 28, 0, f"Review `Record.value` with 15 entries; see {url}"),
                ("A.swift", 4, 0, f"Review `Other.value` with 3.5 entries; see {url}"),
            ],
            "EMPTY": [],
        }
    )

    assert report.grouped == (
        f"SW-MIG0006 Review _ with _ entries; see {url}\n"
        " A.swift 4[Other.value|3.5]\n"
        " B.swift 28[Record.value|15]"
    )
    assert report.total == 2
    assert report.counts == "    2 SW-MIG0006\n    2 findings, 1 rules"


def test_distinct_documentation_targets_do_not_merge() -> None:
    first = "https://developer.apple.com/videos/play/wwdc2026/274/"
    second = "https://developer.apple.com/videos/play/wwdc2026/275/"
    report = Report(
        found={
            "SW-MIG0006": [
                ("A.swift", 1, 0, f"See {first}"),
                ("A.swift", 2, 0, f"See {second}"),
            ]
        }
    )

    assert report.grouped == (
        f"SW-MIG0006 See {first}\n A.swift 1\nSW-MIG0006 See {second}\n A.swift 2"
    )
