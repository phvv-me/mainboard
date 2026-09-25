from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.manifest import Paper
from mainboard.manuscript import Kind, Manuscript, Problem

from .support import Engine, manuscript, synctex

if TYPE_CHECKING:
    from pathlib import Path

# One unresolved reference, as a build that otherwise succeeded logs it.
_WARNED = (
    "(./sections/intro.tex\nLaTeX Warning: Reference `r' on page 1 undefined on input line 2.\n)"
)


def checked(
    tmp_path: Path, tools: Engine, *, limit: int = 0, ends: str = "", gist: int = 9
) -> Manuscript:
    """A manuscript under `tmp_path` whose tools are `tools`, its gist ending on page `gist`."""
    directory = manuscript(tmp_path)
    tools.synctex = synctex(directory, gist_page=gist)
    return Manuscript("lab", Paper(dir="paper", limit=limit, ends=ends), root=tmp_path, run=tools)


@pytest.mark.parametrize(
    ("limit", "ends", "gist", "ends_on", "limited"),
    [
        (9, "The Gist", 9, 9, None),
        (9, "The Gist", 10, 10, "'The Gist' ends on page 10, past the 9-page limit"),
        (0, "The Gist", 10, 0, None),
        (12, "", 10, 0, None),
        (11, "", 10, 0, "12 pages, past the 11-page limit"),
        (9, "Conclusion", 9, 0, "no section titled 'Conclusion' in the sources"),
        (9, "Extra", 9, 0, "SyncTeX placed no line of 'Extra' on any page"),
    ],
    ids=[
        "the closing section ends on the last allowed page",
        "the closing section spills one page over",
        "no page rule",
        "a whole-document limit the build meets",
        "a whole-document limit the build breaks",
        "a closing section the sources never declare",
        "a closing section typeset nowhere",
    ],
)
def test_a_build_reports_its_layout_and_what_the_page_rule_makes_of_it(
    tmp_path: Path, limit: int, ends: str, gist: int, ends_on: int, limited: str | None
) -> None:
    tools = Engine(log=_WARNED)
    report = checked(tmp_path, tools, limit=limit, ends=ends, gist=gist).check()
    assert (report.pages, report.ends_on, report.limit) == (12, ends_on, limit)
    assert [section.page for section in report.sections] == [1, 2, 0]
    assert report.problems[0] == Problem(
        kind=Kind.REFERENCE, where="sections/intro.tex:2", detail="r"
    )
    assert [problem.detail for problem in report.problems[1:]] == ([limited] if limited else [])
    [build] = tools.calls
    assert build[:4] == ["tectonic", "--keep-logs", "--keep-intermediates", "--synctex"]
    assert build[-1] == str(tmp_path / "paper" / "paper.tex")
    assert report.pdf == str(tmp_path / "paper" / "build" / "paper.pdf")


@pytest.mark.parametrize(
    ("log", "stderr", "errors"),
    [
        (
            "(paper.tex\n! Undefined control sequence.\nl.4 \\foo\n" + _WARNED,
            "error: halted",
            [Problem(kind=Kind.ERROR, where="paper.tex:4", detail="Undefined control sequence.")],
        ),
        (
            None,
            "note: running\nerror: the file sections/x.tex was not found\nerror: halted",
            [
                Problem(kind=Kind.ERROR, detail="error: the file sections/x.tex was not found"),
                Problem(kind=Kind.ERROR, detail="error: halted"),
            ],
        ),
    ],
    ids=["the log's own errors, its first-pass warnings dropped", "the engine's own last words"],
)
def test_a_failed_build_reports_its_errors_and_nothing_from_a_stale_layout(
    tmp_path: Path, log: str | None, stderr: str, errors: list[Problem]
) -> None:
    tools = Engine(log=log, stderr=stderr, failed=True)
    report = checked(tmp_path, tools, limit=9, ends="The Gist").check()
    assert list(report.problems) == errors
    assert (report.pages, report.sections) == (0, ())


def test_showing_a_phrase_renders_the_first_page_it_appears_on(tmp_path: Path) -> None:
    """Whitespace, case and a word hyphenated across a line never hide a phrase."""
    pages = ["Title page", "We conclude that the ap-  \nproach is   SOUND.", "approach is sound"]
    tools = Engine(pages=pages)
    shown = checked(tmp_path, tools).show("the approach\nis sound", dpi=90)
    assert shown == tmp_path / "paper" / "build" / "lab-page2.png"
    render = tools.calls[-1]
    assert render[:7] == ["pdftoppm", "-png", "-r", "90", "-f", "2", "-l"]
    assert render[-1] == str(shown.with_suffix(""))


@pytest.mark.parametrize(
    ("tools", "refusal"),
    [
        (Engine(readable=False), "pdftotext could not read"),
        (Engine(pages=["nothing here"]), "no page of paper.pdf contains"),
        (Engine(pages=["the phrase"], renders=False), "pdftoppm could not render page 1"),
    ],
    ids=["an unreadable pdf", "a phrase on no page", "a page that will not render"],
)
def test_showing_refuses_what_it_cannot_find_or_draw(
    tmp_path: Path, tools: Engine, refusal: str
) -> None:
    with pytest.raises(MissionError, match=refusal):
        checked(tmp_path, tools).show("the phrase", dpi=110)
