from typing import TYPE_CHECKING

from hypothesis import given
from hypothesis import strategies as st

from mainboard.manuscript import Kind, Problem
from mainboard.manuscript.log import TexLog, _unwrapped

from .support import WRAP, manuscript, wrapped

if TYPE_CHECKING:
    from pathlib import Path


def logged(directory: Path) -> str:
    """One log carrying every shape the reader has to attribute, wrapped the way TeX wraps it."""
    lines = [
        ")",
        "This is XeTeX, a closing parenthesis above with nothing open",
        "(paper.tex",
        "Overfull \\hbox (1.0pt too wide) in paragraph at lines 9--2",
        "(article.cls (size10.clo) (Font) (3.1.9a)",
        "(./sections/intro.tex LaTeX Warning: Reference `fig:missing' on page 2 undefined on "
        "input line 12.",
        "Package natbib Warning: Citation `key99' on page 2 undefined on input line 13.",
        "LaTeX Warning: Citation `key98' on page 2 undefined on input line 14.",
        "Overfull \\hbox (12.5pt too wide) in paragraph at lines 20--21",
        "Overfull \\vbox (4.0pt too high) detected at line 22",
        ")",
        "(sections/conclusion [2]",
        "Overfull \\hbox (3.0pt too wide) in paragraph at lines 30--1",
        "LaTeX Warning: Label `sec:dup' multiply defined.",
        "LaTeX Warning: Label `sec:dup@cref' multiply defined.",
        "! Undefined control sequence.",
        "l.8 \\foo(bar)",
        "! Missing $ inserted.",
        f"(/elsewhere/x.sty) ({directory / 'sections' / 'abs.tex'}",
        "! Emergency stop.",
    ]
    return "\n".join(chunk for line in lines for chunk in wrapped(line))


def test_every_problem_is_read_once_and_pinned_to_the_file_and_line_it_came_from(
    tmp_path: Path,
) -> None:
    """The open-file stack follows the log across wrapped lines, anonymous parentheses and a
    paragraph whose line range runs backwards because TeX broke it after its file closed."""
    directory = manuscript(tmp_path)
    problems = TexLog(logged(directory), directory=directory).problems()
    assert problems == [
        Problem(kind=Kind.OVERFULL, where="paper.tex:9", detail="1.0pt too wide"),
        Problem(kind=Kind.REFERENCE, where="sections/intro.tex:12", detail="fig:missing"),
        Problem(kind=Kind.CITATION, where="sections/intro.tex:13", detail="key99"),
        Problem(kind=Kind.CITATION, where="sections/intro.tex:14", detail="key98"),
        Problem(kind=Kind.OVERFULL, where="sections/intro.tex:20", detail="12.5pt too wide"),
        Problem(kind=Kind.OVERFULL, where="sections/intro.tex:22", detail="4.0pt too high"),
        Problem(kind=Kind.OVERFULL, where="sections/intro.tex:30", detail="3.0pt too wide"),
        Problem(kind=Kind.LABEL, where="", detail="sec:dup"),
        Problem(
            kind=Kind.ERROR,
            where="sections/conclusion.tex:8",
            detail="Undefined control sequence.",
        ),
        Problem(kind=Kind.ERROR, where="sections/conclusion.tex", detail="Missing $ inserted."),
        Problem(kind=Kind.ERROR, where="sections/abs.tex", detail="Emergency stop."),
    ]


def test_a_path_outside_the_manuscript_is_shown_whole(tmp_path: Path) -> None:
    text = (
        "(/elsewhere/deep.tex\nLaTeX Warning: Reference `r' on page 1 undefined on input line 3."
    )
    [problem] = TexLog(text, directory=tmp_path).problems()
    assert problem.where == "/elsewhere/deep.tex:3"


def test_a_log_that_ends_on_a_full_width_line_keeps_it_and_an_early_error_names_no_place(
    tmp_path: Path,
) -> None:
    assert _unwrapped("x" * WRAP) == ["x" * WRAP]
    assert TexLog("", directory=tmp_path).problems() == []
    assert TexLog("! Oops.", directory=tmp_path).problems() == [
        Problem(kind=Kind.ERROR, where="", detail="Oops.")
    ]


# A log line: any text but the characters Python itself reads as a line break.
_LINE = st.text(st.characters(exclude_characters="\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"))


@given(st.lists(_LINE, max_size=6).filter(lambda lines: not lines or lines[-1]))
def test_unwrapping_undoes_texs_hard_wrap(lines: list[str]) -> None:
    """Whatever TeX broke at its column comes back as the line it wrote, whatever its length."""
    text = "\n".join(chunk for line in lines for chunk in wrapped(line))
    assert _unwrapped(text) == lines
