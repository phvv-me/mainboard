import gzip
from typing import TYPE_CHECKING

import pytest

from mainboard.manuscript import Section
from mainboard.manuscript.layout import Aux, Source, SyncTex, plain

from .support import AUX, manuscript, synctex

if TYPE_CHECKING:
    from pathlib import Path


def test_the_aux_names_the_page_count_and_where_every_top_level_section_starts() -> None:
    """Numbered or not, markup stripped, and a page LaTeX wrote as something other than a
    number read as none rather than as a crash."""
    aux = Aux(AUX)
    assert aux.pages == 12
    assert aux.sections() == [
        Section(number="1", title="Introduction", page=1),
        Section(number="2", title="The Gist", page=2),
        Section(title="References", page=0),
    ]
    assert Aux("\\relax\n\\@writefile{toc}{\\contentsline {section}{Cut").sections() == [
        Section(title="", page=0)
    ]
    assert Aux("").pages == 0


@pytest.mark.parametrize(
    ("title", "lines"),
    [
        ("Introduction", {("intro.tex", 1), ("intro.tex", 2), ("paper.tex", 4), ("paper.tex", 5)}),
        (
            "the gist",
            {
                ("conclusion.tex", 1),
                ("conclusion.tex", 2),
                ("conclusion.tex", 3),
                ("paper.tex", 6),
            },
        ),
        ("Extra", {("paper.tex", 11), ("paper.tex", 12)}),
        ("Conclusion", set()),
    ],
    ids=[
        "up to the next heading, inputs followed",
        "up to an unnumbered statement, matched without markup or case",
        "to the end of the document",
        "a commented-out heading is no heading",
    ],
)
def test_a_section_spans_its_own_lines_in_reading_order(
    tmp_path: Path, title: str, lines: set[tuple[str, int]]
) -> None:
    source = Source(manuscript(tmp_path) / "paper.tex")
    assert {(path.name, number) for path, number in source.span(title)} == lines


def test_synctex_says_the_last_page_a_set_of_lines_reached(tmp_path: Path) -> None:
    """Records before the first page and records of an unknown input are dropped, and lines
    typeset nowhere reach no page."""
    directory = manuscript(tmp_path)
    map_file = tmp_path / "paper.synctex.gz"
    map_file.write_bytes(gzip.compress(synctex(directory, gist_page=9).encode()))
    mapped = SyncTex.read(map_file, directory=directory)
    gist = Source(directory / "paper.tex").span("The Gist")
    assert mapped.last_page(gist) == 9
    assert mapped.last_page(set()) == 0
    assert sorted(mapped.pages) == [1, 2, 9, 10]


def test_plain_text_drops_markup_and_collapses_space() -> None:
    assert plain("The~\\textbf{Big}  \\emph {Gist}") == "The Big Gist"
