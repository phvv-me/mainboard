# `mainboard center paper`: build a declared manuscript and say everything a reviewer would catch
# before a deadline, in one pass that exits nonzero on any of it. Errors, unresolved references
# and citations, labels defined twice, overfull boxes, the page count, where each section
# starts, and whether the section that closes the main text ends inside the venue's limit.
#
# The engine is tectonic and the page tools are poppler's, both run through the workspace
# environment the way any task is, so the build a check reads is the one the manuscript's own
# lock pins rather than whichever TeX a machine happens to carry.

import re
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from .layout import Aux, Section, Source, SyncTex
from .log import Kind, Problem, TexLog

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..engines.compile.backend.result import CommandResult
    from ..manifest.schema.paper import Paper

    type Runner = Callable[[Sequence[str]], CommandResult]

# A word broken across a line on the page, whatever padding the layout put after its hyphen.
_HYPHENATED = re.compile(r"-[ \t]*\n")


class Report(FrozenModel):
    """One build of one manuscript, and everything wrong with it.

    paper: the declared manuscript's name.
    pdf: where the build wrote the PDF.
    pages: how many pages it has, 0 when the build produced none.
    limit: the declared page limit, 0 when the manuscript declares none.
    ends: the section that must end within the limit, empty when none is checked.
    ends_on: the last page that section reaches, 0 when it was not measured.
    sections: every top-level section with the page it starts on.
    problems: everything the build and the check found wrong.
    """

    paper: str
    pdf: str
    pages: int = 0
    limit: int = 0
    ends: str = ""
    ends_on: int = 0
    sections: tuple[Section, ...] = ()
    problems: tuple[Problem, ...] = ()


class Manuscript:
    """One declared manuscript: built on demand, checked against its rule, shown by the page.

    name: the `[papers.<name>]` key.
    paper: the declaration, its directory, root file and page rule.
    root: the workspace root the declared directory is relative to.
    run: runs one argv inside the workspace environment and returns what it printed.
    """

    def __init__(self, name: str, paper: Paper, *, root: Path, run: Runner) -> None:
        self.name = name
        self.paper = paper
        self.directory = root / paper.dir
        self.main = self.directory / paper.main
        self.build = self.directory / "build"
        self.stem = Path(paper.main).stem
        self.run = run

    @property
    def pdf(self) -> Path:
        """Where the build writes the PDF."""
        return self.build / f"{self.stem}.pdf"

    def check(self) -> Report:
        """Build the manuscript, then read the log, the page layout and the rule into a report.

        A build that failed is reported by its errors alone. Its layout files are the previous
        build's, and the warnings of a run that stopped halfway are a first pass's, every
        citation unresolved because the bibliography never ran.
        """
        built = self._compile()
        problems = TexLog(self._read(".log"), directory=self.directory).problems()
        report = Report(paper=self.name, pdf=str(self.pdf), limit=self.paper.limit)
        if not built.succeeded:
            return report.model_copy(update={"problems": self._errors(problems, built)})
        aux = Aux(self._read(".aux"))
        ends_on, limited = self._limited(aux.pages)
        return report.model_copy(
            update={
                "pages": aux.pages,
                "ends": self.paper.ends,
                "ends_on": ends_on,
                "sections": tuple(aux.sections()),
                "problems": (*problems, *limited),
            }
        )

    def show(self, text: str, *, dpi: int) -> Path:
        """Render the first page whose text contains `text` to a PNG and return its path.

        Matching ignores whitespace, case and a word hyphenated across a line break, since that
        is how a phrase copied out of the source differs from the same phrase on the page.

        text: a phrase from the page to find.
        dpi: the rendering resolution.
        """
        extracted = self.run(["pdftotext", "-layout", str(self.pdf), "-"])
        if not extracted.succeeded:
            raise MissionError(f"pdftotext could not read {self.pdf}: {extracted.stderr.strip()}")
        wanted = _squeezed(text)
        page = next(
            (
                number
                for number, body in enumerate(extracted.stdout.split("\f"), start=1)
                if wanted in _squeezed(body)
            ),
            0,
        )
        if not page:
            raise MissionError(f"no page of {self.pdf.name} contains {text!r}")
        stem = self.build / f"{self.name}-page{page}"
        rendered = self.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", str(page), "-l", str(page)]
            + ["-singlefile", str(self.pdf), str(stem)]
        )
        if not rendered.succeeded:
            raise MissionError(f"pdftoppm could not render page {page}: {rendered.stderr.strip()}")
        return stem.with_suffix(".png")

    def _compile(self) -> CommandResult:
        """Run tectonic over the root file, keeping the log, the `.aux` and the SyncTeX map."""
        self.build.mkdir(parents=True, exist_ok=True)
        return self.run(
            ["tectonic", "--keep-logs", "--keep-intermediates", "--synctex"]
            + ["--outdir", str(self.build), str(self.main)]
        )

    @staticmethod
    def _errors(problems: Sequence[Problem], built: CommandResult) -> tuple[Problem, ...]:
        """The errors a failed build reports: the log's own, else the engine's last words."""
        logged = tuple(problem for problem in problems if problem.kind is Kind.ERROR)
        said = built.stderr.splitlines()
        return logged or tuple(
            Problem(kind=Kind.ERROR, detail=line) for line in said if line.startswith("error:")
        )

    def _limited(self, pages: int) -> tuple[int, list[Problem]]:
        """The last page of the closing section, and what the page rule makes of the build."""
        limit, ends = self.paper.limit, self.paper.ends
        if not limit:
            return 0, []
        if not ends:
            over = pages > limit
            return 0, [_over(f"{pages} pages, past the {limit}-page limit")] if over else []
        span = Source(self.main).span(ends)
        if not span:
            return 0, [_over(f"no section titled {ends!r} in the sources")]
        synctex = SyncTex.read(self.build / f"{self.stem}.synctex.gz", directory=self.directory)
        ends_on = synctex.last_page(span)
        if not ends_on:
            return 0, [_over(f"SyncTeX placed no line of {ends!r} on any page")]
        if ends_on > limit:
            past = f"{ends!r} ends on page {ends_on}, past the {limit}-page limit"
            return ends_on, [_over(past)]
        return ends_on, []

    def _read(self, suffix: str) -> str:
        """The build file with `suffix` beside the PDF, empty when the build wrote none."""
        try:
            return (self.build / f"{self.stem}{suffix}").read_text(
                encoding="utf-8", errors="replace"
            )
        except FileNotFoundError:
            return ""


def _over(detail: str) -> Problem:
    """A page-rule problem saying `detail`."""
    return Problem(kind=Kind.LIMIT, detail=detail)


def _squeezed(text: str) -> str:
    """`text` without whitespace, case or line-break hyphenation, for phrase matching."""
    return "".join(_HYPHENATED.sub("", text).split()).casefold()
