import gzip
from pathlib import Path
from typing import TYPE_CHECKING

from mainboard.engines.compile.backend.result import CommandResult

if TYPE_CHECKING:
    from collections.abc import Sequence

# How wide TeX writes a log line before it breaks it.
WRAP = 79

PAPER = r"""\documentclass{article}
\begin{document}
\input{sections/intro}
% \section{Conclusion} is a comment, not a heading
\input{sections/conclusion.tex}
\input{sections/missing}
\subsubsection*{Reproducibility statement}
Every number is regenerated from its record.
\bibliography{refs}
\appendix
\section{Extra}
\end{document}
"""

INTRO = "\\section{Introduction}\nHello.\n"
CONCLUSION = "\\section[Short]{The \\textsc{Gist}}\\label{sec:gist}\nWe conclude.\nMore.\n"

AUX = r"""\relax
\@writefile{toc}{\contentsline {section}{\numberline {1}Introduction}{1}{section.1}}
\@writefile{toc}{\contentsline {section}{\numberline {2}The \textsc  {Gist}}{2}{section.2}}
\@writefile{toc}{\contentsline {subsection}{\numberline {2.1}Inner}{2}{subsection.2.1}}
\@writefile{toc}{\contentsline {section}{References}{x}{section*.3}}
\@writefile{lof}{\contentsline {figure}{\numberline {1}{\ignorespaces A figure}}{3}{figure.1}}
\gdef \@abspage@last{12}
"""


def manuscript(root: Path) -> Path:
    """A small manuscript under `root/paper`: a root file, two inputs and one missing input."""
    directory = root / "paper"
    (directory / "sections").mkdir(parents=True)
    (directory / "paper.tex").write_text(PAPER)
    (directory / "sections" / "intro.tex").write_text(INTRO)
    (directory / "sections" / "conclusion.tex").write_text(CONCLUSION)
    return directory


def synctex(directory: Path, *, gist_page: int) -> str:
    """A SyncTeX map placing the introduction on page 1 and the gist's last line on `gist_page`.

    The back-matter statement lands on page 10, past any limit, and must never count as the
    gist's own text.
    """
    return "\n".join(
        [
            "SyncTeX Version:1",
            "Input:1:./paper.tex",
            f"Input:2:{directory / 'sections' / 'intro.tex'}",
            "Input:3:sections/conclusion.tex",
            "Output:pdf",
            "Content:",
            "h1,1:0,0:0,0,0",
            "{1",
            "[2,1:0,0:0,0,0",
            "h9,3:0,0:0,0,0",
            "}1",
            "{2",
            "(3,2:0,0:0,0,0",
            "}2",
            f"{{{gist_page}",
            "x3,3:0,0",
            f"}}{gist_page}",
            "{10",
            "h1,7:0,0:0,0,0",
            "}10",
        ]
    )


def wrapped(line: str) -> list[str]:
    """`line` broken the way TeX breaks a log line, a full last chunk followed by an empty one."""
    chunks = [line[start : start + WRAP] for start in range(0, len(line), WRAP)] or [""]
    return chunks + ([""] if line and len(line) % WRAP == 0 else [])


class Engine:
    """The three tools a manuscript check runs, answering from what a test configured.

    tectonic writes the configured log, `.aux` and SyncTeX map into its `--outdir` and exits as
    told; pdftotext answers `pages` joined by form feeds; pdftoppm answers as told.

    log: the log a build writes, None to write none at all.
    aux: the `.aux` a build writes.
    synctex: the decompressed SyncTeX map a build writes.
    failed: whether the build exits nonzero.
    stderr: what the build prints to stderr.
    pages: each page's text, for pdftotext.
    readable: whether pdftotext succeeds.
    renders: whether pdftoppm succeeds.
    """

    def __init__(
        self,
        *,
        log: str | None = "",
        aux: str = AUX,
        synctex: str = "",
        failed: bool = False,
        stderr: str = "",
        pages: Sequence[str] = (),
        readable: bool = True,
        renders: bool = True,
    ) -> None:
        self.log = log
        self.aux = aux
        self.synctex = synctex
        self.failed = failed
        self.stderr = stderr
        self.pages = pages
        self.readable = readable
        self.renders = renders
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> CommandResult:
        self.calls.append(list(argv))
        if argv[0] == "pdftotext":
            return CommandResult(0 if self.readable else 1, "\f".join(self.pages), "no pdf")
        if argv[0] == "pdftoppm":
            return CommandResult(0 if self.renders else 1, "", "bad page")
        build = Path(argv[argv.index("--outdir") + 1])
        stem = Path(argv[-1]).stem
        if self.log is not None:
            (build / f"{stem}.log").write_text(self.log)
        (build / f"{stem}.aux").write_text(self.aux)
        (build / f"{stem}.synctex.gz").write_bytes(gzip.compress(self.synctex.encode()))
        return CommandResult(1 if self.failed else 0, "", self.stderr)
