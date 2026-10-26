"""`prose`: the narrative paragraphs of a paper, for manuscript review.

The text is read line by line in SQL. LaTeX loses its preamble, comments, math (inline, display and
equation environments), floats (figures, tables, algorithms), bibliography and layout commands;
the text of formatting commands stays, and a citation or a reference stays as a `[cite:key]` or
`[ref:key]` marker. Markdown loses its front matter, code, tables and markup. A paragraph is the
text between blank lines, headings, list items and removed blocks, reported with the file and
line where it starts. Files are read in the order the document includes them (`\\input`,
`\\include`). There is no parser here: neither LaTeX nor Markdown prose needs a syntax tree.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import duckdb
from cyclopts import Parameter
from patos import FrozenModel

from ....core.errors import MissionError
from ....log import logger
from ..report import Report

_INCLUDE = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")
_COMMENT = re.compile(r"(?<!\\)%.*$")
_BODY = "\\begin{document}"
_ROOTS = ("main.tex", "paper.tex")
_NOT_PROSE = frozenset({"figures", "tikz", "assets", "template", "plots"})

# Environments whose content is not prose, and the commands that start a heading.
_BLOCKS = (
    "figure|table|tabular|tabularx|longtable|tikzpicture|algorithm|algorithmic|equation|align|"
    "alignat|gather|multline|eqnarray|displaymath|verbatim|lstlisting|minted|thebibliography|"
    "wrapfigure|wraptable|subfigure|listing|comment"
)
_HEADINGS = "chapter|section|subsection|subsubsection|paragraph"
_SETUP = (
    "newcommand|renewcommand|def|let|setlength|usepackage|bibliographystyle|bibliography|"
    "documentclass|pagestyle|input|include"
)

# Rewrites applied in order to a kept line, each a regular expression and its replacement.
_LATEX = (
    (r"\\([%&_#])", r"\1"),  # escaped punctuation
    (
        r"\\(?:cite|citep|citet|citeauthor|citeyear|parencite|textcite|autocite)\*?"
        r"(?:\[[^\]]*\])*\{([^}]*)\}",
        r"[cite:\1]",
    ),
    (r"\\href\{[^}]*\}\{([^}]*)\}", r"\1"),
    (r"\\[a-zA-Z]*[rR]ef\*?\{([^}]*)\}", r"[ref:\1]"),  # cref, eqref and a paper's own `\roadref`
    (r"\\(?:begin|end)\{[^}]*\}(?:\[[^\]]*\])?", " "),  # environment delimiters
    (
        r"\\(?:label|index|vspace|hspace|vskip|hskip|includegraphics|noindent|centering|par|"
        r"bigskip|medskip|smallskip|newpage|clearpage|linenumbers|nolinenumbers)\b\*?"
        r"(?:\[[^\]]*\])*(?:\{[^{}]*\})?",
        " ",
    ),  # layout commands and their arguments
    (r"\\item\b(?:\[[^\]]*\])?", " "),
    (rf"\\(?:{_HEADINGS})\b\*?(?:\[[^\]]*\])?", " "),  # a heading keeps its text
    (r"\\[a-zA-Z]+\*?", " "),  # any other command, keeping its argument's text
)

# Applied to a whole paragraph, since inline math runs over line breaks.
_FINISH = (
    (r"\\\(.*?\\\)|\$[^$]*\$", " "),  # inline math
    (r"[{}~\\$^]|<[^>]*>", " "),
)
_FINISH_MARKDOWN = ((r"\$[^\s$](?:[^$]*[^\s$])?\$", " "),)  # inline math, never a price
_MARKDOWN = (
    (r"^\s*(#+\s|[-*+]\s|\d+[.)]\s|>\s*)", ""),  # heading, list and quote markers
    (r"!\[[^\]]*\]\([^)]*\)", " "),  # images
    (r"\[([^\]]*)\]\([^)]*\)", r"\1"),  # links keep their text
    (r"\[\[([^\]|]*)(?:\|[^\]]*)?\]\]", r"\1"),  # wikilinks
    (r"`[^`]*\\[a-zA-Z][^`]*`", " "),  # a code span holding LaTeX is math
    (r"\*\*([^*]+)\*\*|`([^`]*)`", r"\1\2"),  # bold and code
    (r"<[^>]*>", " "),  # HTML
)


def _quoted(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _rewritten(expression: str, rewrites: Sequence[tuple[str, str]]) -> str:
    """`expression` passed through each rewrite in turn, as nested `regexp_replace` calls."""
    for pattern, replacement in rewrites:
        expression = (
            f"regexp_replace({expression}, {_quoted(pattern)}, {_quoted(replacement)}, 'g')"
        )
    return expression


# One row per line through the stages: strip comments; count what opens and closes a removed
# block; carry those counts down the file with window sums; mark each line skipped, heading or
# list item; clean the kept ones; number the blocks between breaks; join each block's lines and
# drop the inline math that may span them.
_PARAGRAPHS = rf"""
    WITH lines AS (
      SELECT position, path, generate_subscripts(parts, 1) AS line, unnest(parts) AS raw
      FROM (SELECT position, path, string_split(content, chr(10)) AS parts FROM manuscript)
    ), stripped AS (
      SELECT *, path LIKE '%.tex' AS tex,
             CASE WHEN path LIKE '%.tex' THEN regexp_replace(raw, '(^|[^\\])%.*$', '\1')
                  ELSE raw END AS text
      FROM lines
    ), counted AS (
      SELECT *,
             len(regexp_extract_all(text, '\\begin\{{(?:{_BLOCKS})\*?\}}'))
               + len(regexp_extract_all(text, '(?:^|[^\\])\\\[')) AS opens,
             len(regexp_extract_all(text, '\\end\{{(?:{_BLOCKS})\*?\}}'))
               + len(regexp_extract_all(text, '(?:^|[^\\])\\\]')) AS closes,
             len(regexp_extract_all(text, '\$\$')) AS dollars,
             regexp_matches(text, '\\begin\{{document\}}')::INT AS begins,
             regexp_matches(text, '\\end\{{document\}}')::INT AS ends,
             regexp_matches(text, '^\s*(```|~~~)') AS fence,
             raw = '---' AS dashes
      FROM stripped
    ), tracked AS (
      SELECT *, sum(opens - closes) OVER w AS depth, sum(dollars) OVER w AS dollar_total,
             sum(begins) OVER w AS began, sum(ends) OVER w AS ended,
             max(begins) OVER (PARTITION BY path) = 1 AS documented,
             sum(fence::INT) OVER w AS fences, sum(dashes::INT) OVER w AS dash_total,
             first(raw) OVER (PARTITION BY path ORDER BY line) = '---' AS front_matter
      FROM counted
      WINDOW w AS (PARTITION BY path ORDER BY line)
    ), kept AS (
      SELECT *,
             CASE WHEN tex THEN
                    depth - opens + closes > 0 OR opens > 0 OR closes > 0
                    OR dollars > 0 OR (dollar_total - dollars) % 2 = 1
                    OR (documented AND (began = 0 OR ended - ends > 0))
                    OR regexp_matches(text, '^\s*\\({_SETUP})\b')
                  ELSE fences % 2 = 1 OR fence OR regexp_matches(text, '^\s*[|]')
                    OR dollars > 0 OR (dollar_total - dollars) % 2 = 1
                    OR regexp_matches(text, '^\s*<!--')
                    OR (front_matter AND (dash_total = 1 OR (dashes AND dash_total = 2)))
             END AS skipped,
             regexp_matches(text, '^\s*(\\({_HEADINGS})\*?(\[[^\]]*\])?[{{]|#+\s)') AS heading,
             regexp_matches(text, '^\s*(\\item\b|[-*+]\s|[0-9]+[.)]\s)') AS item
      FROM tracked
    ), cleaned AS (
      SELECT position, path, line, skipped, heading, item,
             trim(regexp_replace(
               CASE WHEN tex THEN {_rewritten("text", _LATEX)}
                    ELSE {_rewritten("text", _MARKDOWN)} END,
               '\s+', ' ', 'g')) AS clean
      FROM kept
    ), grouped AS (
      SELECT *, sum(CASE WHEN skipped OR heading OR item OR clean = '' THEN 1 ELSE 0 END)
                  OVER (PARTITION BY path ORDER BY line) AS block
      FROM cleaned
    ), blocks AS (
      SELECT position, path, line, 'heading' AS kind, clean AS text
      FROM grouped WHERE heading AND NOT skipped AND clean <> ''
      UNION ALL
      SELECT position, path, min(line), 'paragraph', string_agg(clean, ' ' ORDER BY line)
      FROM grouped WHERE NOT skipped AND NOT heading AND clean <> ''
      GROUP BY position, path, block
    ), finished AS (
      SELECT position, path, line, kind,
             trim(regexp_replace(
               CASE WHEN path LIKE '%.tex' THEN {_rewritten("text", _FINISH)}
                    ELSE {_rewritten("text", _FINISH_MARKDOWN)} END,
               '\s+', ' ', 'g')) AS text
      FROM blocks
    )
    SELECT position, path, line, kind, text FROM finished
    WHERE text <> '' AND (kind = 'heading' OR len(string_split(text, ' ')) >= 3)
    ORDER BY position, line
"""


class Paragraph(FrozenModel):
    """A paragraph or heading and where it starts."""

    path: str
    line: int
    kind: str
    text: str


class Prose(Report):
    """The narrative text of a paper in reading order."""

    paper: str
    files: list[str]
    paragraphs: list[Paragraph]

    def text(self) -> str:
        return "\n\n".join(
            f"{p.path}:{p.line}\n{'# ' if p.kind == 'heading' else ''}{p.text}"
            for p in self.paragraphs
        )


class Manuscript:
    """The files of a paper in the order a reader meets them."""

    def __init__(self, paper: Path) -> None:
        self.paper = paper.resolve()
        if not self.paper.exists():
            raise MissionError(f"{paper} does not exist")

    def _included(self, root: Path) -> list[Path]:
        """`root` and the files it includes after its preamble, figure code left out.

        They come depth first, each once.
        """
        found: dict[Path, None] = {}

        def visit(file: Path) -> None:
            if file in found or not file.is_file():
                return
            found[file] = None
            text = file.read_text(encoding="utf-8", errors="replace")
            for line in (text.partition(_BODY)[2] or text).splitlines():
                for name in _INCLUDE.findall(_COMMENT.sub("", line)):
                    target = (file.parent / name).with_suffix(".tex")
                    if not _NOT_PROSE & set(target.relative_to(root.parent, walk_up=True).parts):
                        visit(target)

        visit(root)
        return list(found)

    def files(self) -> list[Path]:
        """The document's root and the files it includes.

        Without a root, the folder's Markdown and LaTeX files in name order.
        """
        if self.paper.is_file():
            return self._included(self.paper) if self.paper.suffix == ".tex" else [self.paper]
        roots = [
            file
            for file in sorted(self.paper.rglob("*.tex"))
            if _BODY in file.read_text(errors="replace")
        ]
        if roots:
            return self._included(next((root for root in roots if root.name in _ROOTS), roots[0]))
        return sorted(file for file in self.paper.rglob("*") if file.suffix in {".md", ".tex"})


def prose(
    paper: Path,
    *,
    as_json: Annotated[bool, Parameter(name="--json")] = False,
) -> None:
    """The narrative paragraphs of a paper with file and line, math, floats and commands removed.

    paper: a paper's directory, its main `.tex` file, or a Markdown file.
    as_json: print JSON for a program.
    """
    files = Manuscript(paper).files()
    db = duckdb.connect()
    db.execute("CREATE TABLE manuscript (position INT, path VARCHAR, content VARCHAR)")
    db.executemany(
        "INSERT INTO manuscript VALUES (?, ?, ?)",
        [(n, _relative(file), file.read_text(errors="replace")) for n, file in enumerate(files)],
    )
    report = Prose(
        paper=_relative(paper.resolve()),
        files=[_relative(file) for file in files],
        paragraphs=[
            Paragraph(path=path, line=line, kind=kind, text=text)
            for _, path, line, kind, text in db.execute(_PARAGRAPHS).fetchall()
        ],
    )
    logger.result(report.render(as_json=as_json))


def _relative(path: Path) -> str:
    """`path` relative to the working directory when it lies below it."""
    here = Path.cwd()
    return path.relative_to(here).as_posix() if path.is_relative_to(here) else str(path)
