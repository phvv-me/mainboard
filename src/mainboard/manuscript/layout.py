# Where a built manuscript's sections landed on the page, read from the `.aux` (page count and
# each section's first page), the sources (which lines belong to a section) and SyncTeX (the page
# each line was typeset on).
#
# A section's first page is exact in the table of contents. Its last page, the one a venue's rule
# is about, is written nowhere, and the next section's start does not settle it since two
# sections usually share a page, so it is the last page SyncTeX places any line of the section on,
# floats included, exactly what a reviewer counting pages would count.

import gzip
import re
from pathlib import Path

from patos import FrozenModel

_PAGES = re.compile(r"\\gdef\s*\\@abspage@last\{(?P<pages>\d+)\}")
_CONTENTS = "\\@writefile{toc}{\\contentsline {section}"
_NUMBER = re.compile(r"\\numberline\s*\{(?P<number>[^}]*)\}")
_COMMAND = re.compile(r"\\[A-Za-z@]+\*?\s*")
_INPUT = re.compile(r"\\(?:input|include)\s*\{(?P<path>[^}]+)\}")
_SECTION = re.compile(r"\\section\*?\s*(?:\[[^\]]*\])?\s*\{(?P<title>.*)")
# Where a section's text stops: the next top-level heading, the appendix switch, the
# bibliography, or an unnumbered heading at any depth, which is how a venue's reproducibility,
# ethics and acknowledgment statements are set outside the page count.
_BOUNDARY = re.compile(
    r"\\(?:(?:sub)*section\*|paragraph\*|(?:section|appendix|bibliography|printbibliography"
    r"|part|chapter)\b)"
)
_COMMENT = re.compile(r"(?<!\\)%.*")
_RECORD = re.compile(r"^[\[(hvxkg$](?P<tag>\d+),(?P<line>\d+)")


class Section(FrozenModel):
    """One top-level section and the page its heading was typeset on.

    number: as typeset, `3` or `A`, empty for an unnumbered one.
    title: with its TeX markup stripped.
    """

    number: str = ""
    title: str
    page: int


class Aux:
    """The page count and the section starts a build wrote into its `.aux` file."""

    def __init__(self, text: str) -> None:
        self.text = text

    @property
    def pages(self) -> int:
        """The page count LaTeX recorded, 0 when it recorded none."""
        found = _PAGES.search(self.text)
        return int(found["pages"]) if found else 0

    def sections(self) -> list[Section]:
        """Every top-level section in reading order."""
        listed: list[Section] = []
        for line in self.text.splitlines():
            if not line.startswith(_CONTENTS):
                continue
            title, page = _groups(line[len(_CONTENTS) :], count=2)
            number = _NUMBER.search(title)
            listed.append(
                Section(
                    number=number["number"] if number else "",
                    title=plain(_NUMBER.sub("", title)),
                    page=int(page) if page.isdigit() else 0,
                )
            )
        return listed


class SyncTex:
    """Which source lines were typeset on which page, from a build's `.synctex.gz`.

    directory: the manuscript directory relative input paths resolve against.
    """

    def __init__(self, text: str, *, directory: Path) -> None:
        self.inputs: dict[str, Path] = {}
        self.pages: dict[int, set[tuple[Path, int]]] = {}
        page = 0
        for line in text.splitlines():
            if line.startswith("Input:"):
                tag, _, path = line.removeprefix("Input:").partition(":")
                self.inputs[tag] = (directory / path).resolve()
            elif line.startswith("{") and line[1:].isdigit():
                page = int(line[1:])
            elif record := _RECORD.match(line):
                source = self.inputs.get(record["tag"])
                if source is not None and page:
                    self.pages.setdefault(page, set()).add((source, int(record["line"])))

    @classmethod
    def read(cls, path: Path, *, directory: Path) -> SyncTex:
        """The gzipped SyncTeX file a build writes at `path`."""
        return cls(gzip.decompress(path.read_bytes()).decode("utf-8"), directory=directory)

    def last_page(self, lines: set[tuple[Path, int]]) -> int:
        """The last page carrying any of `lines`, 0 when none of them was typeset."""
        return max((page for page, typeset in self.pages.items() if typeset & lines), default=0)


class Source:
    """The source lines of the root `.tex` file `main` in reading order, `\\input` and
    `\\include` followed."""

    def __init__(self, main: Path) -> None:
        self.directory = main.resolve().parent
        self.lines = self._read(main.resolve())

    def span(self, title: str) -> set[tuple[Path, int]]:
        """Every source line of the section titled `title` up to the next `_BOUNDARY`, empty when
        no section has that title."""
        wanted = plain(title).casefold()
        start = next(
            (
                index
                for index, (_, _, text) in enumerate(self.lines)
                if (heading := _SECTION.search(text))
                and plain(_groups("{" + heading["title"], count=1)[0]).casefold() == wanted
            ),
            None,
        )
        if start is None:
            return set()
        end = next(
            (
                index
                for index in range(start + 1, len(self.lines))
                if _BOUNDARY.search(self.lines[index][2])
            ),
            len(self.lines),
        )
        return {(path, number) for path, number, _ in self.lines[start:end]}

    def _read(self, path: Path) -> list[tuple[Path, int, str]]:
        """`path`'s lines with every input inlined where it is read, comments dropped."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        lines: list[tuple[Path, int, str]] = []
        for number, raw in enumerate(text.splitlines(), start=1):
            line = _COMMENT.sub("", raw)
            lines.append((path, number, line))
            for included in _INPUT.finditer(line):
                named = self.directory / included["path"].strip()
                named = named if named.suffix else named.with_suffix(".tex")
                lines.extend(self._read(named.resolve()))
        return lines


def plain(markup: str) -> str:
    """`markup` with TeX commands and braces dropped and its spaces collapsed."""
    text = _COMMAND.sub(" ", markup.replace("~", " ")).replace("{", "").replace("}", "")
    return " ".join(text.split())


def _groups(text: str, *, count: int) -> list[str]:
    """The first `count` brace groups of `text`, their braces balanced and removed."""
    groups: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(text):
        if character == "{":
            depth += 1
            if depth == 1:
                start = index + 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                groups.append(text[start:index])
                if len(groups) == count:
                    break
    return groups + [""] * (count - len(groups))
