"""`evidence`: what each paper rests on, and where that evidence is weak.

A paper cites an experiment by naming its folder (`paper_mentions`). Per paper this lists the
sections with the experiments they cite and each one's status; then three audits a reviewer would
otherwise do by hand: citations of an experiment with no `result.md` yet, cited names that are no
experiment folder (a renamed or removed experiment), and settled experiments no paper cites.
"""

from collections import defaultdict
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from patos import FrozenModel

from ..lint import app
from .experiments import node_status, read_papers
from .report import Report
from .workspace import Workspace

_SHOWN = 12

_CITATIONS = """
    SELECT m.path, m.line, m.section, m.experiment, m.known, coalesce(e.settled, false) AS settled,
           e.dir
    FROM paper_mentions() m LEFT JOIN experiment_nodes() e ON e.name = m.experiment
    ORDER BY m.path, m.line, m.experiment
"""

_UNCITED = """
    SELECT name, dir FROM experiment_nodes()
    WHERE settled AND name NOT IN (SELECT experiment FROM paper_mentions() WHERE known)
    ORDER BY name
"""


def _paper(path: str) -> str:
    """The paper a file belongs to: its folder below `papers/`, else its own folder."""
    parts = Path(path).parts
    return "/".join(
        parts[: min(parts.index("papers") + 2, len(parts) - 1)]
        if "papers" in parts
        else parts[:-1]
    )


class Citation(FrozenModel):
    """A line of a paper citing an experiment.

    status: the experiment's `node.md` status, empty for a name that is no experiment.
    settled: the experiment folder holds a `result.md`.
    """

    path: str
    line: int
    section: str | None
    experiment: str
    status: str
    settled: bool
    known: bool


class Uncited(FrozenModel):
    """A settled experiment no paper cites."""

    name: str
    status: str


class EvidenceReport(Report):
    """The citations of every paper of a project and the weak spots among them."""

    root: str
    citations: list[Citation]
    uncited: list[Uncited]

    def text(self) -> str:
        cited = {c.experiment for c in self.citations if c.known}
        lines = [f"{len(cited)} experiments cited"]
        by_paper: dict[str, list[Citation]] = defaultdict(list)
        for citation in self.citations:
            by_paper[_paper(citation.path)].append(citation)
        for paper, citations in by_paper.items():
            known = {c.experiment for c in citations if c.known}
            lines.append(f"{paper}: {len(known)} experiments in {len(citations)} lines")
        lines += self._audit(
            "cite an experiment with no result.md",
            [c for c in self.citations if c.known and not c.settled],
        )
        lines += self._audit(
            "cite a name that is no experiment folder", [c for c in self.citations if not c.known]
        )
        lines.append(f"{len(self.uncited)} settled experiments no paper cites")
        lines += self._head([f"  {entry.name}  {entry.status or '-'}" for entry in self.uncited])
        return "\n".join(lines)

    def _audit(self, title: str, found: list[Citation]) -> list[str]:
        """A heading with the count and the first citations, or nothing when `found` is empty."""
        where = [
            f"  {c.path}:{c.line}  {c.section or '-'}  <- {c.experiment}"
            f"{f'  ({c.status})' if c.status else ''}"
            for c in found
        ]
        return [f"{len(found)} lines {title}:", *self._head(where)] if found else []

    @staticmethod
    def _head(lines: list[str]) -> list[str]:
        """The first lines, then a count of the rest."""
        more = len(lines) - _SHOWN
        return [*lines[:_SHOWN], *([f"  +{more} more (--json lists all)"] if more > 0 else [])]


class Audit:
    """The paper citations of one project."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def report(self) -> EvidenceReport:
        """Every citation with its experiment's status, and the settled experiments never cited."""
        read_papers(self.workspace)
        return EvidenceReport(
            root=str(self.workspace.root),
            citations=[
                Citation(
                    path=row["path"],
                    line=row["line"],
                    section=row["section"],
                    experiment=row["experiment"],
                    status=self._status(row["dir"]),
                    settled=row["settled"],
                    known=row["known"],
                )
                for row in self.workspace.rows(_CITATIONS)
            ],
            uncited=[
                Uncited(name=row["name"], status=self._status(row["dir"]))
                for row in self.workspace.rows(_UNCITED)
            ],
        )

    def _status(self, folder: str | None) -> str:
        return node_status(self.workspace.root / folder / "node.md") if folder else ""


@app.command
def evidence(
    *,
    root: Path = Path(),
    as_json: Annotated[bool, Parameter(name="--json")] = False,
    memory: str = "2GB",
) -> None:
    """What each paper cites and where the evidence is weak: unsettled, missing, uncited.

    root: the project to analyze.
    as_json: print JSON for a program.
    memory: a DuckDB memory limit such as `4GB`.
    """
    report = Audit(Workspace(root.resolve(), memory=memory)).report()
    print(report.render(as_json=as_json))
