"""`impact`: which sealed results a source change invalidates, and what to rerun.

Two tiers answer it. What *depends* on a change is every file that imports the changed code,
directly or through others (a plain recursive reach over the import edges), the trial tests among
them, the experiment folders those sit in, and the paper sections citing those folders. What the
*seal* marks is wider, because mainboard seals a trial's whole source closure (`jobs/closure.py`):
a trial's node folder, the conftests above it and the workspace manifest, and the whole top-level
package of every installed distribution it imports from. A seal-only trial goes stale though its
numbers cannot change.
"""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from patos import FrozenModel

from ....core.errors import MissionError
from ....git.process import Git
from ....log import logger
from ..report import Report
from ..workspace import Workspace
from .experiments import node_status, read_papers

_MANIFESTS = frozenset({"mainboard.toml", "pytest.ini"})
_SHOWN = 12

_CHANGES = """
    SELECT c.path, f.module, e.name AS experiment
    FROM (SELECT unnest($changed) AS path) c
    LEFT JOIN module_files() f USING (path)
    LEFT JOIN experiment_nodes() e ON starts_with(c.path, e.dir || '/')
    ORDER BY c.path
"""

_REACHED = """
    SELECT r.path, r.hops, f.module, e.name AS experiment, t.path IS NOT NULL AS trial,
           r.path IN (SELECT path FROM design_reach($seeds)) AS design
    FROM reverse_reach($seeds) r
    JOIN module_files() f USING (path)
    LEFT JOIN experiment_nodes() e ON starts_with(r.path, e.dir || '/')
    LEFT JOIN trial_files() t ON t.path = r.path
    ORDER BY r.hops, r.path
"""

_SEALED = """
    WITH changed AS (SELECT unnest($changed) AS path),
    shipped AS (
      SELECT list(m.path) AS paths FROM module_files() m
      WHERE m.packaged AND m.home NOT IN (SELECT home FROM trial_files())
        AND EXISTS (SELECT 1 FROM changed c WHERE starts_with(c.path,
              CASE WHEN m.home = '' THEN '' ELSE m.home || '/' END
                || split_part(m.module, '.', 1) || '/'))
    ), marked AS (
      SELECT path FROM reverse_reach((SELECT paths FROM shipped))
      UNION
      SELECT t.path FROM trial_files() t
      WHERE EXISTS (SELECT 1 FROM changed c WHERE starts_with(c.path, t.experiment_dir || '/'))
      UNION
      SELECT t.path FROM trial_files() t, changed c
      WHERE regexp_extract(c.path, '[^/]*$') = 'conftest.py'
        AND (regexp_extract(c.path, '^(.*)/[^/]*$', 1) = ''
             OR starts_with(t.path, regexp_extract(c.path, '^(.*)/[^/]*$', 1) || '/'))
    )
    SELECT t.path, t.experiment FROM trial_files() t WHERE t.path IN (SELECT path FROM marked)
"""

_TRIALS = """
    SELECT experiment, count(*) AS trials FROM trial_files()
    WHERE experiment IS NOT NULL GROUP BY experiment
"""

_CITED = """
    SELECT path, line, experiment, section FROM paper_mentions()
    WHERE experiment IN (SELECT unnest($names)) ORDER BY path, line
"""


class Change(FrozenModel):
    """One changed file: the Python module it is, or the experiment folder it sits in."""

    path: str
    module: str | None = None
    experiment: str | None = None
    deleted: bool = False


class Reached(FrozenModel):
    """A file that imports changed code, `hops` imports away from it.

    design: it also reaches the code through design imports alone, none under `if TYPE_CHECKING:`
        and none through a package initializer.
    """

    path: str
    module: str
    hops: int
    experiment: str | None
    trial: bool
    design: bool


class Citation(FrozenModel):
    """A line of a paper citing an experiment, under the heading `section`."""

    path: str
    line: int
    section: str | None


class ExperimentImpact(FrozenModel):
    """One experiment folder a change reaches.

    trials: the trial tests that import changed code.
    trial_total: all the folder's trial tests.
    touched: changed files inside the folder.
    status: the folder's `node.md` status.
    """

    name: str
    status: str
    settled: bool
    trials: list[str]
    trial_total: int
    files: int
    touched: list[str]
    cited: list[Citation]


class PaperSection(FrozenModel):
    """A paper heading whose text cites reached experiments."""

    path: str
    line: int
    section: str | None
    experiments: list[str]


class ImpactReport(Report):
    """What a set of changed files reaches."""

    root: str
    changed: list[Change]
    reached: list[Reached]
    experiments: list[ExperimentImpact]
    sealed_trials: int
    sealed_experiments: list[str]
    papers: list[PaperSection]

    @staticmethod
    def _change(change: Change) -> str:
        kind = (
            f"module {change.module}"
            if change.module
            else f"experiment {change.experiment}"
            if change.experiment
            else "deleted"
            if change.deleted
            else "not imported"
        )
        return f"  {change.path}  ({kind})"

    @staticmethod
    def _experiment(experiment: ExperimentImpact) -> str:
        cites = ", ".join(sorted({Path(c.path).name + f":{c.line}" for c in experiment.cited}))
        return (
            f"  {experiment.name}  {experiment.status or '-'}"
            f"{'' if experiment.settled else ' unsettled'}  "
            f"trials {len(experiment.trials)}/{experiment.trial_total}"
            f"{f'  files {experiment.files}' if experiment.files else ''}"
            f"{f'  -> {cites}' if cites else ''}"
        )

    @staticmethod
    def _head(lines: Sequence[str], shown: int = _SHOWN) -> list[str]:
        """The first `shown` lines, then a count of the rest."""
        more = len(lines) - shown
        return [*lines[:shown], *([f"  +{more} more (--json lists all)"] if more > 0 else [])]

    def text(self) -> str:
        if not self.changed:
            return "no changed files"
        importers = [reached for reached in self.reached if reached.hops]
        direct = [reached for reached in importers if reached.hops == 1]
        deepest = max((reached.hops for reached in importers), default=0)
        lines = [f"{len(self.changed)} changed in {self.root}"]
        lines += self._head([self._change(change) for change in self.changed])
        lines.append(
            f"imported by {len(importers)} files over {deepest} hops, {len(direct)} directly, "
            f"{sum(reached.design for reached in importers)} through design imports alone"
        )
        lines += self._head([f"  {reached.path}" for reached in direct])
        if self.experiments:
            trials = sum(len(experiment.trials) for experiment in self.experiments)
            lines.append(f"{len(self.experiments)} experiments reached, {trials} trials to rerun:")
            lines += self._head([self._experiment(e) for e in self.experiments], shown=30)
        if self.sealed_experiments:
            lines.append(
                f"the seal also marks {self.sealed_trials} trials in "
                f"{len(self.sealed_experiments)} experiments that do not import the change"
            )
        if self.papers:
            lines.append(f"{len(self.papers)} paper sections cite the experiments reached:")
            lines += self._head(
                [
                    f"  {paper.path}:{paper.line}  {paper.section or '-'}  <- "
                    f"{', '.join(paper.experiments)}"
                    for paper in self.papers
                ],
                shown=20,
            )
        return "\n".join(lines)


class ImpactAnalysis:
    """The reach of a set of changed files in one project."""

    @staticmethod
    def _papers(experiments: Sequence[ExperimentImpact]) -> list[PaperSection]:
        """The cited headings, one per paper section, with the experiments citing them."""
        sections: dict[tuple[str, str | None], PaperSection] = {}
        for experiment in experiments:
            for citation in experiment.cited:
                key = (citation.path, citation.section)
                known = sections.get(key)
                sections[key] = PaperSection(
                    path=citation.path,
                    line=min(citation.line, known.line) if known else citation.line,
                    section=citation.section,
                    experiments=sorted({*(known.experiments if known else []), experiment.name}),
                )
        return sorted(sections.values(), key=lambda paper: (paper.path, paper.line))

    def __init__(self, workspace: Workspace, changed: list[str]) -> None:
        self.workspace = workspace
        self.changed = changed

    def _citations(self, names: list[str]) -> dict[str, list[Citation]]:
        """The paper lines citing each experiment of `names`."""
        if not names:
            return {}
        read_papers(self.workspace)
        found: dict[str, list[Citation]] = defaultdict(list)
        for row in self.workspace.rows(_CITED, {"names": names}):
            found[row["experiment"]].append(
                Citation(path=row["path"], line=row["line"], section=row["section"])
            )
        return found

    def _experiments(
        self, changes: Sequence[Change], reached: Sequence[Reached]
    ) -> list[ExperimentImpact]:
        """The experiment folders holding reached or changed files, with their trial counts."""
        totals = {row["experiment"]: row["trials"] for row in self.workspace.rows(_TRIALS)}
        files: dict[str, list[Reached]] = defaultdict(list)
        for entry in reached:
            if entry.experiment:
                files[entry.experiment].append(entry)
        touched: dict[str, list[str]] = defaultdict(list)
        for change in changes:
            if change.experiment:
                touched[change.experiment].append(change.path)
        names = sorted({*files, *touched})
        status = {
            row["name"]: row
            for row in self.workspace.rows(
                "SELECT name, dir, settled FROM experiment_nodes() "
                "WHERE name IN (SELECT unnest($names))",
                {"names": names},
            )
        }
        cited = self._citations(names)
        return [
            ExperimentImpact(
                name=name,
                status=node_status(self.workspace.root / status[name]["dir"] / "node.md"),
                settled=status[name]["settled"],
                trials=[Path(e.path).name for e in files[name] if e.trial],
                trial_total=totals.get(name, 0),
                files=len([e for e in files[name] if not e.trial]),
                touched=touched[name],
                cited=cited.get(name, []),
            )
            for name in names
        ]

    def _sealed(self, parameters: dict) -> list[tuple[str, str | None]]:
        """The trials the seal marks as stale, (path, experiment)."""
        if _MANIFESTS & set(self.changed):
            rows = self.workspace.rows("SELECT path, experiment FROM trial_files()")
        else:
            rows = self.workspace.rows(_SEALED, parameters)
        return [(row["path"], row["experiment"]) for row in rows]

    def report(self) -> ImpactReport:
        """The changes and everything they reach, up to the paper sections.

        That is what imports them, the experiments and trials among that, the trials the seal marks
        besides, and the paper sections citing the experiments.
        """
        parameters = {"changed": self.changed}
        changes = [
            Change(
                path=row["path"],
                module=row["module"],
                experiment=row["experiment"],
                deleted=not (self.workspace.root / row["path"]).exists(),
            )
            for row in self.workspace.rows(_CHANGES, parameters)
        ]
        seeds = [change.path for change in changes if change.module]
        reached = [Reached(**row) for row in self.workspace.rows(_REACHED, {"seeds": seeds})]
        trials = {entry.path for entry in reached if entry.trial}
        sealed = [(path, name) for path, name in self._sealed(parameters) if path not in trials]
        experiments = self._experiments(changes, reached)
        return ImpactReport(
            root=str(self.workspace.root),
            changed=changes,
            reached=reached,
            experiments=experiments,
            sealed_trials=len(sealed),
            sealed_experiments=sorted({name for _, name in sealed if name}),
            papers=self._papers(experiments),
        )


def _inside(root: Path, *, file: Path) -> str:
    """`file` relative to `root`, read from the working directory first and from `root` second."""
    for candidate in (file.resolve(), root / file):
        if candidate.is_relative_to(root):
            return candidate.relative_to(root).as_posix()
    raise MissionError(f"{file} lies outside {root}")


def _changed_files(root: Path, since: str | None) -> list[str]:
    """The files of `root` changed since `since` (HEAD without it), relative to `root`.

    They are what differs in the working tree and what git does not yet track.
    """
    git = Git(root)
    named = [
        *git.out("diff", "--name-only", "--relative", "-z", since or "HEAD").split("\0"),
        *git.out("ls-files", "--others", "--exclude-standard", "-z").split("\0"),
    ]
    return sorted({name for name in named if name})


def impact(
    *files: Path,
    since: str | None = None,
    root: Path = Path(),
    as_json: Annotated[bool, Parameter(name="--json")] = False,
    memory: str = "2GB",
) -> None:
    """The files, experiments, trials and paper sections a source change reaches.

    files: the changed files; without them, git's changes under `root`.
    since: a git ref to compare the working tree with instead of HEAD.
    root: the project to analyze.
    as_json: print JSON for a program.
    memory: a DuckDB memory limit such as `4GB`.
    """
    root = root.resolve()
    changed = (
        sorted(_inside(root, file=file) for file in files)
        if files
        else _changed_files(root, since)
    )
    report = ImpactAnalysis(Workspace(root, memory=memory), changed).report()
    logger.result(report.render(as_json=as_json))
