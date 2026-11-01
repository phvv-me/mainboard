from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict

from ..batch.spec import BatchSpec, Selection
from ..core.errors import MissionError
from ..core.section import Section, failed
from ..dispatch.jobs.vocabulary import Request
from ..fleet.board import Board
from ..log import logger
from ..render import Output, Terminal, rows

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..batch.runner import Batch
    from ..manifest.render.interpolate import Json


@Parameter(name="*")
class Asked(FrozenModel):
    """What a submitted job asks of its host and brings home; an unset field is the profile's."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    queue: str = ""
    """a queued host's queue, the profile's default when empty."""
    walltime: str = ""
    """the wall-clock limit, `HH:MM:SS`, the profile's default when empty."""
    mem_gb: int = 0
    """system memory to ask the scheduler for, the profile's default when 0."""
    gpus: int = 0
    """cards per job, the profile's default when 0."""
    gpu_name: str = ""
    """the GPU type to rent, for a metered provider host."""
    arch: str = ""
    """rent the cheapest card of these compute capabilities (`sm_120`, `sm_90+`, `hopper`)
    instead of a named card, for a metered provider host."""
    max_usd: float = 0.0
    """the spend cap a provider host refuses to submit without."""
    spot: bool = False
    """rent interruptible capacity; a job taken back reads `vanished`, and `--resume <name>
    --spot` continues it from its `MB_CHECKPOINT`."""
    datacenter: bool = False
    """rent only from data-center hosts, for a provider that also lists individually run
    machines (vast), whose marketplace hosts drop jobs and download slowly."""
    fetch: str = ""
    """a results path recorded for later collection, the node's evidence directory when unset
    and `--node` names one."""
    node: str = ""
    """the ledger slug this run serves, carried into its record and receipts."""
    needs: tuple[str, ...] = ()
    """a workspace-relative data path the job reads on the host, repeatable."""
    env: str = ""
    """the environment the job enters, the host profile's own when empty."""
    container: str = ""
    """a container override, `none` forcing bare."""

    def given(self) -> dict[str, Json]:
        """The asks set, which a `--job` batch takes as every job's defaults."""
        return self.model_dump(exclude_defaults=True, exclude={"needs"})

    def request(self, command: str, *, target: str, name: str = "", attempt: int = 1) -> Request:
        """The dispatch these asks make of `target` for `command`."""
        return Request(
            target=target,
            command=command,
            name=name,
            attempt=attempt,
            fetch=self.fetch or None,
            **self.model_dump(exclude={"fetch"}),
        )


@Parameter(name="*")
class Declared(FrozenModel):
    """A batch's declaration: its spec file or inline jobs, a selection, `[vars]` values."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    batch: str = ""
    """a batch spec file, relative to the workspace root."""
    job: tuple[str, ...] = ()
    """a `target:command` job, repeatable, for a batch declared without a file."""
    only: str = ""
    """the plan's jobs to act on, names or `kind-*` globs, comma-separated; all when unset."""
    set_: Annotated[tuple[str, ...], Parameter(name="--set", negative="")] = ()
    """a `name=value` filling one of the spec file's `[vars]`, repeatable."""

    def planned(self, board: Board, *, name: str = "", resources: Mapping[str, Json]) -> Batch:
        """The declared batch over `board`'s workspace.

        resources: the asks of the verb, which a `--job` batch takes as every job's defaults and
            a spec file refuses (it declares them itself, per job).
        """
        return board.batch(
            self.spec(board.root, name=name, resources=resources),
            selection=Selection.of(self.only),
        )

    def spec(self, root: Path, *, name: str = "", resources: Mapping[str, Json]) -> BatchSpec:
        """The batch spec declared: the spec file under `root`, else the inline jobs."""
        if self.batch:
            if resources:
                flags = ", ".join(f"--{field.replace('_', '-')}" for field in resources)
                raise MissionError(
                    f"{flags} would be ignored by {self.batch}, which sets them per job: "
                    "declare them in its [defaults] or its [[jobs]]"
                )
            return BatchSpec.load(root / self.batch, answers(self.set_))
        if self.set_:
            raise MissionError("--set fills a spec file's [vars]; a --job batch declares none")
        if not self.job:
            raise MissionError("declare a batch: --batch <spec file>, or --job target:command")
        return BatchSpec.inline(name or "batch", self.job, resources)


@Parameter(name="*")
class Dispatching(FrozenModel):
    """Where a submission goes and how: its host, its name, a lane's split, and whether it asks
    first, stops at the price or waits for the verdict."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    on: str = ""
    """the host alias; comma-separated for a lane."""
    resume: str = ""
    """a failed or cancelled run to continue, by handle or name, as its next attempt."""
    name: str = ""
    """the run's label, or the batch's name for a `--job` batch."""
    attempt: int = 1
    """the 1-based try number feeding expression defaults."""
    split: str = ""
    """a lane's parametrize name; each of its values becomes one job."""
    per_job: int = 0
    """a lane's cells per job when no name groups them, 0 for all in one."""
    cell_timeout: float = 900.0
    """seconds one lane cell may take before its process is killed."""
    estimate: bool = False
    """price the plan and print it; dispatch nothing."""
    wait: bool = False
    """block until every submitted job settles; exit 0 clean, 1 failed, 2 timeout."""
    yes: bool = False
    """dispatch without asking, what a script passes."""

    @property
    def hosts(self) -> list[str]:
        """The hosts `--on` names, a lane's several or a job's one."""
        return [alias.strip() for alias in self.on.split(",") if alias.strip()]

    @property
    def is_lane(self) -> bool:
        """Whether this submission runs a test lane's cells rather than one job."""
        return bool(self.split or self.per_job or "," in self.on)

    def proceeds(self) -> bool:
        """Whether to dispatch: a person at a terminal is asked once unless `--yes`, an agent not.

        The question shares stderr with the expectation line it follows, since stdout belongs to
        the handle or the document the verb prints once the dispatch has happened. An input that
        ends unanswered is a no.
        """
        if self.yes or not Terminal.current().interactive:
            return True
        logger.prompt("dispatch? [y/N] ")
        try:
            answer = input()
        except EOFError:
            answer = None
        if answer is None:
            logger.info("no answer; pass --yes to dispatch without asking")
            return False
        return answer.strip().lower() in {"y", "yes"}


# The columns each table always carries, so an empty one still renders its heading; the batch
# tables' totals row is also summed over this shape, and the job listing lines a settled row's
# empty live columns up under the live rows' own.
_SECTION_ROW = ("section", "verdict", "detail", "fix")


def query_source(sql: str | None, file: Path | None) -> str | Path | None:
    """Select explicit SQL text or a file without interpreting strings as paths."""
    if sql is not None and file is not None:
        raise MissionError("SQL statement and --file are mutually exclusive")
    return file if file is not None else sql


def said(line: str) -> None:
    """One line a wait says while it blocks, on stderr and at once."""
    logger.info(line)


def sectioned(
    sections: list[Section], output: Output, *, title: str, heading: bool = False
) -> int:
    """Print a report's rows and answer its exit status: 1 when any row failed."""
    if heading:
        titled(output, title)
    output.print_rows(
        [section.model_dump() for section in sections], title=title, columns=_SECTION_ROW
    )
    return 1 if failed(sections) else 0


def titled(output: Output, title: str) -> None:
    """Name the next compact table when a verb prints several (rich tables carry their title)."""
    if output.mode is None:
        logger.result(f"\n# {title}")


def answers(given: Sequence[str]) -> dict[str, str]:
    """The `question=value` pairs a caller passed, refusing one written without its value."""
    split = [pair.partition("=") for pair in given]
    if bare := [pair for pair, separator, _ in split if not separator]:
        raise MissionError(f"answers are written question=value, not {bare[0]!r}")
    return {question: answer for question, _, answer in split}


def judged(sections: Iterable[Section], *, mode: str | None, title: str) -> None:
    """Print a machine's findings as a table of their own, under the record they judge."""
    if mode is None:
        logger.result(f"\n# {title}")
    rows(
        [section.model_dump() for section in sections], mode=mode, fields=_SECTION_ROW, title=title
    )
