from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict

from ...core.errors import MissionError
from ...log import logger
from ...probe.reverse.service import DeviceProbe
from ...probe.reverse.service import compact as device_rows
from ...render import progress, record, rows
from ...state import Lake
from ..base import Verbs
from ..options import Output
from ..work.jobs import JobVerbs

if TYPE_CHECKING:
    from ...manuscript import Report as PaperReport
    from ...render.values import Node

# The columns a manuscript check prints, so a clean build still names its problem table.
_SECTION_COLUMNS = ("number", "title", "page", "within")
_PROBLEM_COLUMNS = ("kind", "where", "detail")


@Parameter(name="*")
class Inspected(FrozenModel):
    """Which part of a device artifact `probe device` reads."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    arch: str = ""
    """one exact CUDA target, such as sm_90a."""
    function: str = ""
    """one mangled kernel name."""
    against: str = ""
    """the newer library for compare, with library as the baseline."""


# The verbs' defaults: compact output, every part of a device artifact.
_COMPACT = Output()
_INSPECTED = Inspected()


class PaperVerbs(Verbs):
    """Manuscripts built and checked, and the device code their kernels compiled to."""

    jobs: JobVerbs

    @staticmethod
    def _report(report: PaperReport, *, mode: str | None) -> None:
        """Print one manuscript check: the summary, where each section starts, and every problem.

        The JSON mode prints the report whole, one document a script can read; the other modes
        print three tables, the problem table naming its columns even when it is empty so a
        clean build still says so.
        """
        if mode == "json":
            record(report.model_dump(mode="json"), mode=mode, fields=(), title="paper")
            return
        summary: dict[str, Node] = {
            "pdf": report.pdf,
            "pages": report.pages,
            "limit": report.limit,
            "ends": report.ends,
            "ends_on": report.ends_on,
            "problems": len(report.problems),
        }
        record(summary, mode=mode, fields=(), title=f"paper: {report.paper}")
        last = report.limit or report.pages
        sections = [
            {**section.model_dump(), "within": section.page <= last} for section in report.sections
        ]
        rows(sections, mode=mode, fields=_SECTION_COLUMNS, title="sections")
        problems = [problem.model_dump(mode="json") for problem in report.problems]
        rows(problems, mode=mode, fields=_PROBLEM_COLUMNS, title="problems")

    def device(
        self,
        library: Path,
        operation: Literal["inventory", "sass", "arch", "resource", "compare"],
        *,
        inspected: Inspected = _INSPECTED,
        on: str = "local",
        json: bool = False,
    ) -> None:
        """Inspect CUDA device code and append its evidence to the workspace lake.

        Device analysis runs on this machine; reading a remote host's libraries needs job
        dispatch with artifact staging and evidence collection, which it does not have yet.

        Args:
            library: a native library, fatbin, cubin or PTX artifact.
            operation: the finding to collect; compare reports inventory metadata changes.
            on: local only until device analysis is wired into job dispatch.
            json: print the normalized evidence as JSON instead of compact TSV rows.
        """
        if on != "local":
            raise MissionError("probe device runs locally; dispatching it as a job is not built")
        evidence = DeviceProbe(Lake.at(self.session.workspace_root())).run(
            library, operation, **inspected.model_dump()
        )
        if json:
            record(evidence.model_dump(mode="json"), mode="json", fields=(), title="device")
            return
        payloads = device_rows(evidence)
        columns = tuple(dict.fromkeys(key for row in payloads for key in row))
        _COMPACT.print_rows(payloads, title="device", columns=columns)

    def paper_build(
        self,
        name: str,
        *,
        show: Iterable[str] = (),
        dpi: int = 110,
        output: Output = _COMPACT,
    ) -> int:
        """Build a declared manuscript and report everything wrong with it, exiting 1 on any.

        Built with tectonic in the workspace environment, then read back: errors, undefined
        references and citations, multiply defined labels and overfull boxes with the file and
        line each comes from, the page count, the page every section starts on, and whether
        the section `[papers.<name>] ends` names ends by page `limit`.

        Args:
            name: the `[papers.<name>]` manuscript.
            show: a phrase from the manuscript, repeatable; the page it appears on is rendered to a
                PNG beside the build and its path printed.
            dpi: the resolution a shown page renders at.
        """
        manuscript = self.session.board("local").paper(name)
        with progress(f"building {name}"):
            report = manuscript.check()
        PaperVerbs._report(report, mode=output.mode)
        for phrase in show:
            logger.result(manuscript.show(phrase, dpi=dpi).as_posix())
        return 1 if report.problems else 0
