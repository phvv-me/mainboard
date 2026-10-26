import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ...ci import LocalLeg, Matrix, Package
from ...log import logger
from ...render import progress
from ...upkeep import staleness
from ..base import Verbs
from ..options import Output, said

if TYPE_CHECKING:
    from ...ci import Result as CiResult

_CI_COLUMNS = ("leg", "os", "step", "verdict", "seconds")

# The verbs' default output.
_COMPACT = Output()


class ToolVerbs(Verbs):
    """This tool itself: its CI gate, its update and its version."""

    @staticmethod
    def _local(found: Package) -> list[CiResult]:
        """The gate's steps run here, each step's transcript said as it settles."""
        leg = LocalLeg(found.root)
        results = []
        for result in leg.run(found.definition.on(leg.family)):
            ToolVerbs._told(result)
            results.append(result)
        return results

    @staticmethod
    def _told(result: CiResult) -> None:
        """One settled step's transcript, on stderr and at once, nothing for a step never run."""
        if transcript := result.transcript:
            said(transcript)

    def _matrix(self, found: Package) -> list[CiResult]:
        """The gate run here and on one declared host per other platform, failures said."""
        planned = Matrix.planned(found, self.session.board("local"))
        with progress(f"running the gate on {len(planned.legs)} legs"):
            results = planned.run()
        for result in results:
            if result.failed:
                ToolVerbs._told(result)
        for family in planned.uncovered:
            said(f"no leg ran on {family}; declare a host of it in [ci] hosts")
        return results

    def ci_run(
        self, package: Path | None = None, *, matrix: bool = False, output: Output = _COMPACT
    ) -> int:
        """Run the gate `[tool.mainboard.ci]` declares, exactly as the package's CI job runs it.

        The package is the nearest directory at or above PACKAGE whose pyproject.toml declares a
        gate, and needs no workspace. Its steps run in order from the package directory and stop
        at the first failure; each step's output goes to stderr as it settles, the table to
        stdout. With `--matrix` the gate also runs, at the same time, on every `[ci] hosts` entry
        of a supported platform this machine is not, the working tree shipped there as it
        stands, and only failing steps print their output. Exits 1 when any step failed.

        Args:
            package: a directory inside the package, the working directory when omitted.
            matrix: also run on one declared host per other platform, the check before a push.
        """
        found = Package.found((package or Path.cwd()).resolve())
        results = self._matrix(found) if matrix else ToolVerbs._local(found)
        output.print_rows([result.row() for result in results], title="ci", columns=_CI_COLUMNS)
        return 1 if any(result.failed for result in results) else 0

    def update(self) -> None:
        """Reinstall this tool from its source when the source moved, as `uv self update` does.

        Nothing checks or updates on its own: every other verb runs the code installed.
        """
        found = staleness.check()
        if found.source is None or not found.stale:
            logger.result(f"{self.session.project.name}: {found.detail}")
            return
        logger.result(f"{self.session.project.name}: {found.detail}; {staleness.update(found)}")

    def version(self, *, output: Output = _COMPACT) -> None:
        """Show this installation: its version, where it runs from, and the engines it pins.

        DuckDB is named because a served lake needs the same release at both ends.
        """
        output.print_record(
            {
                "version": metadata.version(self.session.project.package),
                "python": sys.executable,
                "source": str(staleness.check().source or ""),
                "duckdb": duckdb.__version__,
            },
            title="self version",
        )
