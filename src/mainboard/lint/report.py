from patos import FrozenModel

from .process import Outcome


class Report(FrozenModel):
    """What one lint pass did and what it left for a person.

    files: how many files the pass read.
    rewritten: the workspace-relative files a repair or formatter changed.
    failures: the steps that exited nonzero, each with everything it printed.
    """

    files: int
    rewritten: tuple[str, ...] = ()
    failures: tuple[Outcome, ...] = ()

    @property
    def clean(self) -> bool:
        """Whether the files were already right: nothing rewritten and nothing failed."""
        return not self.rewritten and not self.failures

    def findings(self) -> str:
        """Every failing step's own words under a heading naming the step and its owner."""
        return "\n\n".join(
            f"{failure.step} [{failure.owner}] exited {failure.code} after "
            f"{failure.seconds:.1f}s\n{failure.output.rstrip()}"
            for failure in self.failures
        )

    def summary(self) -> str:
        """One line saying how many files were read, which were rewritten, and what failed."""
        rewritten = f"rewrote {len(self.rewritten)}" + (
            f" ({', '.join(self.rewritten)})" if self.rewritten else ""
        )
        failed = ", ".join(sorted({failure.step for failure in self.failures})) or "none"
        return f"lint: {self.files} files, {rewritten}, failed: {failed}"
