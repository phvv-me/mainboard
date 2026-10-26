# The project's proof: its own command, run over a copy of the project with some kept changes
# laid in, which a fix set passes only when the command does.

import os
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from ....proc import Processes
from ..mirror import Mirror

# How long one run of a project's proof may take before it counts as failed, and how many of
# its last output lines a failure is told with.
_PROOF_SECONDS = 3600
_SAID = 5


class Proof:
    """The project's proof command and what its last run said.

    root: the project as it stands, which every run starts from and the proof is told of.
    command: the proof command (`ProjectConfig.proof`), `{root}` standing for the copy; empty
        for a project without one.
    """

    def __init__(self, root: Path, command: Sequence[str]) -> None:
        self.root, self.command, self.said = root.resolve(), tuple(command), ""

    def culprit(self, components: Sequence[set[str]], *, proven: Path) -> set[str]:
        """The first of `components` whose changes, with those before it, fail the proof.

        The proof passes with none of them and fails with all of them, so a binary search over
        the prefixes finds it in a logarithm of their count.
        """
        passing, failing = 0, len(components)
        while failing - passing > 1:
            middle = (passing + failing) // 2
            if self.passes(components[:middle], proven=proven):
                passing = middle
            else:
                failing = middle
        return components[failing - 1]

    def passes(self, components: Sequence[set[str]], *, proven: Path) -> bool:
        """Whether the proof passes on the project with the changes of `components` alone."""
        with closing(Mirror(self.root)) as state:
            state.adopt(proven, set().union(*components))
            command = [part.replace("{root}", str(state.path)) for part in self.command]
            log = state.top / "proof.log"
            with log.open("wb") as output:
                failed = Processes().timeout(
                    _PROOF_SECONDS,
                    command,
                    cwd=state.path,
                    env={
                        **os.environ,
                        "MB_FIX_MIRROR": str(state.path),
                        "MB_FIX_ROOT": str(self.root),
                    },
                    output=output,
                )
            self.said = " | ".join(log.read_text(errors="replace").strip().splitlines()[-_SAID:])
            return not failed
