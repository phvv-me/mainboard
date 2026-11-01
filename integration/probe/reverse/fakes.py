from collections.abc import Sequence
from pathlib import Path

from mainboard.core import Runner, ToolOutput
from mainboard.probe.reverse.constants import STANDARD_FLAGS

from ...constants import ELF, NVDISASM, RESOURCE, SASS

_NO_PTX = "cuobjdump info    : No PTX file found to extract from 'lib.so'\n"

# What each fixture command prints, by its first argument.
_ANSWERS = {
    "--version": "Cuda compilation tools, release 13.4, V13.4.48\n",
    "--list-elf": ELF,
    "--list-ptx": _NO_PTX,
    "--dump-resource-usage": RESOURCE,
    "--dump-sass": SASS,
    "--extract-elf": "",
    "-c": NVDISASM,
}


class FixtureRunner(Runner):
    def __init__(self, outputs: dict[str, ToolOutput] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[tuple[str, tuple[str, ...], Path | None]] = []

    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        self.calls.append((tool, tuple(args), cwd))
        if args[0] in self.outputs:
            return self.outputs[args[0]]
        if args[0] == "--help":
            return ToolOutput(stdout="\n".join(STANDARD_FLAGS[tool]))
        if args[0] not in _ANSWERS:
            raise AssertionError(f"unexpected fixture command: {tool} {args}")
        return ToolOutput(stdout=_ANSWERS[args[0]])
