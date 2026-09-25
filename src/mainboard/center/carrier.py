# How the local tool runs one function of the destination agent on a machine that has no tool.
#
# The destination has uv, which `center migrate` puts there first, and uv has a Python. That
# Python is started with a one-line bootstrap in its argv, the only thing that rides the command
# line, and everything else comes down ssh's stdin: a loader, then one JSON header naming the
# modules to load and the function to call with its arguments, then whatever byte stream that
# function reads, a tar of files say. The argv holds no secret and no path, so it quotes the same
# under cmd.exe, PowerShell and a POSIX login shell, and the answer comes back as one marked line
# so a login banner above it is never mistaken for it.

import json
from collections.abc import Iterable, Mapping
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from ..core.errors import MissionError
from ..dispatch.transport import SshTransport
from ..probe import census
from . import remote

if TYPE_CHECKING:
    from ..probe.census import Json

# The Python every call runs under, the one this tool itself requires, which uv fetches once.
PYTHON = "3.14"

# The prefix the answer line carries, the one line of stdout the caller reads.
MARKER = "mainboard-answer "

# The bootstrap in the argv: read one line of stdin and run it, which is the loader below.
_BOOTSTRAP = "import sys;exec(sys.stdin.buffer.readline())"

# The loader, sent as that first line. It registers each shipped module under its package name,
# parents first, so the agent's own relative import of the census resolves, then calls the named
# function of the last module and prints its answer after the marker.
_LOADER = f"""
import json, sys, types
def load(name, source):
    parts = name.split(".")
    for depth in range(1, len(parts)):
        package = ".".join(parts[:depth])
        if package not in sys.modules:
            made = types.ModuleType(package)
            made.__path__ = []
            sys.modules[package] = made
    module = types.ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    exec(compile(source, name, "exec"), module.__dict__)
    return module
header = json.loads(sys.stdin.buffer.readline())
for name, source in header["modules"].items():
    agent = load(name, source)
answer = getattr(agent, header["call"])(**header["arguments"])
sys.stdout.write({MARKER!r} + json.dumps(answer) + "\\n")
"""


def modules() -> dict[str, str]:
    """The agent and the census it imports, by module name, in the order they must load."""
    return {
        module.__name__: Path(module.__file__ or "").read_text(encoding="utf-8")
        for module in (census, remote)
    }


class Carrier:
    """Calls the destination agent's functions on one machine over the bounded ssh transport.

    host: the ssh alias the machine answers to.
    uv: the uv executable on that machine, as its probe found it.
    transport: the ssh policy the calls ride.
    """

    def __init__(self, host: str, uv: str, transport: SshTransport | None = None) -> None:
        self.host = host
        self.uv = uv
        self.transport = transport or SshTransport()

    @property
    def argv(self) -> tuple[str, ...]:
        """The ssh command line, the one part of a call a process listing shows."""
        line = f'"{self.uv}" run --no-project --quiet --python {PYTHON} python -c "{_BOOTSTRAP}"'
        return ("ssh", *self.transport.options, self.transport.destination(self.host), line)

    def call[Answer](
        self,
        function: str,
        arguments: Mapping[str, Json],
        shape: type[Answer],
        stream: Iterable[bytes] = (),
    ) -> Answer:
        """Run `function(**arguments)` of the agent on the machine and answer what it returned.

        function: the agent function's name.
        arguments: its keyword arguments, JSON data, carried on stdin and never on the argv.
        shape: what the answer is read back as.
        stream: bytes the function reads from stdin after its header.
        """
        header = {"modules": modules(), "call": function, "arguments": arguments}
        preamble = f"exec({_LOADER!r})\n{json.dumps(header)}\n".encode()
        said = self.transport.feed(
            self.argv, self.host, operation=function, chunks=chain([preamble], stream)
        )
        answer = next(
            (line[len(MARKER) :] for line in said.splitlines() if line.startswith(MARKER)), None
        )
        if answer is None:
            raise MissionError(
                f"{self.host!r} answered {function} without a result: {said.strip()[-240:]}"
            )
        return TypeAdapter(shape).validate_json(answer)
