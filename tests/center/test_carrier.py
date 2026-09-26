import getpass
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from mainboard.center import carrier, remote
from mainboard.center.carrier import MARKER, PYTHON, Carrier
from mainboard.center.state import Destination, Parcel, packed
from mainboard.core.errors import MissionError
from mainboard.dispatch.transport import Endpoint, SshTransport
from mainboard.probe import census


class Local(SshTransport):
    """An ssh that runs the bootstrap its command line carries here, through the real feed."""

    def feed(
        self, command: tuple[str, ...], host: str, *, operation: str, chunks: Iterable[bytes]
    ) -> str:
        """Run the argv's own bootstrap under this Python, fed exactly what ssh would carry."""
        bootstrap = command[-1].rpartition(" -c ")[2].strip('"')
        local = (sys.executable, "-c", bootstrap)
        return super().feed(local, host, operation=operation, chunks=chunks)


class Recorder(SshTransport):
    """An ssh that keeps the argv and stdin it was handed and answers a scripted transcript.

    said: what the remote end prints, banner and all.
    """

    said: str = ""
    sent: list[tuple[tuple[str, ...], bytes]] = []

    def feed(
        self, command: tuple[str, ...], host: str, *, operation: str, chunks: Iterable[bytes]
    ) -> str:
        """Record the call and answer the transcript."""
        self.sent.append((command, b"".join(chunks)))
        return self.said


def test_modules_ship_the_census_before_the_agent_that_imports_it() -> None:
    """The loader registers modules in order, so the agent's relative import finds the census."""
    shipped = carrier.modules()
    assert list(shipped) == ["mainboard.probe.census", "mainboard.center.remote"]
    assert shipped["mainboard.center.remote"] == Path(remote.__file__).read_text(encoding="utf-8")
    assert shipped["mainboard.probe.census"] == Path(census.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("uv", ["uv", "/home/me/.local/bin/uv", r"C:\Users\me\.local\bin\uv.exe"])
@pytest.mark.parametrize(
    ("transport", "destination"),
    [
        (SshTransport(), "gold"),
        (SshTransport(endpoint=Endpoint(address="10.0.0.9", user="root")), "root@10.0.0.9"),
    ],
)
def test_the_argv_quotes_uv_and_carries_only_the_one_line_bootstrap(
    uv: str, transport: SshTransport, destination: str
) -> None:
    """A path with spaces or backslashes survives cmd.exe, PowerShell and a POSIX shell alike.

    The command line is the part of a call a process listing shows, so it is built from the host
    and uv alone and can hold neither a path from the call nor a secret.
    """
    argv = Carrier("gold", uv, transport).argv
    assert argv[: 1 + len(transport.options)] == ("ssh", *transport.options)
    assert argv[-2] == destination
    assert argv[-1] == (
        f'"{uv}" run --no-project --quiet --python {PYTHON} python -c "{carrier._BOOTSTRAP}"'
    )


def test_a_call_rides_stdin_and_answers_the_marked_line_typed_past_any_banner() -> None:
    """Arguments and stream travel on stdin only, and a login banner is never read as the answer.

    The header names the agent's modules, the function and its arguments, and the stream follows
    it byte for byte; the answer is validated into the shape the caller asked for.
    """
    transport = Recorder(said=f"Welcome to gold\n{MARKER}{json.dumps({'changed': 2})}\n", sent=[])
    answer = Carrier("gold", "uv", transport).call(
        "login", {"token": "ghp_secret"}, dict[str, int], stream=[b"tail", b"bytes"]
    )
    assert answer == {"changed": 2}
    [(argv, stdin)] = transport.sent
    assert "ghp_secret" not in " ".join(argv)
    loader, header, rest = stdin.split(b"\n", 2)
    assert loader.startswith(b"exec(")
    assert json.loads(header) == {
        "modules": carrier.modules(),
        "call": "login",
        "arguments": {"token": "ghp_secret"},
    }
    assert rest == b"tailbytes"


def test_an_answer_without_its_marker_is_a_mission_error_naming_the_host() -> None:
    """A shell that printed something else never ran the agent, so nothing is read out of it."""
    transport = Recorder(said="Last login: yesterday\nuv: command not found\n", sent=[])
    with pytest.raises(
        MissionError, match=r"(?s)'gold' answered where without a result.*not found"
    ):
        Carrier("gold", "uv", transport).call("where", {"root": "~"}, Destination)


def test_the_real_loader_runs_the_agent_end_to_end_on_its_own_stdin(
    home: Path, tmp_path: Path
) -> None:
    """The bootstrap, the loader and the shipped agent work as one on a Python with no tool.

    The child imports nothing of this package: it answers where it lives, takes a tar stream of
    files and writes them with the secret private, finds nothing left to send, and merges one
    JSON setting, each answer read back as its typed shape.
    """
    agent = Carrier("gold", "uv", Local())
    place = agent.call("where", {"root": "~/projects"}, Destination)
    assert place == Destination(
        root=str(home / "projects"), home=str(home), separator=os.sep, user=getpass.getuser()
    )
    (tmp_path / "receipts.ndjson").write_text("{}\n", encoding="utf-8")
    parcels = [
        Parcel(anchor="root", path=".env", data=b"KEY=1\n", secret=True),
        Parcel(
            anchor="root", path=".mainboard/receipts.ndjson", source=tmp_path / "receipts.ndjson"
        ),
        Parcel(anchor="home", path=".ssh/config", data=b"Host gold\n"),
    ]
    placed = agent.call(
        "place",
        {"root": place.root, "secret": [parcels[0].key]},
        dict[str, int],
        stream=packed(parcels),
    )
    assert placed == {"written": 3}
    assert (home / ".ssh" / "config").read_bytes() == b"Host gold\n"
    listing = [parcel.listing() for parcel in parcels]
    assert agent.call("inventory", {"root": place.root, "parcels": listing}, list[str]) == []
    entries = {place.root: {"hasTrustDialogAccepted": True}}
    merged = agent.call(
        "merge", {"path": ".claude.json", "key": "projects", "entries": entries}, dict[str, int]
    )
    assert merged == {"changed": 1}
    assert json.loads((home / ".claude.json").read_text(encoding="utf-8")) == {"projects": entries}
