"""What a terminal gets when `mb` prints to one, and who counts as an agent."""

import io
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.render import Audience, Streams, Terminal
from mainboard.verbs.enums import Scope
from mainboard.verbs.node import Command, Node, Typed
from mainboard.verbs.options import Dispatching

from ..constants import MB

_MARKERS = (
    "AI_AGENT",
    "CLAUDECODE",
    "CODEX_SANDBOX",
    "CODEX_CI",
    "CODEX_THREAD_ID",
    "OPENCODE",
    "GEMINI_CLI",
    "CURSOR_AGENT",
    "CLINE_ACTIVE",
)
# Variables that say nothing about who reads, including the `AGENT` rejected as ambiguous.
_NOISE = st.dictionaries(
    st.sampled_from(["PATH", "HOME", "TERM", "NO_COLOR", "FORCE_COLOR", "AGENT"]),
    st.text(max_size=4),
)
_STREAMS = st.builds(Streams, stdin=st.booleans(), stdout=st.booleans())
_AT_A_TERMINAL = Streams(stdin=True, stdout=True)
_BOXES = "╭╮╰╯│─┃━"
_ESC = "\x1b"
# A terminal on standard error, the one place the console renderer colors.
_LOGGING = """import sys
class Terminal:
    def isatty(self): return True
    def write(self, text): sys.__stderr__.write(text)
    def flush(self): pass
sys.stderr = Terminal()
from mb import logger
logger.info('reached', ok=True)
"""
# `job` made the owner's act, then the real entry point.
_OWNED = """import sys
import mainboard.cli as cli
from mainboard.verbs.node import Scope
node = cli.TREE.node("job")
commands = tuple(c.model_copy(update={"scope": Scope.OWNER}) for c in node.commands)
owned = node.model_copy(update={"commands": commands})
children = tuple(owned if child.name == "job" else child for child in cli.TREE.children)
cli.TREE = cli.TREE.model_copy(update={"children": children})
sys.argv = ["mb", *sys.argv[1:]]
cli.main()
"""
_SERVING = Node(
    name="mb",
    help="",
    children=(
        Node(
            name="serve",
            help="",
            commands=(
                Command(name="serve", default=True, scope=Scope.OWNER),
                Command(name="status", scope=Scope.AGENT),
                Command(name="stop", scope=Scope.OWNER),
            ),
        ),
    ),
)


class _Typing(io.StringIO):
    """Standard input at a terminal, holding what the person types."""

    def isatty(self) -> bool:
        return True


def _ran(
    *args: str, cwd: Path, env: Mapping[str, str], code: str = ""
) -> subprocess.CompletedProcess:
    """`mb` as a process in `cwd`, run as the entry point or through the Python `code` given."""
    entry = ["-c", code] if code else ["-m", "mainboard"]
    return subprocess.run(
        [sys.executable, *entry, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture
def clean(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """An environment no harness, color setting or `MB_AGENT` has touched."""
    for name in (*_MARKERS, "MB_AGENT", "MAINBOARD_AGENT", "FORCE_COLOR", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    return dict(os.environ)


@given(st.sampled_from(["1", "0"]), _NOISE, st.booleans())
@example(declared="1", noise={"AI_AGENT": "x"}, tty=True)
def test_mb_agent_decides_the_audience_before_anything_else(
    declared: str, noise: Mapping[str, str], tty: bool
) -> None:
    environ = {**noise, **dict.fromkeys(_MARKERS, "1"), "MB_AGENT": declared}
    expected = {"1": Audience.AGENT, "0": Audience.HUMAN}[declared]
    assert Audience.of(environ, tty=tty) is expected


@given(st.sampled_from(_MARKERS), st.sampled_from(["", "x", "2"]), _NOISE, st.booleans())
def test_a_harness_marker_makes_an_agent_even_at_a_terminal(
    marker: str, undecided: str, noise: Mapping[str, str], tty: bool
) -> None:
    environ = {**noise, marker: "1", "MB_AGENT": undecided}
    assert Audience.of(environ, tty=tty) is Audience.AGENT


@given(_NOISE, st.booleans())
@example(noise={"AGENT": "1"}, tty=True)
def test_without_marker_or_declaration_a_pipe_is_an_agent_and_a_terminal_a_person(
    noise: Mapping[str, str], tty: bool
) -> None:
    environ = {**noise, "AI_AGENT": "", "CLAUDECODE": ""}
    assert (Audience.of(environ, tty=tty) is Audience.HUMAN) == tty


@given(_STREAMS, st.sampled_from(["1", "0"]))
def test_only_a_person_with_a_terminal_on_stdin_can_be_asked(
    streams: Streams, declared: str
) -> None:
    terminal = Terminal.of({"MB_AGENT": declared}, streams)
    assert terminal.interactive == (declared == "0" and streams.stdin)


@given(st.sampled_from(_MARKERS), st.sampled_from(["0", "1", "", "x"]), _STREAMS)
@example(marker="CLAUDECODE", declared="0", streams=_AT_A_TERMINAL)
def test_a_harness_marker_refuses_an_owner_command_whatever_mb_agent_says(
    marker: str, declared: str, streams: Streams
) -> None:
    terminal = Terminal.of({marker: "1", "MB_AGENT": declared}, streams)
    with pytest.raises(MissionError, match="the owner's act"):
        terminal.refuse_agents("serve stop")


@pytest.mark.parametrize(
    ("agent", "force", "colored"),
    [
        pytest.param("1", "1", False, id="an-agent-is-never-colored-even-when-forced"),
        pytest.param("0", "1", True, id="a-person-is-colored-when-forced-without-a-terminal"),
        pytest.param("0", "", False, id="a-person-on-a-pipe-is-not-colored"),
    ],
)
def test_color_follows_the_audience_and_force_color(
    clean: Mapping[str, str],
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
    force: str,
    colored: bool,
) -> None:
    monkeypatch.setenv("FORCE_COLOR", force)
    terminal = Terminal.of({"MB_AGENT": agent}, Streams())
    console = terminal.console()
    with console.capture() as captured:
        console.print("word", style="bold")
    assert (_ESC in captured.get()) == colored == terminal.styled()


def test_printed_brackets_are_data_never_markup() -> None:
    console = Terminal.of({"MB_AGENT": "0"}, _AT_A_TERMINAL).console()
    with console.capture() as captured:
        console.print("[bold]word[/bold] [x]")
    assert captured.get() == "[bold]word[/bold] [x]\n"


def test_an_agent_gets_plain_help_and_a_person_rich_panels() -> None:
    agent = Terminal.of({"MB_AGENT": "1"}, _AT_A_TERMINAL)
    person = Terminal.of({"MB_AGENT": "0"}, _AT_A_TERMINAL)
    assert type(agent.formatter()).__name__ == "PlainFormatter"
    assert type(person.formatter()).__name__ == "DefaultFormatter"


@pytest.mark.parametrize(
    ("args", "agent", "force", "escapes"),
    [
        pytest.param(["--help"], "1", "1", False, id="agent-root-listing"),
        pytest.param(["job", "list", "--help"], "1", "1", False, id="agent-command-help"),
        pytest.param(["code", "doctor"], "1", "1", False, id="agent-report"),
        pytest.param(["code", "doctor", "--bogus"], "1", "1", False, id="agent-unknown-option"),
        pytest.param(["--tree", "code"], "1", "1", False, id="agent-tree"),
        pytest.param(["--help"], "0", "1", True, id="person-root-listing"),
        pytest.param(["code", "doctor"], "0", "1", True, id="person-report-table"),
        pytest.param(
            ["code", "doctor", "--bogus"], "0", "1", True, id="person-unknown-option-panel"
        ),
        pytest.param(["--tree", "code"], "0", "1", True, id="person-tree-guides"),
        pytest.param(["code", "doctor"], "0", "", False, id="person-report-without-forced-color"),
    ],
)
def test_escapes_and_boxes_reach_only_a_person_who_can_see_them(
    workspace: Path,
    clean: Mapping[str, str],
    args: list[str],
    agent: str,
    force: str,
    escapes: bool,
) -> None:
    ran = _ran(*args, cwd=workspace, env={**clean, "MB_AGENT": agent, "FORCE_COLOR": force})
    printed = ran.stdout + ran.stderr
    assert (_ESC in printed) == escapes
    assert any(box in printed for box in _BOXES) == (agent == "0")


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["data", "query", "SELECT 1 AS answer", "--json"], id="a-query"),
        pytest.param(["--tree", "--json"], id="the-declaration"),
    ],
)
def test_json_is_identical_for_both_audiences(
    workspace: Path, clean: Mapping[str, str], args: list[str]
) -> None:
    agent = _ran(*args, cwd=workspace, env={**clean, "MB_AGENT": "1", "FORCE_COLOR": "1"})
    person = _ran(*args, cwd=workspace, env={**clean, "MB_AGENT": "0", "FORCE_COLOR": "1"})
    assert agent.stdout == person.stdout
    assert json.loads(agent.stdout)
    assert agent.returncode == person.returncode == 0


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param({"MB_AGENT": "1"}, id="an-agent-audience"),
        pytest.param({"CLAUDECODE": "1", "MB_AGENT": "0"}, id="a-harness-declaring-a-person"),
        pytest.param({"MB_AGENT": "0"}, id="a-person-style-with-no-terminal-on-stdin"),
    ],
)
def test_an_owner_command_refuses_without_a_person_at_a_terminal(
    workspace: Path, clean: Mapping[str, str], declared: Mapping[str, str]
) -> None:
    ran = _ran("job", "list", cwd=workspace, env={**clean, **declared}, code=_OWNED)
    assert ran.returncode == 1
    assert "the owner's act" in ran.stderr
    assert ran.stdout == ""


def test_the_owner_at_a_real_terminal_runs_an_owner_command(
    at_a_terminal, clean: Mapping[str, str]
) -> None:
    ran = at_a_terminal([sys.executable, "-c", _OWNED, "job", "list"], env=clean)
    assert ran.code == 0, ran.said
    assert "the owner's act" not in ran.said
    assert "Traceback" not in ran.said


@pytest.mark.parametrize(
    ("argv", "owned"),
    [
        pytest.param(["serve"], "serve", id="the-default-command-is-named-by-its-node"),
        pytest.param(["serve", "stop"], "serve stop", id="a-named-owner-command"),
        pytest.param(["serve", "status"], "", id="an-agent-command-is-open"),
        pytest.param(["serve", "stop", "--help"], "", id="help-is-always-open"),
        pytest.param([], "", id="the-bare-tool-is-open"),
    ],
)
def test_the_declaration_names_the_owner_command_a_line_runs(argv: list[str], owned: str) -> None:
    assert Typed(node=_SERVING).owned(argv) == owned


@pytest.mark.parametrize(
    ("environ", "streams"),
    [
        pytest.param({}, _AT_A_TERMINAL, id="a-person-at-a-terminal"),
        pytest.param({"MB_AGENT": "0"}, _AT_A_TERMINAL, id="a-declared-person-at-a-terminal"),
        pytest.param({"MB_AGENT": "0"}, Streams(stdin=True), id="a-declared-person-piping-out"),
    ],
)
def test_a_person_typing_at_a_terminal_runs_an_owner_command(
    environ: Mapping[str, str], streams: Streams
) -> None:
    Terminal.of(environ, streams).refuse_agents("serve stop")


@pytest.mark.parametrize(
    ("environ", "streams"),
    [
        pytest.param({"MB_AGENT": "1"}, _AT_A_TERMINAL, id="an-agent-audience-at-a-terminal"),
        pytest.param({}, Streams(stdin=True), id="an-undeclared-pipe-is-an-agent"),
        pytest.param({"MB_AGENT": "0"}, Streams(stdout=True), id="no-terminal-on-stdin"),
    ],
)
def test_an_owner_command_needs_a_person_at_a_terminal(
    environ: Mapping[str, str], streams: Streams
) -> None:
    with pytest.raises(MissionError, match="`mb serve stop` is the owner's act"):
        Terminal.of(environ, streams).refuse_agents("serve stop")


@pytest.mark.parametrize(
    ("agent", "typed", "proceeds"),
    [
        pytest.param("1", "", True, id="an-agent-is-never-asked"),
        pytest.param("0", "y\n", True, id="a-person-answering-yes-dispatches"),
        pytest.param("0", "\n", False, id="a-person-answering-nothing-does-not"),
    ],
)
def test_only_a_person_at_a_terminal_is_asked_before_a_dispatch(
    monkeypatch: pytest.MonkeyPatch, agent: str, typed: str, proceeds: bool
) -> None:
    monkeypatch.setenv("MB_AGENT", agent)
    monkeypatch.setattr(sys, "stdin", _Typing(typed))
    assert Dispatching().proceeds() == proceeds


@pytest.mark.parametrize(
    ("env", "colored"),
    [
        pytest.param({}, True, id="a-terminal-log-is-colored"),
        pytest.param({"NO_COLOR": "1"}, False, id="no-color-keeps-it-plain"),
    ],
)
def test_logging_to_a_terminal_starts_on_every_platform(
    clean: Mapping[str, str], env: Mapping[str, str], colored: bool
) -> None:
    """The console renderer starts on a terminal; it once crashed every command there."""
    done = subprocess.run(
        [sys.executable, "-c", _LOGGING],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**clean, **env},
    )
    assert done.returncode == 0, done.stderr
    assert "reached" in done.stderr
    assert (_ESC in done.stderr) == colored
    assert MB is not None
