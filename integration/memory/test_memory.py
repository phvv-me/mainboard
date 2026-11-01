"""The front of memory: a keep is spooled before it is delivered, delivered once, and delivered
before any question reaches the upstream; a keep the upstream refuses or a spool entry that holds
no keep is set aside and never stops the rest."""

import importlib.util
import string
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from mcp.shared.exceptions import MCPError
from mcp_types import INTERNAL_ERROR, INVALID_PARAMS

from mainboard.core import Refused, Unavailable, Unreachable
from mainboard.memory.enums import State
from mainboard.memory.keeps import Keep
from mainboard.memory.outbox import Memory, Outbox

from .fakes import Recorder

_WORDS = st.text(alphabet=string.ascii_letters, min_size=1, max_size=8)
_BYTES = b"\x89PNG\x00\xff\xfe\x01"
_EXTRA = "M00-MEMORY: the memory extra's aizk_client is not installed beside this interpreter"


def test_a_keep_made_while_the_upstream_is_down_is_delivered_once_when_it_returns(
    workspace: Path,
) -> None:
    upstream = Recorder(down=True)
    memory = Memory(Outbox(workspace), upstream)
    queued = memory.keep([Keep.of("a note")])
    upstream.down = False
    memory.keep([Keep.of("a note")])
    assert ([outcome.state for outcome in queued], len(upstream.kept)) == ([State.QUEUED], 1)


def test_the_same_keep_given_twice_adds_one_record_and_is_known_the_second_time(
    workspace: Path,
) -> None:
    upstream = Recorder()
    memory = Memory(Outbox(workspace), upstream)
    first = memory.keep([Keep.of("a note")])
    second = memory.keep([Keep.of("a note", note="said again")])
    states = [first[0].state, second[0].state]
    assert (states, len(upstream.kept)) == ([State.KEPT, State.KNOWN], 1)
    assert second[0].document == first[0].document


def test_the_upstream_receives_a_queued_keep_before_the_find(workspace: Path) -> None:
    upstream = Recorder(down=True)
    memory = Memory(Outbox(workspace), upstream)
    memory.keep([Keep.of("remember gold")])
    upstream.down = False
    answer = memory.find("what is gold")
    assert (upstream.calls, answer) == (
        [("keep", "remember gold"), ("find", "what is gold")],
        "answer to what is gold",
    )


def test_a_find_while_the_upstream_is_down_is_refused_and_loses_no_keep(workspace: Path) -> None:
    upstream = Recorder(down=True)
    memory = Memory(Outbox(workspace), upstream)
    memory.keep([Keep.of("a note"), Keep.of("another")])
    with pytest.raises(Unreachable):
        memory.find("anything")
    assert (upstream.calls, len(memory.outbox.waiting())) == ([], 2)


def test_a_drain_delivers_the_spool_oldest_first_and_stops_where_the_upstream_fails(
    workspace: Path,
) -> None:
    upstream = Recorder(down=True)
    memory = Memory(Outbox(workspace), upstream)
    memory.keep([Keep.of("first"), Keep.of("second"), Keep.of("third")])
    upstream.down = False
    delivered = memory.drain()
    assert (delivered, [call[1] for call in upstream.calls]) == (3, ["first", "second", "third"])


def test_a_keep_the_upstream_refuses_is_set_aside_and_the_rest_are_delivered(
    workspace: Path,
) -> None:
    upstream = Recorder(down=True, refusing={"first"})
    memory = Memory(Outbox(workspace), upstream)
    memory.keep([Keep.of("first"), Keep.of("second")])
    upstream.down = False
    outcomes = memory.keep([Keep.of("first"), Keep.of("third")])
    assert ([call[1] for call in upstream.calls], [outcome.state for outcome in outcomes]) == (
        ["second", "third"],
        [State.REFUSED, State.KEPT],
    )
    assert (memory.outbox.waiting(), len(list(memory.outbox.spool.glob("*.refused")))) == ([], 1)


def test_a_file_keep_whose_copy_is_gone_is_set_aside_and_the_good_keep_and_find_go_through(
    workspace: Path, tmp_path: Path
) -> None:
    picture = tmp_path / "shot.png"
    picture.write_bytes(_BYTES)
    upstream = Recorder(down=True)
    memory = Memory(Outbox(workspace), upstream)
    memory.keep([Keep.of(str(picture))])
    memory.keep([Keep.of("a good note")])
    next(memory.outbox.spool.glob("*.blob")).unlink()
    upstream.down = False
    answer = memory.find("what was kept")
    assert (upstream.calls, answer) == (
        [("keep", "a good note"), ("find", "what was kept")],
        "answer to what was kept",
    )


@pytest.mark.skipif(
    importlib.util.find_spec("aizk_client") is not None,
    reason="the memory extra is installed here, so nothing refuses",
)
def test_the_aizk_upstream_is_refused_by_naming_the_extra_that_carries_it() -> None:
    with pytest.raises(Unavailable, match=r"mainboard\[memory\]"):
        Memory.aizk()


def test_the_aizk_upstream_with_a_corrupt_profile_says_to_move_it_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("aizk_client", reason=_EXTRA)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    tmp_path.joinpath("aizk").mkdir()
    tmp_path.joinpath("aizk", "profile.json").write_text('{"server": 12', encoding="utf-8")
    with pytest.raises(Unavailable, match="move it aside"):
        Memory.aizk()


def test_the_aizk_upstream_queues_only_what_can_wait_and_refuses_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = pytest.importorskip("aizk_client", reason=_EXTRA)
    aizk = pytest.importorskip("mainboard.memory.aizk", reason=_EXTRA)
    client = wire.MemoryClient(wire.ClientProfile(server="http://127.0.0.1:9/mcp", auth="none"))
    ticket = wire.UploadTicketAccepted(upload_url="http://127.0.0.1:9/u", expires_seconds=60)
    answers = [
        (Unreachable, AsyncMock(side_effect=TimeoutError())),
        (Unreachable, AsyncMock(side_effect=wire.LoginRequiredError("expired"))),
        (Unreachable, AsyncMock(side_effect=MCPError(INTERNAL_ERROR, "503"))),
        (Refused, AsyncMock(side_effect=MCPError(INVALID_PARAMS, "no such scope"))),
        (Refused, AsyncMock(side_effect=ToolError("keep requires text"))),
        (Refused, AsyncMock(side_effect=wire.ProtocolError("a ticket without an upload"))),
        (Refused, AsyncMock(return_value=ticket)),
    ]
    for raised, keep in answers:
        monkeypatch.setattr(client, "keep", keep)
        with pytest.raises(raised):
            aizk.Aizk(client).keep(Keep.of("a note"))


class Keeping(RuleBasedStateMachine):
    """Keeps made and drained while the upstream comes and goes: each reaches it once."""

    @invariant()
    def each_keep_is_delivered_at_most_once_and_the_rest_wait(self) -> None:
        delivered = [keep.content for keep in self.upstream.kept]
        waiting = [waiting.keep.content for waiting in self.memory.outbox.waiting()]
        assert len(delivered) == len(set(delivered))
        assert sorted([*delivered, *waiting]) == sorted(self.made)

    @precondition(lambda self: not self.upstream.down)
    @rule()
    def drain(self) -> None:
        self.memory.drain()

    @rule(down=st.booleans())
    def toggle(self, *, down: bool) -> None:
        self.upstream.down = down

    @rule(word=_WORDS)
    def keep(self, word: str) -> None:
        self.memory.keep([Keep.of(word)])
        self.made.add(word)

    def __init__(self) -> None:
        super().__init__()
        self.home = TemporaryDirectory()
        self.upstream = Recorder()
        self.memory = Memory(Outbox(Path(self.home.name)), self.upstream)
        self.made: set[str] = set()

    def teardown(self) -> None:
        self.home.cleanup()


TestKeeping = Keeping.TestCase
TestKeeping.settings = settings(max_examples=25, stateful_step_count=12)
