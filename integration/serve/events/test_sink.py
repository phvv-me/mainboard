"""The sink: its append-only token, a loopback start with no secret, and the doctor's verdict on
whether this tool's DuckDB still reads the lake the binary writes."""

from collections.abc import Callable
from pathlib import Path

import duckdb
import pytest

from mainboard import MissionError
from mainboard.core import Verdict
from mainboard.serve.compose import Compose
from mainboard.serve.events.sink import Sink

from ...state.telemetry.builders import lake_with
from ..fakes import FakeSink, binaries

_CAPS = {
    "DUCKDB_OTLP_HTTP_THREADS",
    "DUCKDB_OTLP_MAX_BUFFERED_BYTES",
    "DUCKDB_OTLP_SEAL_TARGET_BYTES",
}


def test_a_token_is_made_once_private_and_long_enough_for_the_binary(workspace: Path) -> None:
    sink = Sink(workspace)
    made = sink.token.ensure()
    modes = [sink.token.path.stat().st_mode, sink.token.path.parent.stat().st_mode]
    assert (sink.token.ensure(), len(made) >= 16) == (made, True)
    assert [mode & 0o777 for mode in modes] == [0o600, 0o700]


@pytest.mark.parametrize(
    ("mode", "text", "said"),
    [
        pytest.param(0o644, "a" * 43, "readable beyond its owner", id="a-token-others-may-read"),
        pytest.param(0o600, "short", "under the 16", id="a-token-too-short-to-be-one"),
    ],
)
def test_a_token_others_may_read_or_too_short_is_refused_and_kept(
    workspace: Path, mode: int, text: str, said: str
) -> None:
    sink = Sink(workspace)
    sink.token.path.parent.mkdir(parents=True)
    sink.token.path.write_text(text, encoding="utf-8")
    sink.token.path.chmod(mode)
    with pytest.raises(MissionError, match=said):
        sink.token.ensure()
    assert sink.token.path.read_text(encoding="utf-8") == text


def test_the_sink_listens_on_loopback_alone_with_its_files_in_the_lake(workspace: Path) -> None:
    sink = Sink(workspace)
    command = sink.command(Path("/opt/duckdb-otlp"))
    paths = [Path(word) for word in command[2:] if word.startswith("/")]
    assert command[command.index("--host") + 1 : command.index("--grpc") + 2] == [
        *("127.0.0.1", "--http", "4318", "--grpc", "0")
    ]
    assert (len(paths), all(path.is_relative_to(sink.lake.home) for path in paths)) == (4, True)


def test_the_sink_attaches_the_telemetry_catalog_and_nothing_else(workspace: Path) -> None:
    sink = Sink(workspace)
    script = sink.script()
    assert script.count("ATTACH") == 1
    assert f"ducklake:sqlite:{sink.lake.attachment.catalog}" in script


def test_the_sink_environment_is_the_allowlist_its_caps_and_its_token_alone(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "secret")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://elsewhere:4318/v1/logs")
    environment = Sink(workspace).environment("t" * 43)
    assert set(environment) == {*Compose.allowed(), *_CAPS, "DUCKDB_OTLP_TOKEN"}


@pytest.mark.parametrize(
    ("embedded", "reader", "verdict", "phrase"),
    [
        pytest.param("", "2.0.0", Verdict.WARN, "not installed", id="no-binary-here"),
        pytest.param(
            "1.4.2",
            "2.0.0",
            Verdict.FAIL,
            "no proven pairing",
            id="a-binary-on-a-duckdb-no-pairing-names",
        ),
        pytest.param(
            "1.5.6", "2.1.0", Verdict.FAIL, "outran the sink", id="a-duckdb-upgrade-outran-it"
        ),
        pytest.param("1.5.6", "2.0.3", Verdict.PASS, "no lake yet", id="paired-and-no-lake"),
    ],
)
def test_the_doctor_row_fails_loudly_when_duckdb_outruns_the_sink(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    embedded: str,
    reader: str,
    verdict: Verdict,
    phrase: str,
) -> None:
    resolver = binaries(workspace, embedded)
    monkeypatch.setattr(duckdb, "__version__", reader)
    row = Sink(workspace).section(resolver)
    fallback = "~/.claude/projects" in row.fix
    assert (row.verdict, phrase in row.detail, fallback) == (
        verdict,
        True,
        verdict is Verdict.FAIL,
    )


def test_the_doctor_row_warns_when_the_only_binary_is_a_twin_of_the_pin(workspace: Path) -> None:
    resolver = binaries(workspace, pinned=False)
    row = Sink(workspace).section(resolver)
    assert (row.verdict, "will not start it" in row.detail) == (Verdict.WARN, True)
    with pytest.raises(MissionError, match="pinned but not installed from its archive"):
        resolver.pinned("duckdb-otlp")


def test_the_doctor_row_fails_a_lake_this_duckdb_migrated_past_the_sink(workspace: Path) -> None:
    resolver = binaries(workspace)
    lake_with(workspace, [])
    row = Sink(workspace).section(resolver)
    assert (row.verdict, "a newer DuckDB migrated it" in row.detail) == (Verdict.FAIL, True)


def test_the_sink_answers_its_health_check_only_while_it_listens(
    workspace: Path, reach: Callable[[Path, Path], Sink]
) -> None:
    with FakeSink(workspace / "sink.sock") as fake:
        sink = reach(workspace, fake.path)
        answered = sink.answering()
    assert (answered, sink.answering(), sink.awaited(0.3)) == (True, False, False)
