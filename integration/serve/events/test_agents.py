"""The user-level telemetry settings: only the keys this tool owns change, under the home it is
given, the token only where Codex needs it, and the headers helper reads it from its file."""

import json
import subprocess
import sys
import tomllib
from collections.abc import Collection, Mapping
from inspect import getfile
from pathlib import Path

import pytest
import tomlkit
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.core import Verdict
from mainboard.serve.events import headers
from mainboard.serve.events.agents import AgentTelemetry, ClaudeExporter, CodexExporter
from mainboard.serve.events.sink import Sink

_KEYS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8)
_TEXT = st.text(st.characters(codec="utf-8", categories=["L", "N", "P", "Zs"]), max_size=8)
_SCALARS = _TEXT | st.integers(-1000, 1000) | st.booleans()
# What a person's settings hold beside this tool's keys, in either file.
_SETTINGS = st.dictionaries(
    _KEYS.filter(lambda key: key not in {"env", "otel"}),
    _SCALARS | st.lists(_SCALARS, max_size=3),
    max_size=5,
)
_ENVIRONMENT = st.dictionaries(_KEYS.map(str.upper), _TEXT, max_size=4)
# The `[otel]` keys this tool sets.
_CODEX_OWNED = {"exporter", "log_agent_responses", "log_user_prompt", "tool_result"}


def _unowned[Value](document: Mapping[str, Value], owned: Collection[str]) -> dict[str, Value]:
    return {key: value for key, value in document.items() if key not in owned}


@given(kept=_SETTINGS, environment=_ENVIRONMENT)
@example(kept={"model": "opus"}, environment={"CLAUDE_CODE_ENABLE_TELEMETRY": "0"})
@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_claude_settings_change_only_the_keys_this_tool_owns(
    workspace: Path, kept: dict, environment: dict[str, str]
) -> None:
    sink, before = Sink(workspace), {**kept, "env": environment}
    sink.token.ensure()
    owned = set(json.loads(ClaudeExporter().merged("", sink))["env"])
    merged = ClaudeExporter().merged(json.dumps(before), sink)
    after = json.loads(merged)
    assert ClaudeExporter().merged(merged, sink) == merged
    assert _unowned(after, {"env", "otelHeadersHelper"}) == _unowned(before, {"env"})
    assert _unowned(after["env"], owned) == _unowned(environment, owned)


@given(kept=_SETTINGS, otel=st.dictionaries(_KEYS, _SCALARS, max_size=3))
@example(kept={"model": "gpt-6.1-sol"}, otel={"environment": "center", "log_user_prompt": True})
@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_codex_config_changes_only_its_otel_keys(workspace: Path, kept: dict, otel: dict) -> None:
    sink = Sink(workspace)
    sink.token.ensure()
    before = "# the owner's comment\n" + tomlkit.dumps({**kept, "otel": otel})
    merged = CodexExporter().merged(before, sink)
    after, earlier = tomllib.loads(merged), tomllib.loads(before)
    assert CodexExporter().merged(merged, sink) == merged and merged.startswith("# the owner's")
    assert _unowned(after, {"otel"}) == _unowned(earlier, {"otel"})
    assert _unowned(after["otel"], _CODEX_OWNED) == _unowned(earlier["otel"], _CODEX_OWNED)
    assert after["otel"]["log_user_prompt"] is False


def test_a_sync_writes_both_files_under_the_home_given_and_the_token_only_to_codex(
    workspace: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    home, sink = tmp_path_factory.mktemp("home"), Sink(workspace)
    rows = AgentTelemetry(sink, home=home).sync()
    files = sorted(path.relative_to(home).as_posix() for path in home.rglob("*") if path.is_file())
    token, texts = sink.token.read(), [(home / name).read_text(encoding="utf-8") for name in files]
    assert files == [".claude/settings.json", ".codex/config.toml"]
    assert ([token in text for text in texts], [row.verdict for row in rows]) == (
        [False, True],
        [Verdict.PASS, Verdict.PASS],
    )
    assert (home / ".codex/config.toml").stat().st_mode & 0o777 == 0o600


def test_a_second_sync_finds_both_files_current_and_rewrites_neither(
    workspace: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    home = tmp_path_factory.mktemp("home")
    telemetry = AgentTelemetry(Sink(workspace), home=home)
    telemetry.sync()
    stamps = [path.stat().st_mtime_ns for path in sorted(home.rglob("*.*"))]
    rows = telemetry.sync()
    assert [path.stat().st_mtime_ns for path in sorted(home.rglob("*.*"))] == stamps
    assert all(row.detail.endswith("current") for row in rows)


def _refused(sink: Sink, home: Path, *, name: str, text: str, match: str) -> None:
    """The sync refuses with `match` and leaves the file `name` under `home` as it was."""
    (home / name).parent.mkdir()
    (home / name).write_text(text, encoding="utf-8")
    with pytest.raises(MissionError, match=match):
        AgentTelemetry(sink, home=home).sync()
    assert (home / name).read_text(encoding="utf-8") == text


def test_claude_settings_that_are_not_an_object_are_refused_and_left_alone(
    workspace: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    home = tmp_path_factory.mktemp("home")
    _refused(
        Sink(workspace),
        home,
        name=".claude/settings.json",
        text="[1, 2]\n",
        match="not a JSON object",
    )
    assert not (home / ".codex").exists()


def test_a_codex_config_that_is_not_toml_is_refused_and_left_alone(
    workspace: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    _refused(
        Sink(workspace),
        tmp_path_factory.mktemp("home"),
        name=".codex/config.toml",
        text="model = \n",
        match="not valid TOML",
    )


@pytest.mark.parametrize(
    ("token", "printed"),
    [
        pytest.param("t" * 43, {"Authorization": f"Bearer {'t' * 43}"}, id="a-token-in-its-file"),
        pytest.param("", {}, id="no-token-file-at-all"),
    ],
)
def test_the_headers_helper_prints_the_bearer_header_or_none_without_a_token(
    tmp_path: Path, token: str, printed: dict[str, str]
) -> None:
    held = tmp_path / "otlp.token"
    held.write_text(token, encoding="utf-8")
    said = subprocess.run(
        [sys.executable, "-I", getfile(headers), str(held if token else tmp_path / "absent")],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert (json.loads(said.stdout), said.stderr) == (printed, "")
