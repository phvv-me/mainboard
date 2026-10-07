"""`.agents` rendered into every harness's own files, and judged, on a real git checkout."""

import json
import subprocess
import tomllib
from pathlib import Path

import pytest

from mainboard.agents import Agents
from mainboard.core import MissionError
from mainboard.core.section import Verdict

_SERVERS = {
    "local": {"command": "mainboard", "args": ["run", "--", "tool"], "env": {"TOKEN": "${TOKEN}"}},
    "remote": {
        "type": "http",
        "url": "https://example.test/mcp",
        "headers": {"Authorization": "Bearer ${SECRET}", "X-Plain": "yes"},
        "overrides": {"codex": {"scopes": ["read"]}},
    },
}
_HOOKS = {
    "PreToolUse": [
        {"matcher": "Bash|Edit", "hooks": [{"type": "command", "command": "guard", "timeout": 5}]}
    ],
    "Notification": [{"hooks": [{"type": "command", "command": "notify"}]}],
}
# A description Claude Code accepts and YAML does not: a colon inside, a newline written out.
_CARD = "\n".join(
    [
        "---",
        "name: helper",
        "description: Use it: when a colon appears\\nand more",
        "tools: Read",
        "---",
        "",
        "Do the job.",
        "",
    ]
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())


def _agents(root: Path, environment: dict[str, str] | None = None) -> Agents:
    return Agents(
        root,
        home=root / "home",
        environment=environment or {},
        dotenv={},
        which=lambda program: f"/bin/{program}" if program == "mainboard" else None,
    )


def _read(root: Path, path: str) -> dict:
    text = (root / path).read_text()
    return tomllib.loads(text) if path.endswith(".toml") else json.loads(text)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _write(tmp_path / "AGENTS.md", "Be careful.\n")
    _write(tmp_path / ".agents/mcp.json", json.dumps({"mcpServers": _SERVERS}))
    _write(tmp_path / ".agents/settings.json", json.dumps({"hooks": _HOOKS}))
    _write(tmp_path / ".agents/agents/helper.md", _CARD)
    _write(tmp_path / ".agents/codex.toml", 'model = "m"\n')
    _write(tmp_path / ".agents/opencode.json", json.dumps({"lsp": {}}))
    return tmp_path


@pytest.fixture
def synced(root: Path) -> Path:
    _agents(root).sync()
    return root


def test_claude_reads_agents_through_its_link_and_the_servers_verbatim(synced: Path) -> None:
    assert (synced / ".claude/agents/helper.md").is_file()
    assert (synced / "CLAUDE.md").read_text() == "@AGENTS.md\n"
    assert _read(synced, ".mcp.json")["mcpServers"]["remote"] == {
        key: _SERVERS["remote"][key] for key in ("type", "url", "headers")
    }
    assert _agents(synced).drift() == []


def test_codex_forwards_variables_by_name_and_keeps_its_own_settings(synced: Path) -> None:
    config = _read(synced, ".codex/config.toml")

    assert config["model"] == "m"
    assert config["mcp_servers"]["local"]["env_vars"] == ["TOKEN"]
    assert config["mcp_servers"]["remote"] == {
        "url": "https://example.test/mcp",
        "http_headers": {"X-Plain": "yes"},
        "bearer_token_env_var": "SECRET",
        "scopes": ["read"],
    }
    assert set(_read(synced, ".codex/hooks.json")["hooks"]) == {"PreToolUse"}
    assert _read(synced, ".codex/agents/helper.toml")["developer_instructions"] == "Do the job.\n"


def test_opencode_spells_references_its_own_way_and_keeps_its_settings(synced: Path) -> None:
    opencode = _read(synced, "opencode.json")

    assert opencode["lsp"] == {}
    assert opencode["mcp"]["local"]["environment"] == {"TOKEN": "{env:TOKEN}"}
    assert opencode["mcp"]["remote"]["headers"]["Authorization"] == "Bearer {env:SECRET}"
    assert "Use it: when a colon appears" in (synced / ".opencode/agents/helper.md").read_text()


def test_sync_replaces_a_link_without_writing_through_it(root: Path) -> None:
    (root / ".codex").mkdir()
    (root / ".codex/config.toml").symlink_to("../.agents/codex.toml")

    _agents(root).sync()

    assert (root / ".agents/codex.toml").read_text() == 'model = "m"\n'
    assert not (root / ".codex/config.toml").is_symlink()


def test_sync_removes_what_agents_no_longer_declares(synced: Path) -> None:
    (synced / ".agents/agents/helper.md").unlink()

    changed = _agents(synced).sync()

    assert {".codex/agents/helper.toml", ".opencode/agents/helper.md"} <= set(changed)
    assert not (synced / ".opencode/agents/helper.md").exists()


def test_codex_refuses_a_variable_forwarded_under_another_name(root: Path) -> None:
    servers = {"local": {"command": "tool", "env": {"TOKEN": "${OTHER}"}}}
    _write(root / ".agents/mcp.json", json.dumps({"mcpServers": servers}))

    with pytest.raises(MissionError, match="its own name"):
        _agents(root).rendered()


def test_check_names_drift_unset_variables_and_absent_harnesses(root: Path) -> None:
    sections = {row.section: row for row in _agents(root, {"TOKEN": "t"}).sections()}

    assert sections["agents: files"].verdict is Verdict.WARN
    assert sections["agents: servers"].detail == "remote: SECRET is undefined"
    assert sections["agents: claude"].fix.endswith("agents update")
