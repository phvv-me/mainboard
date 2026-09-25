import json
import shutil
import string
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.center import agents as module
from mainboard.center.agents import Agents, dotenv, junction
from mainboard.center.state import claude_key
from mainboard.core.section import Verdict

from ..git.conftest import Forge

# The programs this fake machine has on its PATH.
_ON_PATH = frozenset({"mainboard", "present", "uvx"})

# The tracked links of an agent workspace: `.claude` to a directory, the Codex config to a file,
# one to nothing and one out of the workspace, which no repair may follow.
_LINKS = {
    ".claude": ".agents",
    ".codex/config.toml": "../.agents/codex.toml",
    "dangling": "missing",
    "escape": "..",
}

# The workspace files every agent reads, as the monorepo tracks them.
_FILES = {
    "AGENTS.md": "# Workspace\n",
    "CLAUDE.md": "@AGENTS.md\n",
    ".agents/settings.json": "{}",
    ".agents/codex.toml": '[mcp_servers.docs]\ncommand = "present"\n',
}

# A variable name as every agent's configuration spells one.
_NAMES = st.from_regex(r"[A-Z_][A-Z0-9_]{0,6}", fullmatch=True)

# A `.env` value: anything but a line break, which would end the line it sits on.
_VALUES = st.text(string.ascii_letters + string.digits + " =:/._-#", max_size=12)


def agents(
    root: Path,
    home: Path,
    *,
    environment: Mapping[str, str] | None = None,
    env_file: Mapping[str, str] | None = None,
    junction: Callable[[Path, Path], str] | None = None,
) -> Agents:
    """The agent checks of `root` on a machine whose PATH holds `_ON_PATH`."""
    return Agents(
        root,
        home=home,
        environment=environment or {},
        dotenv=env_file or {},
        which=lambda program: f"/bin/{program}" if program in _ON_PATH else None,
        junction=junction or copied,
    )


def copied(link: Path, target: Path) -> str:
    """A junction that always succeeds, standing in with a copy of the directory."""
    shutil.copytree(target, link)
    return ""


def written(root: Path, files: Mapping[str, str]) -> Path:
    """`files` written under `root`, parents made."""
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def flattened(tmp_path: Path) -> Path:
    """A workspace checked out by a git that could not make links: each is a text file.

    The index records every link with git's link mode, the way the monorepo tracks them, and
    the disk holds the target's path as plain text, which is what a Windows account without
    Developer Mode gets. No real link is ever needed, so this runs the same on every system.
    """
    return flatten(tmp_path / "work", _LINKS)


def flatten(root: Path, links: Mapping[str, str]) -> Path:
    """`root` holding the agent files and `links` tracked as links but written as text files."""
    written(root, _FILES)
    Forge.git(root, "init", "-q")
    Forge.git(root, "add", "-A")
    written(root, links)
    for relative in links:
        blob = Forge.git(root, "hash-object", "-w", "--", relative)
        Forge.git(root, "update-index", "--add", "--cacheinfo", f"120000,{blob},{relative}")
    return root


def skipped(root: Path) -> set[str]:
    """The paths git was told to stop comparing."""
    listed = Forge.git(root, "ls-files", "-v").splitlines()
    return {line[2:] for line in listed if line.startswith("S ")}


def test_links_git_flattened_are_stood_in_for_before_anything_reads_through_them(
    flattened: Path, home: Path
) -> None:
    """A directory link becomes a junction and a file link a hard link, then git looks away.

    Links to nothing or out of the workspace are left alone, since no stand-in could be right.
    Every later check reads through the repaired links, so the Codex servers behind the file
    link are found, and a second run finds nothing left to repair.
    """
    checks = agents(flattened, home)

    rows = checks.sections()

    assert [row.section for row in rows] == [
        "agents: links",
        "agents: instructions",
        "agents: .mcp.json",
        "agents: opencode.json",
        "agents: .codex/config.toml",
        "agents: memory",
        "agents: logins",
    ]
    assert rows[0].verdict == Verdict.PASS
    assert rows[0].detail == (
        "stood in for 2 links git could not make: .claude, .codex/config.toml"
    )
    assert rows[1].verdict == rows[4].verdict == Verdict.PASS
    assert rows[4].detail == "1 servers can start here"
    assert (flattened / ".claude" / "settings.json").is_file()
    assert (flattened / ".codex" / "config.toml").samefile(flattened / ".agents" / "codex.toml")
    assert skipped(flattened) == {".claude", ".codex/config.toml"}
    assert (flattened / "dangling").read_text(encoding="utf-8") == "missing"
    assert checks.links().detail == "every tracked link resolves"


def test_a_directory_link_that_cannot_be_stood_in_for_fails_and_is_left_as_it_was(
    flattened: Path, home: Path
) -> None:
    """A refused junction puts the text file back and fails with the switch that allows links."""
    row = agents(flattened, home, junction=lambda link, target: "Access is denied.").links()

    assert row.verdict == Verdict.FAIL
    assert row.detail == "cannot stand in for .claude (Access is denied.)"
    assert row.fix == "enable Developer Mode, then git checkout -- <path>"
    assert (flattened / ".claude").read_text(encoding="utf-8") == ".agents"
    assert skipped(flattened) == {".codex/config.toml"}


def test_a_refused_hard_link_leaves_the_tracked_file_exactly_as_git_wrote_it(
    flattened: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is never removed before its stand-in exists, so a refusal costs nothing."""

    def refuse(target: Path, link: Path) -> None:
        raise PermissionError("links are not allowed here")

    monkeypatch.setattr(module.os, "link", refuse)

    row = agents(flattened, home).links()

    assert row.verdict == Verdict.FAIL
    assert row.detail == ("cannot stand in for .codex/config.toml (links are not allowed here)")
    assert (flattened / ".codex" / "config.toml").read_text(encoding="utf-8") == (
        "../.agents/codex.toml"
    )
    assert not list(flattened.rglob("*.mainboard-link"))


@given(depth=st.integers(min_value=1, max_value=3))
def test_a_refused_nested_link_keeps_its_own_text_and_is_found_again(
    tmp_path_factory: pytest.TempPathFactory, depth: int
) -> None:
    """A link nested any depth keeps the POSIX text git wrote, so the next run still finds it."""
    root = tmp_path_factory.mktemp("nested")
    relative = "/".join(["d"] * depth) + "/link"
    target = "../" * depth + ".agents"
    flatten(root, {relative: target})
    checks = agents(root, root / "home", junction=lambda link, into: "Access is denied.")

    first, second = checks.links(), checks.links()

    assert first == second
    assert first.detail == f"cannot stand in for {relative} (Access is denied.)"
    assert (root / relative).read_text(encoding="utf-8") == target


@pytest.mark.parametrize(
    ("files", "verdict", "detail"),
    [
        ({}, Verdict.FAIL, "missing AGENTS.md, CLAUDE.md"),
        (
            {"AGENTS.md": "", "CLAUDE.md": "rules\n", ".claude/notes": ""},
            Verdict.WARN,
            "CLAUDE.md does not include @AGENTS.md, so Claude Code never reads it; "
            ".claude does not reach .agents/settings.json",
        ),
        (
            {"AGENTS.md": "", "CLAUDE.md": "@AGENTS.md\n", ".claude/settings.json": "{}"},
            Verdict.PASS,
            "AGENTS.md, CLAUDE.md and .claude -> .agents in place",
        ),
    ],
    ids=["missing", "unwired", "wired"],
)
def test_the_instructions_every_agent_reads_are_there_and_wired_together(
    tmp_path: Path, home: Path, files: dict[str, str], verdict: Verdict, detail: str
) -> None:
    """AGENTS.md is the one set of rules, reached by Claude Code only through CLAUDE.md."""
    row = agents(written(tmp_path, files), home).instructions()

    assert (row.verdict, row.detail) == (verdict, detail)


def test_each_server_is_judged_by_what_would_stop_it_starting_on_this_machine(
    tmp_path: Path, home: Path
) -> None:
    """A command off PATH, an undefined variable and a path from another machine are each named.

    Entries that are not tables are notes, not servers. A command is a string or a list whose
    first word is the program. A missing file declares nothing, and one that does not parse
    fails, since the agent reading it would fail the same way.
    """
    mcp = {
        "mcpServers": {
            "fine": {"command": ["present", "--stdio"], "env": {"HERE": str(tmp_path), "N": 3}},
            "gone": {"command": "absent-tool", "args": ["${WITH_DEFAULT:-x}"]},
            "empty": {"command": []},
            "note": "the docs server moved",
            "paths": {"command": "uvx", "env": {"A": "/not/on/this/machine", "B": "C:\\gone"}},
            "relative": {"command": "present", "env": "not a table"},
        }
    }
    opencode = {"mcp": {"search": {"command": ["present"], "environment": {"K": "{env:NOPE}"}}}}
    root = written(
        tmp_path,
        {
            ".mcp.json": json.dumps(mcp),
            "opencode.json": json.dumps(opencode),
            ".codex/config.toml": "[mcp_servers\n",
        },
    )

    claude, opencode_row, codex = agents(root, home, environment={"WITH_DEFAULT": "1"}).servers()

    assert claude.verdict == Verdict.WARN
    assert claude.detail == (
        "gone: absent-tool is not on PATH; "
        "paths: /not/on/this/machine does not exist on this machine; "
        "paths: C:\\gone does not exist on this machine"
    )
    assert (opencode_row.verdict, opencode_row.detail) == (
        Verdict.WARN,
        "search: NOPE is undefined",
    )
    assert codex.verdict == Verdict.FAIL
    assert codex.detail.startswith("does not parse: ")
    empty = agents(tmp_path / "nowhere", home).servers()
    assert {(row.verdict, row.detail) for row in empty} == {(Verdict.PASS, "not declared")}


@given(
    placed=st.dictionaries(
        _NAMES, st.sampled_from(["environment", "dotenv", "both", "nowhere"]), max_size=5
    ),
    launched=st.booleans(),
)
def test_a_variable_counts_as_defined_only_where_the_server_s_launcher_will_find_it(
    tmp_path: Path, home: Path, placed: dict[str, str], launched: bool
) -> None:
    """The agent's own environment always counts; the workspace `.env` only under `mainboard`.

    `mainboard run` loads `.env` itself, so a server it launches sees those variables, while an
    agent starting any other command never does and the variable reads as defined nowhere
    useful. Every problem is named once, in name order.
    """
    command = "mainboard" if launched else "present"
    entry = {"command": command, "env": {name: f"${{{name}}}" for name in placed}}
    written(tmp_path, {".mcp.json": json.dumps({"mcpServers": {"s": entry}})})
    environment = {name: "1" for name, where in placed.items() if where in ("environment", "both")}
    env_file = {name: "1" for name, where in placed.items() if where in ("dotenv", "both")}

    row, _, _ = agents(tmp_path, home, environment=environment, env_file=env_file).servers()

    expected = [
        f"s: {name} is only in .env, which this server's launcher does not load"
        if where == "dotenv"
        else f"s: {name} is undefined"
        for name, where in sorted(placed.items())
        if where == "nowhere" or (where == "dotenv" and not launched)
    ]
    assert row.detail == ("; ".join(expected) if expected else "1 servers can start here")
    assert row.verdict == (Verdict.WARN if expected else Verdict.PASS)


@pytest.mark.parametrize("remembered", [True, False])
def test_claude_memory_is_found_under_this_workspace_s_own_key(
    tmp_path: Path, home: Path, remembered: bool
) -> None:
    """Claude Code files memory under the workspace path, so a moved workspace starts empty."""
    key = claude_key(str(tmp_path))
    if remembered:
        written(home / ".claude" / "projects" / key / "memory", {"a.md": "", "b.md": ""})
        (home / ".claude" / "projects" / key / "memory" / "old").mkdir()

    row = agents(tmp_path, home).memory()

    assert row.verdict == (Verdict.PASS if remembered else Verdict.WARN)
    assert row.detail == (
        f"2 memory files under {key}" if remembered else f"no Claude Code memory under {key}"
    )


@pytest.mark.parametrize(
    ("held", "verdict", "fix"),
    [
        ((".codex/auth.json", ".local/share/opencode/auth.json"), Verdict.PASS, ""),
        ((".codex/auth.json",), Verdict.WARN, "opencode auth login"),
        ((), Verdict.WARN, "codex login; opencode auth login"),
    ],
    ids=["both", "codex", "neither"],
)
def test_each_agent_login_is_found_by_the_file_it_keeps_it_in(
    tmp_path: Path, home: Path, held: Sequence[str], verdict: Verdict, fix: str
) -> None:
    """A login missing on a new center is named with the command that makes it."""
    written(home, dict.fromkeys(held, "{}"))

    row = agents(tmp_path, home).logins()

    assert (row.verdict, row.fix) == (verdict, fix)


@given(
    defined=st.dictionaries(_NAMES, _VALUES, max_size=6),
    exported=st.lists(st.booleans(), min_size=6, max_size=6),
)
def test_dotenv_reads_every_assignment_and_skips_comments_and_blank_lines(
    tmp_path: Path, defined: dict[str, str], exported: list[bool]
) -> None:
    """Every `NAME=value` line is read, indented or `export` or not, and no comment is a name."""
    lines = ["# NAME=commented", "", "   # INDENTED=commented", "\t#TABBED=1"]
    for (name, value), export in zip(defined.items(), exported, strict=False):
        lines += [f"  {'export ' if export else ''}{name}={value}", "#OTHER=1", "no sign here"]
    path = tmp_path / ".env"
    path.write_text("\n".join(lines), encoding="utf-8")

    assert dotenv(path) == defined
    assert dotenv(tmp_path / "absent.env") == {}


@pytest.mark.parametrize(("status", "said"), [(0, ""), (1, "Access is denied.")])
def test_a_junction_is_made_by_mklink_and_its_refusal_is_its_own_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int, said: str
) -> None:
    """The one real junction maker runs `mklink /J` through cmd and answers why not, or nothing."""
    ran: list[list[str]] = []

    def run(argv: list[str], **options: bool) -> subprocess.CompletedProcess[str]:
        ran.append(argv)
        return subprocess.CompletedProcess(argv, status, "", f"{said}\r\n" if said else "")

    monkeypatch.setattr(module.subprocess, "run", run)

    assert junction(tmp_path / "link", tmp_path / "target") == said
    assert ran == [
        ["cmd", "/d", "/c", "mklink", "/J", str(tmp_path / "link"), str(tmp_path / "target")]
    ]
