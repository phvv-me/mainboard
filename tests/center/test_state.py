import hashlib
import io
import json
import re
import sqlite3
import string
import tarfile
import tomllib
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.center.state import (
    Carried,
    Destination,
    Parcel,
    SshConfig,
    claude_key,
    packed,
    snapshot,
)
from mainboard.core.section import Verdict
from mainboard.manifest.held import Held, Holdings
from mainboard.manifest.loading import load
from mainboard.manifest.schema.host import HostProfile

from ..strategies import WORDS

# The workspace the center runs: three declared hosts and a second environment.
_MANIFEST = """[workspace]
name = "lab"

[hosts.gold]
kind = "ssh"

[hosts.tunnel]
kind = "ssh"

[hosts.direct]
kind = "ssh"

[envs.serving]
"""

# An ssh config with every shape the cut must handle: a leading block, a wildcard block with
# settings some clients refuse, a bastion chain by ProxyJump and by ProxyCommand, an opt-out
# jump, a Match block, an unrelated host, an identity outside home and an empty IdentityFile.
_SSH_CONFIG = """# my machines
Include extra.conf

Host *
    ServerAliveInterval 30
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
    UseKeychain yes

Host bastion
    HostName bastion.example.com
    IdentityFile ~/.ssh/id_bastion

Host gold
    HostName gold.lab
    ProxyJump me@bastion:2222
    IdentityFile ~/.ssh/id_gold

Host tunnel
    ProxyCommand ssh relay -W %h:%p
    IdentityFile "/outside/home/key"

Host relay
    HostName relay.example.com
    IdentityFile ~/.ssh/id_relay

Host unrelated
    HostName unrelated.example.com
    IdentityFile ~/.ssh/id_unrelated

Host direct
    ProxyJump none

Host rented
    HostName ssh5.vast.ai

Match host *.corp exec "true"
    User corp

Host github.com
    IdentityFile ~/.ssh/id_github
    IdentityFile
"""

# Where the new center lives, spelled the way a Windows destination spells it.
_DESTINATION = Destination(root=r"C:\Users\me\projects", home=r"C:\Users\me", system="Windows")


def database(path: Path, *rows: str) -> Path:
    """A real SQLite file at `path` holding `rows` in one table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE IF NOT EXISTS notes (text TEXT)")
        connection.executemany("INSERT INTO notes VALUES (?)", [(row,) for row in rows])
    return path


def notes(data: bytes, scratch: Path) -> list[str]:
    """The rows of a snapshot's `notes` table, read back through SQLite itself."""
    copy = scratch / "read-back.sqlite"
    copy.write_bytes(data)
    with closing(sqlite3.connect(copy)) as connection:
        return [row for (row,) in connection.execute("SELECT text FROM notes ORDER BY rowid")]


def unpacked(stream: bytes) -> dict[str, tuple[int, int, bytes]]:
    """Each member of a tar stream by name, as its mode, its stamp and its bytes."""
    members: dict[str, tuple[int, int, bytes]] = {}
    with tarfile.open(fileobj=io.BytesIO(stream), mode="r|") as archive:
        for member in archive:
            content = archive.extractfile(member)
            assert content is not None
            members[member.name] = (member.mode, int(member.mtime), content.read())
    return members


def write(path: Path, text: str = "x") -> Path:
    """Write `text` at `path`, making its folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """The center's workspace: its manifest, a `.env`, and `.mainboard/` state kept and local."""
    root = tmp_path / "projects"
    write(root / "mainboard.toml", _MANIFEST)
    write(root / ".env", "VAST_API_KEY=secret\n")
    for kept in ("batches/x/receipts.ndjson", "costs/costs.ndjson"):
        write(root / ".mainboard" / kept)
    for local in (
        "envs/default/prefix/bin/python",
        "pins/abc/model.safetensors",
        "activate.sh",
        "activate-serving.sh",
        "collection.digests.json",
        "dispatch/queue.lock",
        "dispatch/locks/gold",
        "dispatch/db.sqlite-wal",
        "dispatch/db.sqlite-shm",
    ):
        write(root / ".mainboard" / local)
    for artifact in ("pixi.toml", "pixi.lock", "state.toml"):
        write(root / ".mainboard" / "envs" / "default" / artifact)
    database(root / ".mainboard" / "dispatch" / "db.sqlite", "job 1")
    return root


@pytest.fixture
def carried(root: Path, home: Path) -> Carried:
    """What this center would carry to `_DESTINATION`."""
    return Carried(root, load(root / "mainboard.toml"), _DESTINATION, home=home)


@pytest.fixture
def ssh(home: Path, root: Path) -> Path:
    """A home `.ssh/` holding the config above, its keys, and one rented machine held."""
    folder = home / ".ssh"
    write(folder / "config", _SSH_CONFIG)
    for key in ("id_bastion", "id_gold", "id_relay", "id_unrelated", "id_github", "id_ed25519"):
        write(folder / key, "PRIVATE")
    for public in ("id_bastion", "id_relay", "id_unrelated", "id_ed25519"):
        write(folder / f"{public}.pub", "PUBLIC")
    write(folder / "known_hosts", "gold ssh-ed25519 AAAA\n")
    Holdings(root).save(
        Held(
            alias="rented",
            provider="vast",
            handle="7",
            deadline=datetime(2030, 1, 1, tzinfo=UTC),
            profile=HostProfile(kind="ssh"),
        )
    )
    return folder


@pytest.fixture
def agents(home: Path, root: Path) -> Path:
    """Claude Code, Codex and opencode state under home, with what must stay beside what moves."""
    claude = home / ".claude"
    for name in ("settings.json", "CLAUDE.md", "agents/reviewer.md", "history.jsonl"):
        write(claude / name)
    write(claude / "projects" / claude_key(str(root)) / "memory" / "MEMORY.md")
    write(claude / "projects" / claude_key(str(root)) / "memory" / "notes" / "day.md")
    write(claude / "projects" / "-elsewhere" / "memory" / "MEMORY.md")
    for name in ("installed_plugins.json", "known_marketplaces.json", "cache/x/plugin.json"):
        write(claude / "plugins" / name)
    write(claude / ".credentials.json", "{}")
    codex = home / ".codex"
    # Codex keys a project by its path in a literal string, which a Windows backslash survives.
    write(
        codex / "config.toml", f'model = "o3"\n\n[projects.\'{root}\']\ntrust_level = "trusted"\n'
    )
    for name in ("auth.json", ".credentials.json", "rules/default.rules", "sessions/a.jsonl"):
        write(codex / name)
    database(codex / "memories_1.sqlite", "remember this")
    write(home / ".config" / "opencode" / "opencode.json")
    write(home / ".config" / "opencode" / "node_modules" / "pkg" / "index.js")
    write(home / ".local" / "share" / "opencode" / "auth.json")
    return home


@given(st.text(max_size=40))
def test_claude_key_keeps_ascii_alphanumerics_and_dashes_everything_else(path: str) -> None:
    """Claude Code files a workspace under this name, so it must match it char for char."""
    keyed = claude_key(path)
    assert len(keyed) == len(path)
    assert re.fullmatch(r"[A-Za-z0-9-]*", keyed)
    alphanumeric = set(string.ascii_letters + string.digits)
    assert all(
        new == (old if old in alphanumeric else "-") for old, new in zip(path, keyed, strict=True)
    )


@given(
    st.dictionaries(WORDS, st.tuples(st.binary(max_size=32), st.booleans()), max_size=6),
    st.sampled_from(["root", "home"]),
)
def test_packed_streams_one_member_per_parcel_named_by_key_with_secrets_private(
    tmp_path: Path, contents: dict[str, tuple[bytes, bool]], anchor: str
) -> None:
    """The tar a destination reads back holds each parcel's bytes under its key, nothing else.

    It streams a member at a time, credentials travel with the owner-only mode, and a file
    parcel keeps its stamp so the destination's mtime fingerprint matches after placement.
    """
    source = write(tmp_path / "source.txt", "from disk")
    parcels = [
        Parcel.model_validate(
            {"anchor": anchor, "path": f"d/{name}", "data": data, "secret": secret}
        )
        for name, (data, secret) in contents.items()
    ] + [Parcel(anchor="root", path="file/source.txt", source=source)]
    chunks = list(packed(parcels))
    assert len(chunks) == len(parcels) + 1
    assert unpacked(b"".join(chunks)) == {
        parcel.key: (
            0o600 if parcel.secret else 0o644,
            int(source.stat().st_mtime) if parcel.source else 0,
            parcel.source.read_bytes() if parcel.source else parcel.data,
        )
        for parcel in parcels
    }


def test_a_parcel_fingerprints_made_bytes_by_content_and_a_file_by_its_stamp(
    tmp_path: Path,
) -> None:
    """Made content compares by hash, a file tree by time and size, so gigabytes go unread."""
    made = Parcel(anchor="home", path=".ssh/config", data=b"Host gold\n")
    assert made.listing() == {
        "key": "home/.ssh/config",
        "anchor": "home",
        "path": ".ssh/config",
        "fingerprint": "sha256:" + hashlib.sha256(b"Host gold\n").hexdigest(),
    }
    source = write(tmp_path / "receipts.ndjson", "12345")
    stamped = Parcel(anchor="root", path=".mainboard/receipts.ndjson", source=source)
    assert stamped.fingerprint == f"mtime:{int(source.stat().st_mtime)}:5"


@pytest.mark.parametrize(
    ("system", "control", "keychain"),
    [("Windows", False, False), ("Linux", True, False), ("Darwin", True, True)],
)
def test_the_ssh_cut_carries_each_hosts_jump_chain_and_speaks_the_destinations_client(
    carried: Carried, ssh: Path, system: str, control: bool, keychain: bool
) -> None:
    """Every declared, held and GitHub host arrives with the bastions it goes through.

    Windows OpenSSH refuses the Control settings and only macOS knows UseKeychain, so the
    rendered config drops what the destination's client would reject. Unrelated hosts, Match
    blocks and the leading lines stay behind, and a `none` jump names no host.
    """
    carried.destination = _DESTINATION.model_copy(update={"system": system})
    parcels = carried.ssh()
    config = parcels[0].data.decode()
    hosts = [line.split()[1] for line in config.splitlines() if line.startswith("Host ")]
    assert hosts == ["*", "bastion", "gold", "tunnel", "relay", "direct", "rented", "github.com"]
    assert ("ControlMaster auto" in config, "ControlPersist" in config) == (control, control)
    assert ("UseKeychain" in config) is keychain
    assert "Match" not in config and "Include" not in config and config.endswith("\n")
    assert [(parcel.path, parcel.secret) for parcel in parcels[1:]] == [
        (".ssh/id_bastion", True),
        (".ssh/id_bastion.pub", False),
        (".ssh/id_gold", True),
        (".ssh/id_relay", True),
        (".ssh/id_relay.pub", False),
        (".ssh/id_github", True),
        (".ssh/id_ed25519", True),
        (".ssh/id_ed25519.pub", False),
        (".ssh/known_hosts", False),
    ]


def test_the_ssh_cut_carries_nothing_it_does_not_find(carried: Carried, home: Path) -> None:
    """No config means no ssh parcel at all, and absent known hosts are not invented."""
    assert carried.ssh() == []
    write(home / ".ssh" / "config", "Host gold\n    HostName gold.lab\n")
    assert [parcel.path for parcel in carried.ssh()] == [".ssh/config"]


def test_identities_are_read_as_written_and_empty_ones_skipped() -> None:
    """A quoted path loses its quotes, a bare `IdentityFile` names nothing, jumps are followed."""
    blocks = SshConfig(_SSH_CONFIG).needed(["github.com", "tunnel"])
    assert SshConfig.identities(blocks) == [
        "/outside/home/key",
        "~/.ssh/id_relay",
        "~/.ssh/id_github",
    ]


def test_the_workspace_carries_ledgers_and_locks_and_leaves_machine_local_state(
    carried: Carried, root: Path, tmp_path: Path
) -> None:
    """Receipts, costs and each environment's lock move; prefixes, pins and caches are rebuilt.

    The live dispatch registry travels as a consistent SQLite snapshot rather than as bytes a
    writer may be halfway through, and the `.env` of provider keys travels as a secret.
    """
    parcels = carried.workspace()
    assert [(parcel.path, parcel.secret, parcel.source is None) for parcel in parcels] == [
        (".env", True, False),
        (".mainboard/batches/x/receipts.ndjson", False, False),
        (".mainboard/costs/costs.ndjson", False, False),
        (".mainboard/dispatch/db.sqlite", False, True),
        (".mainboard/envs/default/pixi.toml", False, False),
        (".mainboard/envs/default/pixi.lock", False, False),
        (".mainboard/envs/default/state.toml", False, False),
    ]
    assert notes(parcels[3].data, tmp_path) == ["job 1"]
    (root / ".env").unlink()
    (root / ".mainboard" / "dispatch" / "db.sqlite").unlink()
    assert [parcel.path for parcel in carried.workspace()][:1] == [
        ".mainboard/batches/x/receipts.ndjson"
    ]
    assert ".mainboard/dispatch/db.sqlite" not in [parcel.path for parcel in carried.workspace()]


def test_agent_state_moves_rekeyed_to_the_new_workspace_and_history_stays(
    carried: Carried, agents: Path, root: Path, tmp_path: Path
) -> None:
    """Memory lands under the key Claude Code derives from the new root, Codex trusts the new path.

    Settings, rules and logins move, credentials as secrets; history, sessions, plugin caches,
    other workspaces' memory and an installed node_modules tree stay behind.
    """
    parcels = {parcel.path: parcel for parcel in carried.agents()}
    memory = f".claude/projects/{claude_key(_DESTINATION.root)}/memory"
    assert {path: parcel.secret for path, parcel in parcels.items()} == {
        f"{memory}/MEMORY.md": False,
        f"{memory}/notes/day.md": False,
        ".claude/settings.json": False,
        ".claude/CLAUDE.md": False,
        ".claude/agents/reviewer.md": False,
        ".claude/plugins/installed_plugins.json": False,
        ".claude/plugins/known_marketplaces.json": False,
        ".claude/.credentials.json": True,
        ".codex/rules/default.rules": False,
        ".codex/auth.json": True,
        ".codex/.credentials.json": True,
        ".config/opencode/opencode.json": False,
        ".local/share/opencode/auth.json": True,
        ".codex/config.toml": True,
        ".codex/memories_1.sqlite": False,
    }
    codex = tomllib.loads(parcels[".codex/config.toml"].data.decode())
    assert codex == {"model": "o3", "projects": {_DESTINATION.root: {"trust_level": "trusted"}}}
    assert notes(parcels[".codex/memories_1.sqlite"].data, tmp_path) == ["remember this"]


@pytest.mark.parametrize(
    "config",
    ['model = "o3"\n', '[projects."/elsewhere"]\ntrust_level = "trusted"\n', None],
)
def test_a_codex_config_not_naming_this_root_moves_unchanged_and_an_absent_one_not_at_all(
    carried: Carried, home: Path, config: str | None
) -> None:
    """Only this workspace's trust entry is re-keyed; a missing config or memory is no parcel."""
    if config is not None:
        write(home / ".codex" / "config.toml", config)
    made = {parcel.path: parcel.data.decode() for parcel in carried.agents()}
    assert made == ({".codex/config.toml": config} if config is not None else {})


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (None, {}),
        ('{"theme": "dark"}', {}),
        ('{"projects": {ROOT: {"history": ["hi"], "lastCost": 1.5}}}', {}),
        (
            '{"projects": {ROOT: {"allowedTools": ["Bash(git status)"], "history": ["hi"],'
            ' "hasTrustDialogAccepted": true}, "/elsewhere": {"hasTrustDialogAccepted": true}}}',
            {
                _DESTINATION.root: {
                    "allowedTools": ["Bash(git status)"],
                    "hasTrustDialogAccepted": True,
                }
            },
        ),
    ],
)
def test_the_claude_project_record_keeps_settings_under_the_new_path_and_drops_history(
    carried: Carried,
    home: Path,
    root: Path,
    document: str | None,
    expected: dict[str, dict[str, list[str] | bool]],
) -> None:
    """Trust and allowed tools let the new center open the workspace as this one did."""
    if document is not None:
        write(home / ".claude.json", document.replace("ROOT", json.dumps(str(root))))
    assert carried.claude_project() == expected


@pytest.mark.parametrize("credentials", [True, False])
def test_what_stays_behind_is_named_with_how_the_new_center_gets_it(
    carried: Carried, home: Path, credentials: bool
) -> None:
    """A Claude login kept in the keychain cannot move, so the report says to sign in again."""
    if credentials:
        write(home / ".claude" / ".credentials.json", "{}")
    left = carried.left()
    login = [row for row in left if row.section == "left: Claude Code login"]
    assert len(login) == (0 if credentials else 1)
    assert all(row.verdict in (Verdict.PASS, Verdict.WARN) for row in left)
    assert all(row.fix for row in left if row.verdict is Verdict.WARN)


def test_a_snapshot_holds_what_was_committed_and_not_what_a_writer_has_open(
    tmp_path: Path,
) -> None:
    """A registry being written while the migration runs arrives consistent, never half-written."""
    live = database(tmp_path / "db.sqlite", "settled")
    with closing(sqlite3.connect(live)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO notes VALUES ('in flight')")
        assert notes(snapshot(live), tmp_path) == ["settled"]
