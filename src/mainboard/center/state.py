# Everything the center holds that git does not, and that its successor needs on day one.
#
# A clone brings the tracked tree and nothing else. The center also runs on a `.env` of provider
# keys, on the `.mainboard/` registry of every job it dispatched and every ledger it settled, on
# the ssh keys and aliases the host profiles name, on a GitHub login, and on each AI agent's
# per-user state: Claude Code's memory for this workspace, Codex's config, credentials and
# memories, opencode's config and logins. This module finds each of those here and says where
# it lands there, re-keying what a tool files under the workspace's own path, and names what it
# deliberately leaves behind. It reads; shipping is the migration's business.

import hashlib
import io
import json
import re
import sqlite3
import tarfile
from collections.abc import Iterator, Sequence
from contextlib import closing
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

import tomlkit
from patos import FrozenModel

from ..core.project import Project
from ..core.section import Section, Verdict
from ..engines.compile.provisioner import Provisioner
from ..manifest.held import Holdings

if TYPE_CHECKING:
    from ..manifest.schema.root import Manifest
    from ..probe.census import Json

type Anchor = Literal["root", "home"]

# What `.mainboard/` keeps that belongs to this machine alone or is rebuilt from what does ship:
# environment prefixes (reinstalled from the shipped lock), the hub pin cache (refetched at its
# recorded revisions), generated activation scripts, a digest cache and every lock file.
_MACHINE_LOCAL = (
    "envs/*",
    "pins/*",
    "activate*.sh",
    "collection.digests.json",
    "*.lock",
    "*.sqlite-wal",
    "*.sqlite-shm",
    "dispatch/locks/*",
)

# The SQLite stores shipped as a consistent snapshot rather than as the bytes a writer may be
# halfway through: the dispatch registry here, Codex's memories under home.
_REGISTRY = "dispatch/db.sqlite"

# Each agent's per-user state under home: what is copied as is, and what is a credential.
_CLAUDE = ("settings.json", "CLAUDE.md", "keybindings.json", "agents", "skills", "commands")
_CLAUDE_PLUGINS = ("plugins/installed_plugins.json", "plugins/known_marketplaces.json")
_CLAUDE_SECRETS = (".credentials.json",)
_CODEX = ("AGENTS.md", "rules", "skills", "prompts")
_CODEX_SECRETS = ("auth.json", ".credentials.json")
_CODEX_MEMORIES = "memories_1.sqlite"
_OPENCODE_CONFIG = ".config/opencode"
_OPENCODE_SECRETS = (".local/share/opencode/auth.json", ".local/share/opencode/mcp-auth.json")

# What of Claude Code's per-project record in `~/.claude.json` is a setting rather than history.
_CLAUDE_PROJECT = (
    "allowedTools",
    "mcpContextUris",
    "enabledMcpjsonServers",
    "disabledMcpjsonServers",
    "hasTrustDialogAccepted",
    "hasClaudeMdExternalIncludesApproved",
    "hasClaudeMdExternalIncludesWarningShown",
)

# ssh settings the Windows OpenSSH client refuses outright, and the macOS-only one other clients
# refuse unless told to ignore it.
_WINDOWS_REFUSES = ("controlmaster", "controlpath", "controlpersist")
_MACOS_ONLY = ("usekeychain",)

# The keys ssh offers when a host names none, carried when present.
_DEFAULT_KEYS = ("id_ed25519", "id_ecdsa", "id_rsa")

# The host GitHub is reached at over ssh, which the workspace's own submodules name.
_GITHUB = "github.com"


def claude_key(path: str) -> str:
    """The directory name Claude Code files a workspace's state under: every other char a dash.

    path: the workspace root as that machine spells it, `C:\\Users\\me\\projects` say.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", path)


class Destination(FrozenModel):
    """Where the new center keeps things, as the destination agent reported it.

    root: the workspace root, absolute and in that machine's spelling.
    home: the user's home directory.
    separator: that machine's path separator.
    system: the platform as `platform.system()` spells it there.
    """

    root: str
    home: str
    separator: str = "/"
    system: str = ""


class Parcel(FrozenModel):
    """One file to put on the destination, from a local file or from bytes made here.

    anchor: whether `path` is relative to the workspace root or to home.
    path: the relative path, forward slashes.
    source: the local file, None when `data` carries the content.
    data: the content made here, a re-keyed config or a database snapshot.
    secret: whether the file is a credential, written readable by its owner alone.
    """

    anchor: Anchor
    path: str
    source: Path | None = None
    data: bytes = b""
    secret: bool = False

    @property
    def key(self) -> str:
        """The member name this parcel travels under, `<anchor>/<path>`."""
        return f"{self.anchor}/{self.path}"

    @property
    def fingerprint(self) -> str:
        """What the destination compares its own copy against: content, or a tree's stamp."""
        if self.source is None:
            return "sha256:" + hashlib.sha256(self.data).hexdigest()
        status = self.source.stat()
        return f"mtime:{int(status.st_mtime)}:{status.st_size}"

    def listing(self) -> dict[str, str]:
        """This parcel as the destination's inventory reads it."""
        return {
            "key": self.key,
            "anchor": self.anchor,
            "path": self.path,
            "fingerprint": self.fingerprint,
        }


def packed(parcels: Sequence[Parcel]) -> Iterator[bytes]:
    """`parcels` as one uncompressed tar stream, yielded a member at a time.

    Streamed so a tree of receipts never has to fit in memory, and uncompressed because most of
    it is already compressed evidence and the rest is small.
    """
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w|") as archive:
        for parcel in parcels:
            content = parcel.source.read_bytes() if parcel.source is not None else parcel.data
            member = tarfile.TarInfo(parcel.key)
            member.size = len(content)
            member.mtime = int(parcel.source.stat().st_mtime) if parcel.source is not None else 0
            member.mode = 0o600 if parcel.secret else 0o644
            archive.addfile(member, io.BytesIO(content))
            yield sink.getvalue()
            sink.seek(0)
            sink.truncate()
    yield sink.getvalue()


class SshConfig:
    """The user's ssh client config, cut to the blocks this workspace's hosts need.

    text: the config file's content.
    """

    def __init__(self, text: str) -> None:
        self.blocks = self._blocks(text)

    def needed(self, aliases: Sequence[str]) -> list[list[str]]:
        """The blocks reaching any of `aliases`, with every jump host they go through.

        A block is kept when one of its patterns matches an alias, and the aliases its
        `ProxyJump` or `ProxyCommand ssh <alias>` go through are followed until nothing new is
        named, so a host behind a bastion arrives with its bastion.
        """
        wanted = set(aliases)
        kept: list[list[str]] = []
        grew = True
        while grew:
            grew = False
            for block in self.blocks:
                if block in kept or not any(
                    fnmatch(alias, pattern) for alias in wanted for pattern in _patterns(block)
                ):
                    continue
                kept.append(block)
                hops = _hops(block) - wanted
                wanted |= hops
                grew = True
        return [block for block in self.blocks if block in kept]

    @staticmethod
    def identities(blocks: Sequence[list[str]]) -> list[str]:
        """Every `IdentityFile` the blocks name, as written."""
        return [
            line.split(None, 1)[1].strip().strip('"')
            for block in blocks
            for line in block
            if line.strip().lower().startswith("identityfile") and len(line.split(None, 1)) > 1
        ]

    @staticmethod
    def rendered(blocks: Sequence[list[str]], *, system: str) -> str:
        """The blocks as a config the destination's ssh client accepts."""
        refused = (_WINDOWS_REFUSES if system == "Windows" else ()) + (
            _MACOS_ONLY if system != "Darwin" else ()
        )
        lines = [
            line
            for block in blocks
            for line in block
            if not line.strip().lower().startswith(refused)
        ]
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _blocks(text: str) -> list[list[str]]:
        """The config split at each `Host` or `Match` line, leading lines their own block."""
        blocks: list[list[str]] = [[]]
        for line in text.splitlines():
            if line.strip().lower().startswith(("host ", "match ")):
                blocks.append([])
            blocks[-1].append(line)
        return [block for block in blocks if any(line.strip() for line in block)]


def _patterns(block: Sequence[str]) -> list[str]:
    """The host patterns a `Host` block answers to, none for a `Match` or leading block."""
    head = block[0].split()
    return head[1:] if head and head[0].lower() == "host" else []


def _hops(block: Sequence[str]) -> set[str]:
    """The aliases a block's jump settings go through."""
    hops: set[str] = set()
    for line in block:
        key, _, value = line.strip().partition(" ")
        if key.lower() == "proxyjump":
            hops |= {hop.split("@")[-1].split(":")[0] for hop in value.split(",") if hop.strip()}
        elif key.lower() == "proxycommand":
            words = value.split()
            hops |= {words[index + 1] for index, word in enumerate(words[:-1]) if word == "ssh"}
    return {hop.strip() for hop in hops if hop.strip() and hop.strip().lower() != "none"}


class Carried:
    """The state this center carries to a destination, and what it leaves behind on purpose.

    root: this workspace's root.
    manifest: its loaded manifest, which names the environments and the hosts.
    destination: where the new center keeps its root and home.
    home: this machine's home directory.
    """

    def __init__(
        self, root: Path, manifest: Manifest, destination: Destination, *, home: Path
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.destination = destination
        self.home = home

    def ssh(self) -> list[Parcel]:
        """The ssh config blocks the declared hosts need, the keys they name, and known hosts."""
        folder = self.home / ".ssh"
        try:
            config = SshConfig((folder / "config").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        aliases = [
            *self.manifest.profiles(),
            *Holdings(self.root).read(),
            _GITHUB,
        ]
        blocks = config.needed(aliases)
        named = [self._home_path(identity) for identity in config.identities(blocks)] + [
            folder / key for key in _DEFAULT_KEYS
        ]
        keys = [path for path in dict.fromkeys(named) if path is not None and path.is_file()]
        parcels = [
            Parcel(
                anchor="home",
                path=".ssh/config",
                data=SshConfig.rendered(blocks, system=self.destination.system).encode(),
            ),
            *(
                parcel
                for key in keys
                for parcel in (
                    self._home(key, secret=True),
                    *(
                        [self._home(key.with_name(key.name + ".pub"))]
                        if key.with_name(key.name + ".pub").is_file()
                        else []
                    ),
                )
            ),
        ]
        if (folder / "known_hosts").is_file():
            parcels.append(self._home(folder / "known_hosts"))
        return parcels

    def workspace(self) -> list[Parcel]:
        """The `.env`, the `.mainboard/` registry and ledgers, and every environment's lock."""
        out = self.root / Project().out_dir
        parcels = (
            [self._root(self.root / ".env", secret=True)] if (self.root / ".env").is_file() else []
        )
        parcels += [
            self._root(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(out).as_posix() != _REGISTRY
            and not any(fnmatch(path.relative_to(out).as_posix(), rule) for rule in _MACHINE_LOCAL)
        ]
        registry = out / _REGISTRY
        if registry.is_file():
            parcels.append(
                Parcel(
                    anchor="root",
                    path=registry.relative_to(self.root).as_posix(),
                    data=snapshot(registry),
                )
            )
        provisioner = Provisioner(self.root, self.manifest)
        artifacts = dict.fromkeys(
            relative
            for environment in ("default", *self.manifest.envs)
            for relative in provisioner.artifact_for(environment)
        )
        parcels += [
            self._root(self.root / relative)
            for relative in artifacts
            if (self.root / relative).is_file()
        ]
        return parcels

    def agents(self) -> list[Parcel]:
        """Claude Code, Codex and opencode per-user state, re-keyed to the new workspace path."""
        claude = self.home / ".claude"
        codex = self.home / ".codex"
        memory = claude / "projects" / claude_key(str(self.root)) / "memory"
        rekeyed = f".claude/projects/{claude_key(self.destination.root)}/memory"
        parcels = [
            *self._tree(memory, into=rekeyed),
            *(parcel for name in _CLAUDE for parcel in self._tree(claude / name)),
            *(parcel for name in _CLAUDE_PLUGINS for parcel in self._tree(claude / name)),
            *(
                parcel
                for name in _CLAUDE_SECRETS
                for parcel in self._tree(claude / name, secret=True)
            ),
            *(parcel for name in _CODEX for parcel in self._tree(codex / name)),
            *(
                parcel
                for name in _CODEX_SECRETS
                for parcel in self._tree(codex / name, secret=True)
            ),
            *self._tree(self.home / _OPENCODE_CONFIG),
            *(
                parcel
                for name in _OPENCODE_SECRETS
                for parcel in self._tree(self.home / name, secret=True)
            ),
        ]
        if (codex / "config.toml").is_file():
            parcels.append(
                Parcel(
                    anchor="home",
                    path=".codex/config.toml",
                    data=self._codex_config(codex / "config.toml").encode(),
                    secret=True,
                )
            )
        if (codex / _CODEX_MEMORIES).is_file():
            parcels.append(
                Parcel(
                    anchor="home",
                    path=f".codex/{_CODEX_MEMORIES}",
                    data=snapshot(codex / _CODEX_MEMORIES),
                )
            )
        return parcels

    def claude_project(self) -> dict[str, Json]:
        """This workspace's settings from Claude Code's `~/.claude.json`, under the new path.

        Trust, the allowed tools and which project MCP servers are on, re-keyed so the new
        center opens the workspace as this one did. Usage history stays behind.
        """
        try:
            document = json.loads((self.home / ".claude.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        record = document.get("projects", {}).get(str(self.root), {})
        kept = {key: record[key] for key in _CLAUDE_PROJECT if key in record}
        return {self.destination.root: kept} if kept else {}

    def left(self) -> list[Section]:
        """What the migration deliberately leaves behind, each with how the new center gets it."""
        notes = [
            Section(
                section="left: env prefixes",
                verdict=Verdict.PASS,
                detail="environment prefixes are rebuilt from the shipped lock, never copied",
            ),
            Section(
                section="left: pins",
                verdict=Verdict.PASS,
                detail="the hub pin cache refetches at each recorded revision on first use",
            ),
            Section(
                section="left: ignored data",
                verdict=Verdict.PASS,
                detail="ignored data (/data/, outputs/) is not workspace state; copy it by hand",
            ),
            Section(
                section="left: ChatGPT desktop",
                verdict=Verdict.WARN,
                detail="the desktop apps keep their sign-in in the app; Codex shares ~/.codex",
                fix="sign in to ChatGPT desktop on the destination",
            ),
        ]
        if not (self.home / ".claude" / ".credentials.json").is_file():
            notes.append(
                Section(
                    section="left: Claude Code login",
                    verdict=Verdict.WARN,
                    detail="Claude Code keeps this login in the system keychain, which stays here",
                    fix="run `claude` on the destination and sign in with /login",
                )
            )
        return notes

    def _codex_config(self, path: Path) -> str:
        """Codex's config with its trusted projects re-keyed to the new workspace path."""
        document = tomlkit.parse(path.read_text(encoding="utf-8"))
        projects = document.get("projects")
        if projects is not None and str(self.root) in projects:
            projects[self.destination.root] = projects.pop(str(self.root))
        return tomlkit.dumps(document)

    def _tree(self, path: Path, *, into: str = "", secret: bool = False) -> list[Parcel]:
        """Every file at or under `path` in home, landing under `into` when re-keyed."""
        files = (
            [path]
            if path.is_file()
            else sorted(item for item in path.rglob("*") if item.is_file())
        )
        base = into or path.relative_to(self.home).as_posix()
        return [
            Parcel(
                anchor="home",
                path=str(PurePosixPath(base, file.relative_to(path).as_posix()))
                if file != path
                else base,
                source=file,
                secret=secret,
            )
            for file in files
            if "node_modules" not in file.parts
        ]

    def _home(self, path: Path, *, secret: bool = False) -> Parcel:
        """One file under home as a parcel."""
        return Parcel(
            anchor="home", path=path.relative_to(self.home).as_posix(), source=path, secret=secret
        )

    def _root(self, path: Path, *, secret: bool = False) -> Parcel:
        """One file under the workspace as a parcel."""
        return Parcel(
            anchor="root", path=path.relative_to(self.root).as_posix(), source=path, secret=secret
        )

    def _home_path(self, written: str) -> Path | None:
        """An `IdentityFile` as written, resolved here, None when it lies outside home."""
        path = Path(
            written.replace("~", str(self.home), 1) if written.startswith("~") else written
        )
        return path if path.is_relative_to(self.home) else None


def snapshot(database: Path) -> bytes:
    """A consistent copy of a live SQLite database, taken through SQLite's own backup."""
    with TemporaryDirectory() as scratch:
        copy = Path(scratch) / database.name
        with (
            closing(sqlite3.connect(database)) as source,
            closing(sqlite3.connect(copy)) as target,
        ):
            source.backup(target)
        return copy.read_bytes()
