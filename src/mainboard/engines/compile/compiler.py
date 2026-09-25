import hashlib
import json
import tomllib
from typing import TYPE_CHECKING

from ...core import MissionError, Project
from .generated import GeneratedFiles
from .pixi_manifest import PixiManifest, cleared, rerooted, selected_manifest
from .state import SyncState
from .vendor import path_deps, relocated

if TYPE_CHECKING:
    from pathlib import Path

    from ...manifest import Manifest
    from .backend import Pixi
    from .ecosystems import SecondStage
    from .generated import Writer
    from .toml import Toml
    from .vendor import Vendor

_DOTENV_SH_FILE = "dotenv.sh"
_DOTENV_BAT_FILE = "dotenv.bat"
_UNSET_SH_FILE = "unset.sh"
_UNSET_BAT_FILE = "unset.bat"


class Compiler:
    """Turns a manifest into the generated `.mainboard/` env, and says when that env is stale.

    It provisions nothing, so the same write runs under `Provisioner.activated`'s short lock and
    inside `provision`'s longer one.
    """

    def __init__(
        self,
        root: Path,
        manifest: Manifest,
        out: Path,
        pixi: Pixi,
        stage: SecondStage,
        vendor: Vendor,
        *,
        environment: str = "default",
    ) -> None:
        self.root = root
        self.environment = environment
        self.manifest = selected_manifest(manifest, environment)
        self.out = out
        self.generated_dir = out.relative_to(root)
        self.pixi = pixi
        self.stage = stage
        self.vendor = vendor

    def digest(self) -> str:
        """The hash of what a compile reads, leaving out tables that only configure a verb."""
        payload = self.manifest.model_dump(
            mode="json", round_trip=True, exclude=set(self.manifest.uncompiled)
        )
        return hashlib.sha256(_canonical(payload)).hexdigest()

    def install_locked(self, files: Writer, *, resolve: bool) -> None:
        """Install this shard, blessing its lock only after a solve returned without raising.

        The blessing records the digest solved from and the pixi that wrote the lock, so a host
        that never solved installs any lock matching what it received, and a host reaching a
        different address can name which two pixis disagreed.
        """
        if not resolve:
            self.vouch()
            self.pixi.install(self.environment)
            return
        # Solved and blessed before any install, so a platform this machine cannot run (a Windows
        # card, from Linux) still leaves a lock to ship.
        self.pixi.solve()
        state = SyncState.load(self.out)
        self.__persist_state(
            files,
            state.model_copy(
                update={
                    "solved_from": self.resolution_digest(),
                    "solved_by": self.pixi.version(),
                }
            ),
        )
        if self.pixi.runs_here():
            self.pixi.install(self.environment)

    def vouch(self) -> None:
        """Refuse unless the lock on disk was solved from this manifest and package metadata.

        A host's own question, asked before the mirror leaves so a stale lock fails in a second.
        Asked under the (reentrant) workspace-root lock the compile takes, since a compile landing
        between a solve and this read made the refusal name the command that had just succeeded.
        """
        with GeneratedFiles(directory=self.root / Project().out_dir).locked():
            state = SyncState.load(self.out)
            if not self.pixi.lock.exists():
                return
            current = self.resolution_digest()
            if state.environment == self.environment and state.solved_from == current:
                return
        raise MissionError(self.__unvouched(state, current))

    def __unvouched(self, state: SyncState, current: str) -> str:
        """Why the lock could not be vouched for, naming both ways out of a digest mismatch.

        A stale lock and a compile landing since the solve look alike from the file on disk.

        current: what the compiled manifest and package metadata hash to now.
        """
        tool = Project().name
        if state.environment and state.environment != self.environment:
            return (
                f"{self.pixi.lock} is blessed for environment {state.environment!r}, not "
                f"{self.environment!r}. Run `{tool} install {self.environment} --resolve`."
            )
        return (
            f"{self.pixi.lock} was not solved from the manifest now compiled at "
            f"{self.pixi.manifest}: that file and the package metadata beside it hash to "
            f"{current[:12]}, while the lock is blessed for {state.solved_from[:12] or 'nothing'}."
            f" Either the lock is stale, in which case `{tool} install {self.environment} "
            "--resolve` on a solve-capable machine settles it, or another process compiled into "
            "this workspace between that solve and now, in which case run it again with nothing "
            "else writing here."
        )

    def resolution_digest(self) -> str:
        """Hash what a solve reads: the compiled manifest on disk and local projects' metadata.

        Tasks and activation cannot move a resolution, and activation would tie the digest to
        the workspace's location, refusing every host rooted elsewhere.
        """
        digest = hashlib.sha256()
        compiled = self._resolution_manifest(self.pixi.manifest.read_text(encoding="utf-8"))
        digest.update(_canonical(compiled))
        for declared in self._local_python_projects():
            project = self.root / declared / "pyproject.toml"
            digest.update(declared.encode())
            try:
                metadata = self._resolution_metadata(project.read_text(encoding="utf-8"))
            except FileNotFoundError:
                digest.update(b"\0")
                continue
            digest.update(_canonical(metadata))
        return digest.hexdigest()

    @staticmethod
    def _resolution_metadata(text: str) -> dict[str, Toml]:
        """The tables of a local project's `pyproject.toml` a solve reads, and nothing else.

        Other `tool` tables cannot move a resolution; hashing them refused every host setup
        after a codespell edit. `project` stays whole since all of it reaches the resolver.
        """
        parsed = tomllib.loads(text)
        tool = parsed.get("tool", {})
        tables: dict[str, Toml] = {
            name: parsed[name]
            for name in ("build-system", "project", "dependency-groups")
            if name in parsed
        }
        resolvers = {name: tool[name] for name in ("pixi", "uv") if name in tool}
        if resolvers:
            tables["tool"] = resolvers
        return tables

    def stale(self) -> bool:
        """Whether an existing compile predates its selected manifest content.

        Nothing compiled yet is not stale: first provisioning is `provision`'s job.
        """
        if not self.pixi.manifest.exists():
            return False
        state = SyncState.load(self.out)
        return (
            state.environment != self.environment
            or state.compiled_from != self.digest()
            or state.runtime_from != self.stage.digest()
        )

    def write(self, files: Writer) -> None:
        """Write this shard's generated files through the workspace-locked writer."""
        # Taken first, so an edit landing mid-compile leaves the workspace stale.
        source_digest = self.digest()
        # Before the manifest naming them, so a solve never reads an unfilled vendored location.
        self.vendor.refresh(files)
        project = Project()
        compiled = PixiManifest.from_manifest(
            self.manifest,
            project_name=project.name,
            environment=self.environment,
            generated_dir=self.generated_dir,
        ).to_toml()
        state = SyncState.load(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._write_generated_files(files, compiled=compiled)
        # Last, so a crash above leaves the workspace stale. `solved_from` is carried through:
        # only a solve says what the lock answered to.
        self.__persist_state(
            files,
            state.model_copy(
                update={
                    "environment": self.environment,
                    "compiled_from": source_digest,
                    "compiled_at": str(self.root),
                    "runtime_from": self.stage.digest(),
                }
            ),
        )

    @staticmethod
    def runtime_manifest(text: str) -> dict[str, Toml]:
        """The generated manifest without tasks, which belong to each source snapshot."""
        return _stripped(tomllib.loads(text), "tasks")

    @staticmethod
    def _resolution_manifest(text: str) -> dict[str, Toml]:
        """The generated manifest without tasks and activation, anywhere they appear.

        Features and targets too: per-environment tasks compile into `[feature.<name>.tasks]`,
        and renaming one used to force a re-solve and invalidate the lock on every host.
        """
        return _stripped(Compiler.runtime_manifest(text), "activation")

    def _local_python_projects(self) -> list[str]:
        """Every local Python project path in this shard's solve, spelled as compiled.

        The vendored spelling, since a declared path leaving the root exists on one machine only
        and hashing it refused every host.
        """
        return sorted({relocated(name, path) for name, path in path_deps(self.manifest).items()})

    def _write_generated_files(self, files: Writer, *, compiled: str) -> None:
        """Write the pixi manifest, the dotenv and unset loaders, and second-stage files."""
        files.write(self.pixi.manifest, compiled)
        if self.manifest.workspace.dotenv:
            workspace = rerooted("", generated_dir=self.generated_dir)
            files.write(self.out / _DOTENV_SH_FILE, _dotenv_sh(workspace))
            files.write(self.out / _DOTENV_BAT_FILE, _dotenv_cmd(workspace))
        taken = cleared(self.manifest.env)
        if taken:
            files.write(self.out / _UNSET_SH_FILE, _unset_sh(taken))
            files.write(self.out / _UNSET_BAT_FILE, _unset_cmd(taken))
        else:
            files.remove(self.out / _UNSET_SH_FILE)
            files.remove(self.out / _UNSET_BAT_FILE)
        self.stage.generate(files, self.environment)

    def __persist_state(self, files: Writer, state: SyncState) -> None:
        files.write(SyncState.path(self.out), state.render())


def _canonical(value: Toml) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _tables(parent: Toml | None) -> list[dict[str, Toml]]:
    """The tables under `parent`, read defensively from re-parsed TOML."""
    if not isinstance(parent, dict):
        return []
    return [table for table in parent.values() if isinstance(table, dict)]


def _stripped(document: dict[str, Toml], key: str) -> dict[str, Toml]:
    """`document` with `key` popped from the root, every feature, and their targets."""
    for table in (document, *_tables(document.get("feature"))):
        table.pop(key, None)
        for target in _tables(table.get("target")):
            target.pop(key, None)
    return document


def _dotenv_sh(workspace: str) -> str:
    """POSIX dotenv loader rooted from this shard back to the workspace."""
    dotenv = f"{workspace}/.env"
    return f'''#!/usr/bin/env bash
# Generated by mainboard's Provisioner (workspace.dotenv = true). Do not edit by hand.
# A variable already exported in the shell wins over one in `.env`, which only fills gaps.
snapshot="$(mktemp)"
export -p > "$snapshot"
set -a
[ -f "{dotenv}" ] && . "{dotenv}"
set +a
. "$snapshot" 2>/dev/null || true
rm -f "$snapshot"
unset snapshot
'''


def _dotenv_cmd(workspace: str) -> str:
    """Windows dotenv loader rooted from this shard back to the workspace."""
    dotenv = f"{workspace}/.env".replace("/", "\\")
    return rf'''@echo off
rem Generated by mainboard's Provisioner (workspace.dotenv = true). Do not edit by hand.
rem Existing variables win; .env only fills names the calling environment did not define.
if not exist "{dotenv}" goto :eof
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("{dotenv}") do (
    if not defined %%A set "%%A=%%B"
)
'''


def _unset_sh(names: list[str]) -> str:
    """The script taking `names` out of pixi's environment, sourced after the dotenv loader.

    `unset -v`, so a shell function of the same name survives.
    """
    lines = "\n".join(f"unset -v {name}" for name in names)
    return (
        "#!/usr/bin/env bash\n"
        "# Generated by mainboard from the [env] table. Do not edit by hand.\n"
        "# Each name below is declared `false`, which asks for it to be unset rather than set to\n"
        "# an empty string: an empty variable is still defined, and that is a different thing.\n"
        f"{lines}\n"
    )


def _unset_cmd(names: list[str]) -> str:
    """The Windows activation script that removes the environment variables in `names`."""
    lines = "\n".join(f"set {name}=" for name in names)
    return (
        "@echo off\n"
        "rem Generated by mainboard from the [env] table. Do not edit by hand.\n"
        f"{lines}\n"
    )
