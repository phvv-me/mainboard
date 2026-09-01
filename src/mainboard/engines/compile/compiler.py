import hashlib
import json
import tomllib
from typing import TYPE_CHECKING

from ...core import MissionError, Project
from .pixi_manifest import PixiManifest, cleared, rerooted, selected_manifest
from .state import SyncState

if TYPE_CHECKING:
    from pathlib import Path

    from ...manifest import Manifest, Scope
    from .backend import Pixi
    from .ecosystems import SecondStage
    from .generated import Writer
    from .toml import Toml

# Freshness truth lives in one atomically-replaced state file (see `state.SyncState`), and the
# compiler reads a coherent snapshot and writes the whole next snapshot in a single replace,
# never a partial marker.

_DOTENV_SH_FILE = "dotenv.sh"
_DOTENV_BAT_FILE = "dotenv.bat"
_UNSET_SH_FILE = "unset.sh"
_UNSET_BAT_FILE = "unset.bat"


class Compiler:
    """Turns a manifest into the generated `.mainboard/` env, and says when that env is stale.

    A compile writes the pixi manifest, the dotenv loader, whatever the second-stage
    toolchains install from, and the digest marker that later calls compare against. Nothing
    here provisions anything, so the same write runs under `Provisioner.activated`'s short
    lock and inside `provision`'s longer one.
    """

    def __init__(
        self,
        root: Path,
        manifest: Manifest,
        out: Path,
        pixi: Pixi,
        stage: SecondStage,
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

    def digest(self) -> str:
        """A content hash of the manifest, the key that decides whether a compile is current.

        Over what a compile actually reads, so the tables that only configure a verb are left
        out and editing one never makes an installed environment look stale.
        """
        payload = self.manifest.model_dump(
            mode="json", round_trip=True, exclude=set(self.manifest.uncompiled)
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def install_locked(self, files: Writer, *, resolve: bool) -> None:
        """Install this shard and bless its lock in the order that keeps state honest.

        The refusal compares the lock's own recorded resolution against what is on disk right
        now, so it is a fact about the pair rather than a flag a machine has to remember to
        clear. A host that never solved can therefore install from a lock somebody else solved,
        as long as the manifest and package metadata it received are the ones that lock came
        from. Blessing happens only after a solve returned without raising, so a failed solve
        never leaves a lock that nothing on disk vouches for looking fresh.
        """
        state = SyncState.load(self.out)
        digest = self.resolution_digest()
        if (
            not resolve
            and self.pixi.lock.exists()
            and (state.environment != self.environment or state.solved_from != digest)
        ):
            raise MissionError(
                f"pixi.lock was not solved from this manifest and package metadata. Run "
                f"`{Project().name} install {self.environment} --resolve` on a solve-capable "
                "machine, then "
                "set this host up again."
            )
        self.pixi.install(self.environment, resolve=resolve)
        if resolve:
            self.__persist_state(files, state.model_copy(update={"solved_from": digest}))

    def resolution_digest(self) -> str:
        """Hash everything a solve reads, so a lock can be checked against the tree it sits in.

        The generated pixi manifest answers for the declared dependencies, and every local
        Python project's own metadata answers for the path dependencies pixi resolves through
        `pyproject.toml` rather than through the manifest. Tasks and activation are excluded on
        both counts, since neither can change which versions resolve, and leaving activation in
        would make the digest depend on where the workspace happens to live and so refuse every
        host whose root differs from the machine that solved.

        Reads the compiled manifest from disk, since that is the file pixi will resolve, and
        every caller compiles before asking.
        """
        digest = hashlib.sha256()
        compiled = self._resolution_manifest(self.pixi.manifest.read_text(encoding="utf-8"))
        digest.update(json.dumps(compiled, sort_keys=True, separators=(",", ":")).encode())
        for declared in self._local_python_projects():
            project = self.root / declared / "pyproject.toml"
            digest.update(declared.encode())
            try:
                digest.update(project.read_bytes())
            except FileNotFoundError:
                digest.update(b"\0")
        return digest.hexdigest()

    def stale(self) -> bool:
        """Whether this shard's generated env predates its selected manifest content.

        True once a compile exists but the manifest has changed since it ran, so a caller can
        recompile before activating rather than serve env vars and deps from a stale
        `.mainboard/`. A workspace with nothing compiled yet is not stale, since first
        provisioning is `provision`'s job, not `activated`'s.
        """
        if not self.pixi.manifest.exists():
            return False
        state = SyncState.load(self.out)
        return state.environment != self.environment or state.compiled_from != self.digest()

    def write(self, files: Writer) -> None:
        """Write this shard's generated files through the workspace-locked writer."""
        # The digest is taken before the translation, so an edit landing mid-compile leaves
        # the workspace stale rather than blessing output built from content nobody compiled.
        source_digest = self.digest()
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
        # A crash anywhere above must leave the workspace stale, so the state snapshot only
        # lands once every file compiled from this manifest is already on disk. `solved_from`
        # is carried through untouched: a compile changes what the lock would have to answer
        # to, never what it already answered to, and only a solve may say otherwise.
        self.__persist_state(
            files,
            state.model_copy(
                update={"environment": self.environment, "compiled_from": source_digest}
            ),
        )

    @staticmethod
    def _resolution_manifest(text: str) -> dict[str, Toml]:
        """Return generated Pixi data that can affect dependency resolution.

        Tasks and activation come out of every table that carries them, the workspace-wide pair
        and each feature's own. Only the top-level pair used to be dropped, and a manifest's
        per-environment tasks compile into `[feature.<name>.tasks]`, so editing what
        `[envs.serving.tasks]` runs moved this digest, refused the lock beside it, forced a full
        re-solve on a machine that only wanted to rename a command, and then invalidated the same
        lock on every host already holding it. Renaming a command cannot change which versions
        resolve, which is exactly what this hash is supposed to mean.
        """
        document = tomllib.loads(text)
        for table in (document, *_features(document)):
            table.pop("tasks", None)
            table.pop("activation", None)
            if isinstance(targets := table.get("target"), dict):
                for target in targets.values():
                    if isinstance(target, dict):
                        target.pop("activation", None)
        return document

    def _local_python_projects(self) -> list[str]:
        """Every local Python project path that can participate in this shard's solve.

        Only the `python` ecosystem is considered: pixi resolves against a path dependency's
        own `pyproject.toml`, and that is the one ecosystem this port translates. The manifest
        was projected onto one environment at construction, so unrelated environments never
        enter this metadata digest.
        """
        scopes: list[Scope] = [self.manifest, self.manifest.dev, *self.manifest.on.values()]
        for env in self.manifest.envs.values():
            scopes.extend([env, *env.on.values()])
        python_deps = (
            spec
            for scope in scopes
            if (python := scope.toolchains().get("python"))
            for spec in python.all_deps().values()
        )
        return sorted(
            {
                path
                for spec in python_deps
                if spec.is_path and isinstance(path := (spec.model_extra or {}).get("path"), str)
            }
        )

    def _write_generated_files(self, files: Writer, *, compiled: str) -> None:
        """Write the compiled pixi manifest, the dotenv loader, and the second-stage files.

        pixi is handed conda and Python (see `pixi_manifest.dependency_tables`); every other
        declared ecosystem generates whatever its own manager reads through the second stage.
        """
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
        """Replace the whole state snapshot in one atomic write."""
        files.write(SyncState.path(self.out), state.render())


def _features(document: dict[str, Toml]) -> list[dict[str, Toml]]:
    """Every `[feature.<name>]` table in a compiled pixi document, in declaration order.

    Read defensively because the document is re-parsed TOML rather than a model: a compiled
    manifest that declares no feature, or whose `feature` key is somehow not a table of tables,
    simply contributes nothing to strip.
    """
    features = document.get("feature")
    if not isinstance(features, dict):
        return []
    return [table for table in features.values() if isinstance(table, dict)]


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
    """The generated script that takes `names` out of the environment pixi hands a command.

    Sourced after the dotenv loader, so a variable the workspace declares clear stays clear even
    when `.env` fills it in. `unset -v` rather than a bare `unset` so a shell function of the
    same name is never removed by accident.
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
