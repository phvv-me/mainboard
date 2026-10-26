from contextlib import ExitStack, contextmanager
from functools import cached_property
from typing import TYPE_CHECKING

from packaging.utils import canonicalize_name
from patos import FrozenModel

from ..core.errors import MissionError
from ..engines.compile.pixi.lockfile import Lockfile
from ..engines.compile.provisioner import Provisioner
from ..manifest.composing import composition
from .editing import ManifestText
from .indexes import Index
from .slots import Edit, Slot, Tables, declared

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from pathlib import Path

    from ..fleet.board import Board
    from ..manifest import Manifest

_ABSENT = "absent"
_DEFAULT = "default"
_LOCK = "pixi.lock"
# pixi's `[pypi-options]` key naming the Python index a workspace resolves through.
_INDEX_URL = "index-url"

# What separates a name from its constraint; a leading one belongs to the name, keeping a scoped
# npm package (`@openai/codex`) whole while splitting npm's own `name@range`.
_OPERATORS = "<>=!~^@ "


# Every table, re-locked and installed: what an edit does unless its flags narrow it.
_EDIT = Edit()


class Change(FrozenModel):
    """One requirement or locked version an edit moved, and where it moved.

    where: the manifest table declaring it, or the lock when the solve moved it.
    before: `absent` when nothing declared it; `after` likewise when the edit dropped it.
    """

    name: str
    where: str
    before: str
    after: str


class Dependencies:
    """What the workspace declares, edited as pixi's verbs edit it, then re-locked and installed.

    `add`, `remove` and `upgrade` change the manifest, `update` only the lock; each then ends in
    `install`'s provisioner, so nothing is left declared but unsolved unless `--frozen` asked
    for the manifest alone, and reports every constraint and locked version that moved.
    """

    @cached_property
    def _conda_names(self) -> dict[str, str]:
        return Lockfile(self.board.root).conda_names()

    @contextmanager
    def _kept(self, *paths: Path) -> Generator[None]:
        """Put `paths` back as they were when the block fails in any way.

        Absent ones are removed, and the manifest is read again.
        """
        kept = {path: path.read_bytes() if path.is_file() else None for path in paths}
        with ExitStack() as undo:
            undo.callback(self._restore, kept)
            yield
            undo.pop_all()

    @property
    def path(self) -> Path:
        return self.board.project.manifest(self.board.root)

    @staticmethod
    def _split(spec: str) -> tuple[str, str]:
        """A requirement split into its name and constraint, npm's `@` separator dropped."""
        cut = next((at for at, mark in enumerate(spec) if at and mark in _OPERATORS), len(spec))
        return spec[:cut].strip(), spec[cut:].strip().removeprefix("@").strip()

    def __init__(self, board: Board) -> None:
        self.board = board

    def _environment(self, env: str) -> str:
        """`env` confirmed against the manifest, the default environment when empty."""
        if not env:
            return _DEFAULT
        self.board.manifest.environment(env)
        return env

    def _held(self, name: str, slot: Slot) -> bool:
        """Whether a hold keeps `name` as written in `slot`.

        That is the root's `[holds]`, or its environment's when `slot` is one of its tables.
        """
        manifest = self.board.manifest
        holds = manifest.holds
        if slot.path[0] == "envs" and (env := manifest.envs.get(slot.path[1])) is not None:
            holds = holds | env.holds
        return canonicalize_name(name) in {canonicalize_name(held) for held in holds}

    def _locate(self, name: str, tables: Tables) -> Slot:
        """The one table declaring `name`, refusing none, and several rather than guessing."""
        searched = self._searched(tables)
        found = [slot for slot in searched if name in searched[slot]]
        if not found:
            where = ", ".join(sorted(slot.table for slot in searched)) or "no table"
            raise MissionError(f"nothing declares {name!r}. Searched {where}.")
        if len(found) > 1:
            where = ", ".join(sorted(slot.table for slot in found))
            raise MissionError(
                f"{name!r} is declared in {where}. Name one with --lang, --env or --dev."
            )
        return found[0]

    def _pinned(self, name: str, slot: Slot) -> str:
        """The requirement naming what `slot`'s ecosystem publishes as the newest `name`.

        A Python requirement for a package the lock takes from a conda channel follows the
        channels instead: pixi installs the conda build and holds the PyPI requirement to it, so
        PyPI's newer `filelock` 4.2.4 over conda-forge's 4.2.0 made the solve unsatisfiable
        (2026-10-23).
        """
        ecosystem = slot.ecosystem
        if ecosystem == "python" and (held := self._conda_names.get(canonicalize_name(name))):
            ecosystem, name = "conda", held
        index = Index.of(ecosystem)
        index.sources = self._registries(ecosystem)
        return index.pin(index.latest(name))

    def _raised(self, manifest: ManifestText, name: str, slot: Slot) -> Change:
        """`name` raised in `manifest` to what its index publishes, unless already there."""
        before = manifest.constraint(slot.path, name)
        after = self._pinned(name, slot)
        if after != before:
            manifest.put(slot.path, name, spec=after)
        return Change(name=name, where=slot.table, before=before, after=after)

    def _registries(self, ecosystem: str) -> tuple[str, ...]:
        """The conda channels, or a non-PyPI Python index, the manifest resolves from."""
        manifest = self.board.manifest
        if ecosystem == "conda":
            return tuple(manifest.workspace.channels)
        chain = manifest.toolchains().get(ecosystem)
        declared = (chain.model_extra or {}) if chain else {}
        index = declared.get(_INDEX_URL)
        return (index,) if isinstance(index, str) else ()

    def _reloaded(self) -> Manifest:
        """The manifest read again from its file, everything the board derived from it dropped."""
        self.board.shared.pop("manifest", None)
        self.board.shared.pop("resolver", None)
        return self.board.manifest

    def _resolved(
        self,
        manifest: Manifest,
        *,
        env: str,
        update: Sequence[str] | None = None,
        install: bool = True,
    ) -> list[Change]:
        """Re-lock and report every pin the solve itself moved, read via `pixi list`.

        `env` alone when named; otherwise every declared environment, as pixi re-locks its one
        lock file whole, since a host refuses any environment the committed lock left stale.
        Only `env`, or `default`, is installed.

        update: ask the indexes for newer releases inside the declared bounds (`pixi update`),
            for these packages alone when any are named.
        install: install what the lock now pins.
        """
        provisioner = Provisioner(self.board.root, manifest)
        installed = env or _DEFAULT
        changes: list[Change] = []
        for target in [env] if env else [_DEFAULT, *manifest.envs]:
            pixi = provisioner.pixi_for(target)
            before = pixi.locked(target)
            provisioner.provision(target, update=update, install=install and target == installed)
            after = pixi.locked(target)
            where = _LOCK if target == _DEFAULT else f"{_LOCK} [{target}]"
            changes += [
                Change(
                    name=name,
                    where=where,
                    before=before.get(name, _ABSENT),
                    after=after.get(name, _ABSENT),
                )
                for name in sorted(before.keys() | after.keys())
                if before.get(name) != after.get(name)
            ]
        return changes

    def _restore(self, kept: Mapping[Path, bytes | None]) -> None:
        """Write each kept file back, remove those that did not exist, reload the manifest."""
        for path, written in kept.items():
            if written is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(written)
        self._reloaded()

    def _searched(self, tables: Tables) -> dict[Slot, tuple[str, ...]]:
        """The tables a lookup covers, every declared one until a flag narrows it."""
        found = declared(self.board.manifest)
        if not tables.is_narrowed:
            return found
        wanted = set(tables.candidates())
        return {slot: names for slot, names in found.items() if slot in wanted}

    def _settled(
        self,
        manifest: ManifestText,
        changes: list[Change],
        *,
        edit: Edit,
        update: Sequence[str] | None = None,
    ) -> list[Change]:
        """Write the edited manifest, reload it, then re-lock unless `edit` is frozen.

        Reloading fails fast on a bad edit, and a frozen edit leaves the lock for a later solve.

        A bad edit or a failed solve or install puts the manifest and lock back as they were, as
        pixi does, so no `run` is left re-locking against a manifest that cannot solve.

        update: the packages the re-lock frees from their locked versions, every one when empty
            (`resolved`); none when None, the old lock's pins kept wherever they still satisfy.
        """
        with self._kept(self.path, self.board.project.lock(self.board.root)):
            self.path.write_text(manifest.text(), encoding="utf-8", newline="\n")
            reloaded = self._reloaded()
            env, install = edit.tables.env, not edit.relock.no_install
            moved = (
                []
                if edit.relock.frozen
                else self._resolved(reloaded, env=env, update=update, install=install)
            )
        return [*changes, *moved]

    def _slot(self, tables: Tables) -> Slot:
        """Where a new requirement of this shape belongs, preferring a table already there."""
        options = tables.candidates()
        present = declared(self.board.manifest)
        return next((slot for slot in options if slot in present), options[0])

    def _targets(
        self, names: Sequence[str], tables: Tables, *, exclude: Sequence[str]
    ) -> list[tuple[str, Slot]]:
        """What an upgrade raises: the named requirements, else every unheld one but `exclude`."""
        if names:
            return [(name, self._locate(name, tables)) for name in names]
        return [
            (name, slot)
            for slot, declared_names in self._searched(tables).items()
            for name in declared_names
            if name not in exclude and not self._held(name, slot)
        ]

    def add(
        self,
        spec: str,
        *,
        edit: Edit = _EDIT,
    ) -> list[Change]:
        """Declare `spec` in the table `edit` names, then re-lock and install, as `pixi add`.

        A bare name is pinned to what the ecosystem's index publishes, as `upgrade` would.
        """
        self._environment(edit.tables.env)
        name, constraint = Dependencies._split(spec)
        slot = self._slot(edit.tables)
        manifest = ManifestText(self.path.read_text(encoding="utf-8"))
        before = manifest.constraint(slot.path, name) if manifest.declares(slot.path, name) else ""
        after = constraint or self._pinned(name, slot)
        manifest.put(slot.path, name, spec=after)
        change = Change(name=name, where=slot.table, before=before or _ABSENT, after=after)
        return self._settled(manifest, [change], edit=edit)

    def remove(
        self,
        name: str,
        *,
        edit: Edit = _EDIT,
    ) -> list[Change]:
        """Drop `name` from the one table declaring it, then re-lock and install, as `pixi remove`.

        The whole manifest is searched unless `edit` narrows it.
        """
        slot = self._locate(name, edit.tables)
        manifest = ManifestText(self.path.read_text(encoding="utf-8"))
        change = Change(
            name=name,
            where=slot.table,
            before=manifest.constraint(slot.path, name),
            after=_ABSENT,
        )
        manifest.drop(slot.path, name)
        return self._settled(manifest, [change], edit=edit)

    def update(
        self, names: Sequence[str] = (), *, env: str = "", install: bool = True
    ) -> list[Change]:
        """Move the lock to the newest releases the manifest allows, as `pixi update` does.

        Only the named packages move when any are named, and the manifest is left as written.
        """
        self._environment(env)
        return self._resolved(self.board.manifest, env=env, update=tuple(names), install=install)

    def upgrade(
        self,
        names: Sequence[str] = (),
        *,
        edit: Edit = _EDIT,
        exclude: Sequence[str] = (),
    ) -> list[Change]:
        """Raise requirements to their newest releases and re-lock, as `pixi upgrade` does.

        It raises the named ones, or every declared one but `exclude` and the manifest's `[holds]`
        when none is named; a held one is raised only by naming it.

        A requirement with no version to raise (a path, git or url source, or a workspace member
        named by version) is left as written, and one written as a table keeps every field but
        its version.
        """
        manifest = ManifestText(self.path.read_text(encoding="utf-8"))
        # A member installs from its own directory wherever it is named, so no index speaks for it.
        members = {
            member.package.name for member in composition(self.path).members if member.package
        }
        raised = [
            self._raised(manifest, name, slot)
            for name, slot in self._targets(names, edit.tables, exclude=exclude)
            if canonicalize_name(name) not in members and manifest.versioned(slot.path, name)
        ]
        changes = [change for change in raised if change.before != change.after]
        # Freed as `pixi upgrade` frees them: a raised requirement locked where an older pin
        # still satisfies its other half (a conda build under a PyPI requirement) cannot solve.
        return self._settled(manifest, changes, edit=edit, update=tuple(names))
