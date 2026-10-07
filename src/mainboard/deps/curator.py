from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from ..engines.compile.provisioner import Provisioner
from .editing import ManifestText
from .indexes import Index
from .slots import Slot, candidates, declared

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ..board import Board
    from ..manifest import Manifest

_ABSENT = "absent"
_DEFAULT = "default"
_LOCK = "pixi.lock"
# pixi's `[pypi-options]` key naming the Python index a workspace resolves through.
_INDEX_URL = "index-url"

# What separates a name from its constraint; a leading one belongs to the name, keeping a scoped
# npm package (`@openai/codex`) whole while splitting npm's own `name@range`.
_OPERATORS = "<>=!~^@ "


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

    def __init__(self, board: Board) -> None:
        self.board = board

    @property
    def path(self) -> Path:
        return self.board.project.manifest(self.board.root)

    def add(
        self,
        spec: str,
        *,
        ecosystem: str = "conda",
        env: str = "",
        dev: bool = False,
        install: bool = True,
        frozen: bool = False,
    ) -> list[Change]:
        """Declare `spec` in the table its flags name, then re-lock and install, as `pixi add`.

        A bare name is pinned to what the ecosystem's index publishes, as `upgrade` would.

        env: an environment name, the workspace-wide table when empty.
        install: install what the new lock pins (`--no-install` leaves the environment).
        frozen: edit the manifest alone, leaving the lock as it stands (`--frozen`).
        """
        self.environment(env)
        name, constraint = Dependencies._split(spec)
        slot = self.slot(ecosystem=ecosystem, env=env, dev=dev)
        manifest = ManifestText(self.path.read_text(encoding="utf-8"))
        before = manifest.constraint(slot.path, name) if manifest.declares(slot.path, name) else ""
        after = constraint or self.pinned(name, slot)
        manifest.put(slot.path, name, spec=after)
        change = Change(name=name, where=slot.table, before=before or _ABSENT, after=after)
        return self.settled(manifest, [change], env=env, install=install, frozen=frozen)

    def environment(self, env: str) -> str:
        """`env` confirmed against the manifest, the default environment when empty."""
        if not env:
            return _DEFAULT
        self.board.manifest.environment(env)
        return env

    def locate(self, name: str, *, ecosystem: str, env: str, dev: bool) -> Slot:
        """The one table declaring `name`, refusing none, and several rather than guessing."""
        searched = self.searched(ecosystem=ecosystem, env=env, dev=dev)
        found = [slot for slot in searched if name in searched[slot]]
        if not found:
            where = ", ".join(sorted(slot.table for slot in searched)) or "no table"
            raise MissionError(f"nothing declares {name!r}. Searched {where}.")
        if len(found) > 1:
            tables = ", ".join(sorted(slot.table for slot in found))
            raise MissionError(
                f"{name!r} is declared in {tables}. Name one with --lang, --env or --dev."
            )
        return found[0]

    def pinned(self, name: str, slot: Slot) -> str:
        """The requirement naming what `slot`'s ecosystem publishes as the newest `name`."""
        index = Index.of(slot.ecosystem)
        index.sources = self.registries(slot.ecosystem)
        return index.pin(index.latest(name))

    def registries(self, ecosystem: str) -> tuple[str, ...]:
        """The conda channels, or a non-PyPI Python index, the manifest resolves from."""
        manifest = self.board.manifest
        if ecosystem == "conda":
            return tuple(manifest.workspace.channels)
        chain = manifest.toolchains().get(ecosystem)
        declared = (chain.model_extra or {}) if chain else {}
        index = declared.get(_INDEX_URL)
        return (index,) if isinstance(index, str) else ()

    def remove(
        self,
        name: str,
        *,
        ecosystem: str = "",
        env: str = "",
        dev: bool = False,
        install: bool = True,
        frozen: bool = False,
    ) -> list[Change]:
        """Drop `name` from the one table declaring it, then re-lock and install, as `pixi remove`.

        The whole manifest is searched unless the flags narrow it to the tables they name.
        """
        slot = self.locate(name, ecosystem=ecosystem, env=env, dev=dev)
        manifest = ManifestText(self.path.read_text(encoding="utf-8"))
        change = Change(
            name=name,
            where=slot.table,
            before=manifest.constraint(slot.path, name),
            after=_ABSENT,
        )
        manifest.drop(slot.path, name)
        return self.settled(manifest, [change], env=env, install=install, frozen=frozen)

    def resolved(
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

    def searched(self, *, ecosystem: str, env: str, dev: bool) -> dict[Slot, tuple[str, ...]]:
        """The tables a lookup covers, every declared one until a flag narrows it."""
        found = declared(self.board.manifest)
        if not ecosystem and not env and not dev:
            return found
        wanted = set(candidates(ecosystem=ecosystem or "conda", env=env, dev=dev))
        return {slot: names for slot, names in found.items() if slot in wanted}

    def settled(
        self,
        manifest: ManifestText,
        changes: list[Change],
        *,
        env: str,
        install: bool,
        frozen: bool = False,
    ) -> list[Change]:
        """Write the edited manifest, reload it (failing fast on a bad edit), then re-lock,
        unless `frozen` leaves the lock for a later solve."""
        self.path.write_text(manifest.text(), encoding="utf-8", newline="\n")
        self.board.shared.pop("manifest", None)
        self.board.shared.pop("resolver", None)
        reloaded = self.board.manifest
        if frozen:
            return changes
        return [*changes, *self.resolved(reloaded, env=env, install=install)]

    def slot(self, *, ecosystem: str, env: str, dev: bool) -> Slot:
        """Where a new requirement of this shape belongs, preferring a table already there."""
        options = candidates(ecosystem=ecosystem, env=env, dev=dev)
        present = declared(self.board.manifest)
        return next((slot for slot in options if slot in present), options[0])

    @staticmethod
    def _split(spec: str) -> tuple[str, str]:
        """A requirement split into its name and constraint, npm's `@` separator dropped."""
        cut = next((at for at, mark in enumerate(spec) if at and mark in _OPERATORS), len(spec))
        return spec[:cut].strip(), spec[cut:].strip().removeprefix("@").strip()

    def update(
        self, names: Sequence[str] = (), *, env: str = "", install: bool = True
    ) -> list[Change]:
        """Move the lock to the newest releases the manifest already allows, the named packages
        alone when any are named, as `pixi update` does; the manifest is left as written."""
        self.environment(env)
        return self.resolved(self.board.manifest, env=env, update=tuple(names), install=install)

    def upgrade(
        self,
        names: Sequence[str] = (),
        *,
        ecosystem: str = "",
        env: str = "",
        dev: bool = False,
        exclude: Sequence[str] = (),
        install: bool = True,
    ) -> list[Change]:
        """Raise requirements to the newest releases their ecosystems publish, then re-lock, as
        `pixi upgrade` does: the named ones, or every declared one but `exclude` when none is.

        A requirement with no version to raise (a path, git or url source) is left as written,
        and one written as a table keeps every field but its version.
        """
        if names:
            targets = [
                (name, self.locate(name, ecosystem=ecosystem, env=env, dev=dev)) for name in names
            ]
        else:
            targets = [
                (name, slot)
                for slot, declared_names in self.searched(
                    ecosystem=ecosystem, env=env, dev=dev
                ).items()
                for name in declared_names
                if name not in exclude
            ]
        manifest = ManifestText(self.path.read_text(encoding="utf-8"))
        changes: list[Change] = []
        for name, slot in targets:
            if not manifest.versioned(slot.path, name):
                continue
            before = manifest.constraint(slot.path, name)
            after = self.pinned(name, slot)
            if after != before:
                manifest.put(slot.path, name, spec=after)
                changes.append(Change(name=name, where=slot.table, before=before, after=after))
        return self.settled(manifest, changes, env=env, install=install)
