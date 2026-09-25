from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from ..engines.compile.provisioner import Provisioner
from .editing import ManifestText
from .indexes import Index
from .slots import Slot, candidates, declared

if TYPE_CHECKING:
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
    """What the workspace declares: add, remove or upgrade in the manifest, then re-solve.

    Each verb ends in `install`'s provisioner, so nothing is left declared but unsolved, and
    reports every constraint and locked version that moved.
    """

    def __init__(self, board: Board) -> None:
        self.board = board

    @property
    def path(self) -> Path:
        return self.board.root / self.board.project.manifest

    def add(
        self,
        spec: str,
        *,
        ecosystem: str = "conda",
        env: str = "",
        dev: bool = False,
        resolve: bool = True,
    ) -> list[Change]:
        """Declare `spec` in the table its flags name, then re-solve.

        A bare name is pinned to what the ecosystem's index publishes, as `upgrade` would.

        env: an environment name, the workspace-wide table when empty.
        """
        self.environment(env)
        name, constraint = Dependencies._split(spec)
        slot = self.slot(ecosystem=ecosystem, env=env, dev=dev)
        manifest = ManifestText(self.path.read_text(encoding="utf-8"))
        before = manifest.constraint(slot.path, name) if manifest.declares(slot.path, name) else ""
        after = constraint or self.pinned(name, slot)
        manifest.put(slot.path, name, spec=after)
        change = Change(name=name, where=slot.table, before=before or _ABSENT, after=after)
        return self.settled(manifest, change, env=env, resolve=resolve)

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
        resolve: bool = True,
    ) -> list[Change]:
        """Drop `name` from the one table declaring it, then re-solve.

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
        return self.settled(manifest, change, env=env, resolve=resolve)

    def resolved(self, manifest: Manifest, *, env: str, refresh: bool = False) -> list[Change]:
        """Re-solve `env` and report every pin the solve itself moved, read via `pixi list`.

        refresh: ask the indexes for newer releases inside the declared bounds.
        """
        provisioner = Provisioner(self.board.root, manifest)
        target = env or _DEFAULT
        pixi = provisioner.pixi_for(target)
        before = pixi.locked(target)
        provisioner.provision(target, resolve=True, refresh=refresh)
        after = pixi.locked(target)
        return [
            Change(
                name=name,
                where=_LOCK,
                before=before.get(name, _ABSENT),
                after=after.get(name, _ABSENT),
            )
            for name in sorted(before.keys() | after.keys())
            if before.get(name) != after.get(name)
        ]

    def searched(self, *, ecosystem: str, env: str, dev: bool) -> dict[Slot, tuple[str, ...]]:
        """The tables a lookup covers, every declared one until a flag narrows it."""
        found = declared(self.board.manifest)
        if not ecosystem and not env and not dev:
            return found
        wanted = set(candidates(ecosystem=ecosystem or "conda", env=env, dev=dev))
        return {slot: names for slot, names in found.items() if slot in wanted}

    def settled(
        self, manifest: ManifestText, change: Change, *, env: str, resolve: bool
    ) -> list[Change]:
        """Write the edited manifest, reload it (failing fast on a bad edit), then re-solve."""
        self.path.write_text(manifest.text(), encoding="utf-8")
        self.board.shared.pop("manifest", None)
        self.board.shared.pop("resolver", None)
        reloaded = self.board.manifest
        if not resolve:
            return [change]
        return [change, *self.resolved(reloaded, env=env)]

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

    def upgrade(
        self, name: str = "", *, ecosystem: str = "", env: str = "", dev: bool = False
    ) -> list[Change]:
        """Move `name` to its newest release, past its ceiling, or the whole lock within bounds."""
        if not name:
            self.environment(env)
            return self.resolved(self.board.manifest, env=env, refresh=True)
        slot = self.locate(name, ecosystem=ecosystem, env=env, dev=dev)
        manifest = ManifestText(self.path.read_text(encoding="utf-8"))
        before = manifest.constraint(slot.path, name)
        after = self.pinned(name, slot)
        manifest.put(slot.path, name, spec=after)
        change = Change(name=name, where=slot.table, before=before, after=after)
        return self.settled(manifest, change, env=env, resolve=True)
