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

# What a change reports where a requirement was not there, on either side of the edit.
_ABSENT = "absent"

# The environment a workspace-wide edit resolves against, pixi's own name for the one every
# bare command runs in.
_DEFAULT = "default"

# Where a moved pin is reported from, since the lock is one file for every environment.
_LOCK = "pixi.lock"

# pixi's `[pypi-options]` key naming the Python index a workspace resolves through.
_INDEX_URL = "index-url"

# The characters that separate a name from the constraint after it. A leading one is part of
# the name instead, which is what keeps a scoped npm package (`@openai/codex`) whole while
# still splitting the `@` that npm itself writes between a scoped name and its range.
_OPERATORS = "<>=!~^@ "


class Change(FrozenModel):
    """One requirement or locked version an edit moved, and where it moved.

    name: the dependency that moved.
    where: the manifest table declaring it, or the lock when the solve moved it.
    before: what it was, `absent` when nothing declared it.
    after: what it is now, `absent` when the edit dropped it.
    """

    name: str
    where: str
    before: str
    after: str


class Dependencies:
    """What the workspace declares, edited in the manifest and then re-solved.

    The three verbs a dependency change already needed and had to be done by hand: write the
    requirement into the right table, drop it, or move it to the newest release. Each one ends
    the same way, by handing the manifest to the provisioner that `install` uses, so a change
    is never left declared but unsolved, and each reports the same thing, every constraint and
    every locked version that actually moved.
    """

    def __init__(self, board: Board) -> None:
        """board: the workspace whose manifest is edited and whose environment is re-solved."""
        self.board = board

    @property
    def path(self) -> Path:
        """The workspace manifest file."""
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

        A spec carrying its own constraint is written exactly as given. A bare name is asked of
        the ecosystem's index and pinned to what it publishes, so `add tqdm` and `upgrade tqdm`
        write the same requirement and neither one has to be looked up by hand.

        spec: the requirement, a bare name or a name with the constraint it carries.
        ecosystem: whose resolver installs it, `conda` for the manifest's default one.
        env: an environment name, the workspace-wide table when empty.
        dev: whether it is a development-only requirement.
        resolve: re-solve after the edit, which is what makes the lock answer for it.
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
        """The one table declaring `name`, refusing when none does or several do.

        Refusing on several is the honest answer rather than a guess, since a name declared for
        conda and again for an environment's Python is two different requirements and dropping
        the wrong one is silent until the next solve.
        """
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
        """Where the manifest says this ecosystem resolves from, empty for a public default.

        Only the two the manifest actually configures answer here, the conda channels the
        workspace declares and a Python index it points somewhere other than PyPI, since those
        are the two whose newest release genuinely differs from the public one.
        """
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

        With no flags the whole manifest is searched, so dropping a requirement never asks the
        caller to remember which table it was written into. Flags narrow that search to the
        tables they name, which is also how a name declared in several tables is told apart.

        name: the dependency to drop.
        ecosystem: narrow the search to one resolver's tables.
        env: narrow the search to one environment's tables.
        dev: narrow the search to development-only tables.
        resolve: re-solve after the edit.
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
        """Re-solve the environment and report every pin the lock moved.

        The lock is read before and after through pixi's own frozen listing, so what is
        reported is what the solve did rather than what the edit asked for, which is the
        difference between adding one requirement and learning it dragged forty others with it.

        manifest: the manifest to solve from, the reloaded one after an edit.
        env: the environment to re-solve, the default one when empty.
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
        """Write the edited manifest, then re-solve, reporting the constraint and the lock.

        The manifest is validated by reloading it before anything solves against it, so an edit
        that would not parse is reported as the edit it was rather than as a solver failure
        several minutes later.
        """
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
        """A requirement as written, split into the name and whatever constraint it carries.

        The `@` npm writes between a package and its range is a separator rather than part of the
        constraint, so it is dropped once it has done its job of marking where the name ended.
        """
        cut = next((at for at, mark in enumerate(spec) if at and mark in _OPERATORS), len(spec))
        return spec[:cut].strip(), spec[cut:].strip().removeprefix("@").strip()

    def upgrade(
        self, name: str = "", *, ecosystem: str = "", env: str = "", dev: bool = False
    ) -> list[Change]:
        """Move `name` to its newest release, or the whole lock forward inside its bounds.

        Named, this rewrites one constraint to what the ecosystem's index publishes now, which
        is the only way past a ceiling the manifest itself declares. Unnamed, nothing in the
        manifest changes and the lock is re-solved against the indexes, which moves every pin
        as far as the constraints already allow.

        name: the dependency to bump, every declared one inside its bounds when empty.
        ecosystem: narrow the search to one resolver's tables.
        env: narrow the search to one environment's tables.
        dev: narrow the search to development-only tables.
        """
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
