from typing import TYPE_CHECKING, Annotated, NoReturn

from cyclopts import Parameter

from ...deps import Edit
from ...engines.compile import Lockfile
from ...engines.compile.generated.files import GeneratedFiles
from ...engines.compile.pixi.policy import Locking
from ...engines.pack import Formats
from ...render import progress
from ..base import Verbs
from ..options import Output

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...deps import Change

# The verbs' defaults: compact output, every manifest table re-locked and installed, and the
# executable alone.
_COMPACT = Output()
_EDIT = Edit()
_LOCKING = Locking()
_EXECUTABLE = Formats()


def _changed(changes: Sequence[Change], output: Output, *, title: str) -> None:
    """Print one edit's constraint move and every pin its solve moved, as one table.

    Both are one fact, a version moving somewhere, so they share one shape; `where` tells a
    manifest table's requirement from the lock's pins the solve dragged along.
    """
    output.print_rows([change.model_dump() for change in changes], title=title)


class EnvironmentVerbs(Verbs):
    """The `environments` verbs."""

    def add(
        self,
        spec: str,
        *,
        edit: Edit = _EDIT,
        output: Output = _COMPACT,
    ) -> None:
        """Declare a dependency in the manifest, re-lock and install, as `pixi add` does.

        A bare name is pinned to whatever the ecosystem's index publishes right now, and a name
        carrying its own constraint is written exactly as given. The table it lands in is the
        one the flags name, and where the manifest already writes that kind of requirement in a
        particular table, the edit joins it there. Shows what the manifest and the lock did.
        """
        with progress(f"adding {spec}"):
            changes = self.session.board("local").deps().add(spec, edit=edit)
        _changed(changes, output, title="add")

    def install(
        self,
        env: str = "",
        *,
        all_: Annotated[bool, Parameter(name=["--all", "-a"])] = False,
        locking: Locking = _LOCKING,
        profile: str = "",
    ) -> None:
        """Install an environment, solving the lock first when it no longer answers the manifest.

        What `pixi install` does: `--locked` refuses a lock that no longer answers the manifest
        instead of solving it, and `--frozen` installs the lock as it stands without asking.
        Another machine is onboarded with `host setup`, which runs `install --locked` there, so a
        host installs what this machine solved and never solves for itself.

        Args:
            env: the environment name, this machine's declared profile choice when omitted.
            all_: install every declared environment.
            profile: the declared host profile describing this machine, so the environment's
                activation carries that host's modules; what `setup` passes when a host installs.
        """
        policy = locking.policy
        names = self.session.environments() if all_ else [env]
        for name in names:
            with progress(f"installing {name or 'the environment'}"):
                self.session.board("local").install(name, policy=policy, profile=profile)

    def list_(self, *command: str, env: str = "default") -> NoReturn:
        """List the packages installed in an environment, through `pixi list`.

        Args:
            command: pixi's own arguments, a package regex first (`mb list torch --json`).
            env: the environment name.
        """
        self.session.pixi_verb("list", command, env=env)

    def lock(self, env: str = "", *, check: bool = False) -> int:
        """Solve the lock where it no longer answers the manifest, installing nothing.

        What `pixi lock` does. Every declared environment is locked unless one is named, since
        a shared input (a workspace package's pyproject.toml) moves them all at once and a lock
        left behind fails on the host that installs it. A lock that already answers is left as
        it is, and a section of an environment the manifest no longer declares goes.

        Args:
            env: the environment to lock, every declared one when omitted.
            check: exit nonzero when the lock had to change, the gate for a commit or CI.
        """
        root = self.session.workspace_root()
        committed = self.session.project.lock(root)
        before = committed.read_bytes() if committed.is_file() else b""
        for name in [env] if env else self.session.environments():
            with progress(f"locking {name}"):
                self.session.board("local").install(name, install=False)
        with GeneratedFiles(directory=self.session.project.out(root)).locked() as files:
            Lockfile(root).retain(files, self.session.environments())
        return int(check and committed.read_bytes() != before)

    def pack_(
        self,
        env: str = "default",
        *,
        on: str = "local",
        formats: Formats = _EXECUTABLE,
        output: Output = _COMPACT,
    ) -> None:
        """Build an environment into files a machine runs without installing anything.

        Always a self-extracting executable (pixi-pack: `./file` unpacks `env/` and
        `activate.sh`, no pixi or network needed); `--image` adds an OCI image (slim Debian plus
        the environment, CUDA from its own wheels, the driver from `--gpus all`), `--sif` an
        Apptainer file for HPC, `--push` the image to a registry. Each is named by the
        environment's digest, so an unchanged lock is never built twice. Built on the machine
        `--on` names, whose package mirrors are close, never uploaded from here. The workspace's
        own code is not inside: a dispatch ships it, as always.

        Args:
            env: the environment, installed on that machine already (`host setup --env`).
            on: the host to build on, `local` for this machine (Linux or macOS).
        """
        with progress(f"packing {env} on {on}"):
            packed = self.session.board(on).pack(env, formats)
        output.print_record(packed.model_dump(), title="pack")

    def remove(
        self,
        name: str,
        *,
        edit: Edit = _EDIT,
        output: Output = _COMPACT,
    ) -> None:
        """Drop a dependency from the manifest, re-lock and install, as `pixi remove` does.

        With no flags the whole manifest is searched, so dropping a requirement never asks
        which table it was written into. `--lang`, `--env` and `--dev` narrow that search to one
        ecosystem's, one environment's or the development-only tables, which is also how a name
        declared in more than one table is told apart.

        Args:
            name: the dependency to drop.
        """
        with progress(f"removing {name}"):
            changes = self.session.board("local").deps().remove(name, edit=edit)
        _changed(changes, output, title="remove")

    def tree(self, *command: str, env: str = "default") -> NoReturn:
        """Show an environment's dependency tree, through `pixi tree`.

        Args:
            command: pixi's own arguments, a package regex first (`mb tree numpy --invert`).
            env: the environment name.
        """
        self.session.pixi_verb("tree", command, env=env)

    def update_lock(
        self, *names: str, env: str = "", no_install: bool = False, output: Output = _COMPACT
    ) -> None:
        """Move the lock to the newest releases the manifest allows, as `pixi update` does.

        The manifest is left as written, so no requirement moves past its declared bounds; that
        is `upgrade`. Every locked package moves unless some are named, and every environment
        is re-locked unless one is named.

        Args:
            names: the packages to move, every one when none is named.
            env: only this environment's lock.
            no_install: re-lock without installing.
        """
        with progress(f"updating {', '.join(names) or 'the lock'}"):
            changes = (
                self.session.board("local").deps().update(names, env=env, install=not no_install)
            )
        _changed(changes, output, title="update")

    def upgrade(
        self,
        *names: str,
        edit: Edit = _EDIT,
        exclude: tuple[str, ...] = (),
        output: Output = _COMPACT,
    ) -> None:
        """Raise requirements to their newest releases, re-lock and install, as `pixi upgrade`.

        The manifest's constraint is rewritten to what each ecosystem publishes now, which is
        the way past a ceiling the manifest declares; staying inside the bounds is `update`.
        Every declared requirement moves unless some are named; one written as a table keeps
        every field but its version, and a path, git or url source is left alone. `--lang`,
        `--env` and `--dev` narrow which tables are searched.

        Args:
            names: the requirements to raise, every declared one when none is named.
            exclude: a requirement to leave as written when raising every one, repeatable.
        """
        with progress(f"upgrading {', '.join(names) or 'every requirement'}"):
            changes = self.session.board("local").deps().upgrade(names, edit=edit, exclude=exclude)
        _changed(changes, output, title="upgrade")
