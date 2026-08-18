import os
import shlex
from typing import TYPE_CHECKING

from jinja2 import Environment as Jinja
from jinja2 import PackageLoader

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

# Shell templates ship as package data so they stay diffable and shellcheck-able. Autoescaping
# is a jinja2 default meant for HTML/XML output, but this template renders bash, where escaping
# `&` `<` `>` into HTML entities would corrupt the script, so it is off on purpose.
_TEMPLATES = Jinja(
    loader=PackageLoader("mainboard.engines.compile"),
    autoescape=False,  # ruff:ignore[jinja2-autoescape-false]
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

# Where Lmod / environment-modules drops its shell init. The first that exists wins, and on a
# host without modules (a laptop, gold) none exist and module setup degrades to a harmless no-op.
_MODULE_INITS = (
    "/usr/share/lmod/lmod/init/bash",
    "/etc/profile.d/modules.sh",
    "/etc/profile.d/z00_lmod.sh",
    "/etc/profile.d/lmod.sh",
)


def module_init_snippet(inits: Sequence[str] = _MODULE_INITS) -> str:
    """A bash snippet that loads `module` into a non-login shell from the first init that exists.

    `module` is a shell function and is undefined in PBS non-login shells, so a job must source
    the Lmod/environment-modules init before any `module load`. The loop stops at the first
    present init and is a clean no-op on a host that ships none.
    """
    candidates = " ".join(shlex.quote(init) for init in inits)
    return (
        f"for _modinit in {candidates}; do "
        '[ -f "$_modinit" ] && . "$_modinit" && break; done; unset _modinit'
    )


class ActivationScript:
    """Generates a per-host `.mainboard/activate.sh` that sets up the whole runtime in one
    `source`.

    The script sources the module init, `module purge`s, `module load`s the pinned modules
    (`modules` is a per-host `name -> version` map, since Lmod stacks differ machine to
    machine), then applies pixi's own activation (the same env vars, PATH, and activation
    scripts `Provisioner.activated()` applies), and finally exports the directories the
    second-stage toolchains linked their executables into. Sourcing it makes `python -m
    <module>` and an npm-installed tool alike Just Work from a bare PBS or interactive shell.
    Off-cluster the `module` lines are guarded by `command -v module`, so the script degrades
    to pure pixi activation.

    binaries: directories to prepend to PATH after pixi's own activation, the same ones
        `Provisioner.activated()` exports in-process.
    """

    def __init__(self, path: Path, hook: str, binaries: Sequence[Path] = ()) -> None:
        self.path = path
        self.hook = hook
        self.binaries = binaries

    def render(self, modules: Mapping[str, str]) -> str:
        """The `activate.sh` text: module init + purge + load, pixi activation, then PATH.

        With no ``modules`` declared the whole module block is omitted, so the script never
        purges whatever stack the surrounding job had loaded.
        """
        specs = shlex.join(f"{name}/{version}" for name, version in modules.items())
        return _TEMPLATES.get_template("activate.sh.j2").render(
            module_init=module_init_snippet(),
            modules=specs,
            hook=self.hook.strip(),
            binaries=os.pathsep.join(shlex.quote(str(path)) for path in self.binaries),
        )

    def write(self, modules: Mapping[str, str]) -> Path:
        """Write the `activate.sh` loading ``modules`` to :attr:`path` and return it."""
        self.path.write_text(self.render(modules))
        return self.path
