import shlex
import sys
from functools import cache
from typing import TYPE_CHECKING

from ....core.project import Project

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from jinja2 import Environment


@cache
def _templates() -> Environment:
    """The shell templates shipped as package data, so they stay diffable and shellcheck-able.

    Autoescaping is off because HTML-escaping `&` `<` `>` corrupts bash. Built lazily because
    jinja2 costs 6 ms of every cold start and only an install renders one.
    """
    from jinja2 import Environment, PackageLoader

    return Environment(
        loader=PackageLoader("mainboard.engines.compile"),
        autoescape=False,  # ruff:ignore[jinja2-autoescape-false]
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


# Where Lmod / environment-modules drops its shell init, first existing one wins.
_MODULE_INITS = (
    "/usr/share/lmod/lmod/init/bash",
    "/etc/profile.d/modules.sh",
    "/etc/profile.d/z00_lmod.sh",
    "/etc/profile.d/lmod.sh",
)


def module_init_snippet(inits: Sequence[str] = _MODULE_INITS) -> str:
    """Bash that defines the `module` function, undefined in PBS non-login shells.

    It sources the first init that exists and is a no-op on a host with none (a laptop, gold).
    """
    candidates = " ".join(shlex.quote(init) for init in inits)
    return (
        f"for _modinit in {candidates}; do "
        '[ -f "$_modinit" ] && . "$_modinit" && break; done; unset _modinit'
    )


def module_specs(modules: Mapping[str, str]) -> tuple[str, ...]:
    """Each declared module, in load order, as `name/version` or bare `name`."""
    return tuple(f"{name}/{version}" if version else name for name, version in modules.items())


class ActivationScript:
    """A per-host `.mainboard/activate.sh` that sets up the whole runtime in one `source`.

    In order: the module init, `module purge` and `module load` of the per-host pinned modules
    (which must load; an empty map skips the block), pixi's activation (as
    `Provisioner.activated()` applies it), `binaries` on PATH, then what `runtime.activation`
    prints, run by the interpreter that wrote the script so no PATH decides which tool answers.

    binaries: the second-stage directories `Provisioner.activated()` also exports.
    """

    def __init__(self, path: Path, hook: str, binaries: Sequence[Path] = ()) -> None:
        self.path = path
        self.hook = hook
        self.binaries = binaries

    def render(self, modules: Mapping[str, str]) -> str:
        """The `activate.sh` text; no `modules` omits the block, never purging the job's stack."""
        specs = shlex.join(module_specs(modules))
        return (
            _templates()
            .get_template("activate.sh.j2")
            .render(
                module_init=module_init_snippet(),
                modules=specs,
                hook=self.hook.strip(),
                # A colon even on Windows: only bash reads this, and `;` made one unusable entry.
                binaries=":".join(shlex.quote(str(path)) for path in self.binaries),
                runtime=shlex.join([sys.executable, "-m", f"{Project().name}.runtime.activation"]),
            )
        )

    def write(self, modules: Mapping[str, str]) -> Path:
        """Write `activate.sh` with line feeds, since bash reads `\\r` as part of each command."""
        self.path.write_text(self.render(modules), encoding="utf-8", newline="\n")
        return self.path
