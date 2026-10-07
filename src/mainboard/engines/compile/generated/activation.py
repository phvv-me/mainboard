# The bash that enters an environment: the host's module stack, pixi's own activation, the
# second-stage binaries and what `runtime.activation` adds. It is part of the environment it
# enters, written into that environment's own directory by the install that built it (a
# content-addressed prefix, or `<state>/envs/<env>/`), never into the state directory itself;
# `mb shell-hook` names it for a shell to source.

import shlex
import sys
from typing import TYPE_CHECKING

from ....core.project import Project

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

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


def activation(
    hook: str,
    *,
    modules: Sequence[str] = (),
    binaries: Sequence[Path] = (),
    runtime: bool = True,
) -> str:
    """The bash entering an environment whose pixi activation is `hook`.

    Declared modules are part of the environment's identity, so they must load: a host without a
    module system is refused rather than run on whatever it has. Module inits and conda
    activation scripts are not nounset-safe, so `set -u` is relaxed while activating and
    restored after, for callers running under it.

    modules: the host's module stack, `name/version` specs in load order, loaded before pixi.
    binaries: second-stage executable directories, leading pixi's `bin/`.
    runtime: end with what `runtime.activation` prints, for a caller not applying it itself.
    """
    # A failure returns its status from a sourced activation and exits one run by `bash -c`.
    stop = "{ _mb_status=$?; return $_mb_status 2>/dev/null || exit $_mb_status; }"
    lines = ["case $- in *u*) _mb_nounset=1; set +u;; *) _mb_nounset=0;; esac"]
    if modules:
        lines += [
            module_init_snippet(),
            "command -v module >/dev/null 2>&1 || { echo 'mainboard: declared host modules "
            f"require a working module system' >&2; false; }} || {stop}",
            f"module purge || {stop}",
            f"module load {shlex.join(modules)} || {stop}",
        ]
    lines.append(hook.strip())
    if binaries:
        joined = ":".join(shlex.quote(str(path)) for path in binaries)
        lines.append(f'export PATH={joined}:"$PATH"')
    if runtime:
        module = shlex.join([sys.executable, "-m", f"{Project().package}.runtime.activation"])
        lines.append(f'eval "$({module})"')
    lines.append('if [ "$_mb_nounset" = 1 ]; then set -u; fi; unset _mb_nounset')
    return "\n".join(lines) + "\n"


def write(
    path: Path, hook: str, *, modules: Mapping[str, str], binaries: Sequence[Path] = ()
) -> Path:
    """Write the activation of the environment at `path`'s directory, with line feeds, since
    bash reads a carriage return as part of each command."""
    text = activation(hook, modules=module_specs(modules), binaries=binaries)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
