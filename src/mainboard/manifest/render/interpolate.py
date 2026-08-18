import os
import platform
import sys
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from plumbum import local

from ...core.errors import MissionError

if TYPE_CHECKING:
    from pathlib import Path

_EXEC_TIMEOUT = 20.0

type Json = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
type Scope = dict[str, Json | Callable[..., Json]]


class Interpolator:
    """Renders `{{ }}` templates across a parsed manifest tree.

    The vocabulary follows mise's proven names: `config_root`, `env(name,
    default)`, `num_cpus()`, `arch()`, `os_name()`, and `exec(cmd)`. `[vars]`
    entries render first, in declaration order, each seeing the ones before
    it, then every string in the tree renders with `vars.*` in scope. Strings
    without `{{` pass through untouched, which is what keeps submit-time
    expressions (`mem_gb = "attempt * 50"`) out of load-time rendering.
    """

    def __init__(self, root: Path) -> None:
        """root: the directory holding the manifest, exposed as `config_root`."""
        self.root = root
        self.engine = SandboxedEnvironment(undefined=StrictUndefined)
        self.globals: Scope = {
            "config_root": str(root),
            "env": _env,
            "num_cpus": os.cpu_count,
            "arch": platform.machine,
            "os_name": _os_name,
            "exec": _exec,
        }

    def rendered(self, tree: dict[str, Json]) -> dict[str, Json]:
        """The manifest tree with every template string rendered in place.

        tree: the parsed TOML document.
        """
        scope = dict(self.globals)
        scope["vars"] = self.__rendered_vars(tree, scope)
        rendered = {key: self.__walk(value, scope, at=key) for key, value in tree.items()}
        rendered["vars"] = scope["vars"]
        return rendered

    def __render(self, text: str, scope: Scope, *, at: str) -> str:
        if "{{" not in text and "{%" not in text:
            return text
        try:
            return self.engine.from_string(text).render(scope)
        except Exception as error:
            raise MissionError(f"template at {at} failed: {error}") from error

    def __rendered_vars(self, tree: Mapping[str, Json], scope: Scope) -> dict[str, Json]:
        """`[vars]` rendered in declaration order, each seeing its predecessors."""
        raw = tree.get("vars", {})
        if not isinstance(raw, dict):
            raise MissionError("[vars] must be a table of strings")
        landed: dict[str, Json] = {}
        for name, value in raw.items():
            landed[name] = self.__walk(value, {**scope, "vars": landed}, at=f"vars.{name}")
        return landed

    def __walk(self, value: Json, scope: Scope, *, at: str) -> Json:
        if isinstance(value, str):
            return self.__render(value, scope, at=at)
        if isinstance(value, dict):
            return {key: self.__walk(item, scope, at=f"{at}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [self.__walk(item, scope, at=f"{at}[]") for item in value]
        return value


def _env(name: str, default: str = "") -> str:
    """The environment variable `name`, or `default` when unset."""
    return os.environ.get(name, default)


def _os_name() -> str:
    """The running platform family: `linux`, `macos`, or `windows`."""
    return {"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux")


def _exec(command: str) -> str:
    """The stripped stdout of `command`, run through the shell with a hard cap."""
    try:
        return str(local["bash"]["-c", command].run(timeout=_EXEC_TIMEOUT)[1]).strip()
    except Exception as error:
        raise MissionError(f"exec({command!r}) failed: {error}") from error
