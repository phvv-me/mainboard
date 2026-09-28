import ast
import os
import platform
import re
import shlex
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=exec() launches an explicitly declared manifest command as argv without a shell
import sys
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from ...core.errors import MissionError

if TYPE_CHECKING:
    from pathlib import Path

_EXEC_TIMEOUT = 20.0

# A `{{ }}` template, and the two things one may hold: a dotted name (`vars.home`) or a call of a
# scope function with literal arguments (`env('HOME', '')`). Nothing else is evaluated, so a
# manifest can never run code beyond the functions named below.
_TEMPLATE = re.compile(r"\{\{\s*(.*?)\s*\}\}", re.DOTALL)
_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_CALL = re.compile(r"([A-Za-z_]\w*)\((.*)\)", re.DOTALL)

type Json = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
type Scope = dict[str, Json | Callable[..., Json]]


class Interpolator:
    """Renders `{{ }}` templates across a parsed manifest tree.

    mise's names: `config_root`, `env(name, default)`, `num_cpus()`, `arch()`, `os_name()`,
    `exec(cmd)`. `[vars]` render first in order, each seeing its predecessors, then every string
    with `vars.*` in scope. Strings without templates pass through, keeping submit-time
    expressions (`mem_gb = "attempt * 50"`) out of load time.
    """

    def __init__(self, root: Path) -> None:
        """root: the directory holding the manifest, exposed as `config_root`."""
        self.root = root
        self.globals: Scope = {
            "config_root": str(root),
            "env": _env,
            "num_cpus": os.cpu_count,
            "arch": platform.machine,
            "os_name": _os_name,
            "exec": _exec,
        }

    def rendered(self, tree: dict[str, Json]) -> dict[str, Json]:
        """The parsed manifest tree with every template string rendered."""
        scope = dict(self.globals)
        variables = self.__rendered_vars(tree, scope)
        scope["vars"] = variables
        rendered = {key: self.__walk(value, scope, at=key) for key, value in tree.items()}
        rendered["vars"] = variables
        return rendered

    def __render(self, text: str, scope: Scope, *, at: str) -> str:
        if "{{" not in text and "{%" not in text:
            return text
        try:
            if "{%" in text:
                raise ValueError("`{% %}` statements are not evaluated; use `{{ name }}`")
            return _TEMPLATE.sub(lambda found: str(_evaluated(found[1], scope)), text)
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


def _evaluated(expression: str, scope: Scope) -> Json:
    """What one template's `expression` names: a scope value by dotted name, or a call."""
    if call := _CALL.fullmatch(expression):
        function = scope.get(call[1])
        if not callable(function):
            raise ValueError(f"{call[1]!r} is not a function a template can call")
        arguments = ast.literal_eval(f"({call[2]},)") if call[2].strip() else ()
        return function(*arguments)
    if not _NAME.fullmatch(expression):
        raise ValueError(f"{expression!r} is not a name or a call a template evaluates")
    value: object = scope
    for part in expression.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"{expression!r} is undefined")
        value = value[part]
    if callable(value):
        raise ValueError(f"{expression!r} is a function; call it as {expression}()")
    return value  # type: ignore[return-value]


def _env(name: str, default: str = "") -> str:
    """The environment variable `name`, or `default` when unset."""
    return os.environ.get(name, default)


def _os_name() -> str:
    """The running platform family: `linux`, `macos`, or `windows`."""
    return {"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux")


def _exec(command: str) -> str:
    """The stripped stdout of `command`, parsed consistently and run without a shell."""
    try:
        result = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            timeout=_EXEC_TIMEOUT,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise MissionError(f"exec({command!r}) failed: {error}") from error
