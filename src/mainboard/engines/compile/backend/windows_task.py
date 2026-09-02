import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from plumbum import local

from ....core import MissionError
from ....manifest.schema.spec import Json
from .result import CommandResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from plumbum.commands.base import BaseCommand

_TEMPLATE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def _table(value: Json | None, *, field: str, task: str) -> dict[str, Json]:
    """Return one task table, refusing a generated manifest whose shape is not executable."""
    if not isinstance(value, dict):
        raise MissionError(f"task {task!r} has a non-table {field!r} value")
    return value


def _strings(value: Json | None, *, field: str, task: str) -> tuple[str, ...]:
    """Normalize one string-or-string-list task field."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(cast("list[str]", value))
    raise MissionError(f"task {task!r} has a non-string {field!r} value")


@dataclass(frozen=True)
class WindowsTask:
    """One generated Pixi task reduced to Mainboard's declared task surface."""

    name: str
    command: str
    cwd: Path
    env: dict[str, str]
    dependencies: tuple[str, ...]
    arguments: tuple[str, ...]

    @classmethod
    def parse(cls, name: str, value: Json, *, manifest: Path) -> WindowsTask:
        """Parse the generated shorthand or table form relative to its Pixi manifest."""
        if isinstance(value, str):
            body: dict[str, Json] = {"cmd": value}
        else:
            body = _table(value, field="definition", task=name)
        command = body.get("cmd", "")
        if not isinstance(command, str):
            raise MissionError(f"task {name!r} has a non-string 'cmd' value")
        cwd_value = body.get("cwd", "")
        if not isinstance(cwd_value, str):
            raise MissionError(f"task {name!r} has a non-string 'cwd' value")
        cwd = Path(cwd_value)
        if not cwd.is_absolute():
            cwd = manifest.parent / cwd
        environment = _table(body.get("env", {}), field="env", task=name)
        env = {key: item for key, item in environment.items() if isinstance(item, str)}
        if len(env) != len(environment):
            raise MissionError(f"task {name!r} has a non-string environment value")
        return cls(
            name=name,
            command=command,
            cwd=cwd.resolve(),
            env=env,
            dependencies=_strings(body.get("depends-on"), field="depends-on", task=name),
            arguments=_strings(body.get("args"), field="args", task=name),
        )

    def invocation(self, argv: Sequence[str]) -> tuple[tuple[str, ...], dict[str, str]]:
        """Bind typed arguments and return plain executable argv plus the task environment."""
        if not self.arguments:
            command = self.command
            environment = self.env
            trailing = tuple(argv)
        else:
            try:
                separator = argv.index("--")
            except ValueError:
                values = tuple(argv)
                trailing = ()
            else:
                values = tuple(argv[:separator])
                trailing = tuple(argv[separator + 1 :])
            if len(values) != len(self.arguments):
                raise MissionError(
                    f"task {self.name!r} needs {len(self.arguments)} arguments "
                    f"({', '.join(self.arguments)}), got {len(values)}"
                )
            bindings = dict(zip(self.arguments, values, strict=True))
            command = self._render(self.command, bindings)
            environment = {key: self._render(value, bindings) for key, value in self.env.items()}
        self._refuse_task_shell(self.name, command)
        try:
            tokens = tuple(shlex.split(command, posix=True))
        except ValueError as error:
            raise MissionError(
                f"task {self.name!r} has invalid command quoting ({error})"
            ) from error
        if command and not tokens:
            raise MissionError(f"task {self.name!r} has an empty command")
        return (*tokens, *trailing), environment

    def _render(self, value: str, bindings: dict[str, str]) -> str:
        """Render the simple named argument templates Mainboard's task schema accepts."""

        def replace(match: re.Match[str]) -> str:
            try:
                return bindings[match.group(1)]
            except KeyError as error:
                raise MissionError(
                    f"task {self.name!r} refers to undeclared argument {match.group(1)!r}"
                ) from error

        rendered = _TEMPLATE.sub(replace, value)
        if "{{" in rendered or "}}" in rendered:
            raise MissionError(
                f"task {self.name!r} uses a template expression the restricted Windows "
                "runner cannot reproduce"
            )
        return rendered

    @staticmethod
    def _refuse_task_shell(name: str, command: str) -> None:
        """Reject syntax whose Deno task-shell meaning plain Windows argv cannot preserve."""
        message = (
            f"task {name!r} uses task-shell syntax unsupported by the restricted Windows "
            "runner; split shell chains into task dependencies or invoke one cross-platform "
            "executable"
        )
        quote = ""
        escaped = False
        for character in command:
            if escaped:
                escaped = False
                continue
            if character == "\\" and quote != "'":
                escaped = True
                continue
            if character in "'\"":
                if not quote:
                    quote = character
                elif quote == character:
                    quote = ""
                continue
            if (
                character in "\r\n"
                or (character in "$`" and quote != "'")
                or (not quote and character in "|&;<>()^*?")
            ):
                raise MissionError(message)


class WindowsTaskRunner:
    """Run a generated task graph without starting Pixi inside a restricted Windows app.

    Pixi still compiles, solves, installs, and captures the complete activation. This runner
    only replaces ``pixi run`` after installation, where Pixi 0.78 otherwise initializes its
    authentication store before launching a child and fails to discover the sandboxed profile.
    """

    def __init__(self, manifest: Path, environment: str) -> None:
        self.manifest = manifest
        self.environment = environment
        self.tasks = self._tasks()
        self.initial_cwd = Path.cwd()

    def run(
        self,
        command: Sequence[str],
        action: Callable[[BaseCommand], CommandResult],
    ) -> CommandResult:
        """Run dependencies then the named task, preserving output and the first failure code."""
        if not command or command[0] not in self.tasks:
            raise MissionError("restricted Windows task runner received no declared task")
        completed: set[str] = set()
        visiting: list[str] = []
        results: list[CommandResult] = []
        failure = self._run_task(
            command[0], tuple(command[1:]), action, completed, visiting, results
        )
        return CommandResult(
            failure.returncode if failure else 0,
            "".join(result.stdout for result in results),
            "".join(result.stderr for result in results),
        )

    def _run_task(
        self,
        name: str,
        argv: tuple[str, ...],
        action: Callable[[BaseCommand], CommandResult],
        completed: set[str],
        visiting: list[str],
        results: list[CommandResult],
    ) -> CommandResult | None:
        """Depth-first ordered task execution with cycle detection and shared-dependency dedup."""
        if name in completed:
            return None
        if name in visiting:
            cycle = " -> ".join([*visiting[visiting.index(name) :], name])
            raise MissionError(f"task dependency cycle: {cycle}")
        try:
            value = self.tasks[name]
        except KeyError as error:
            raise MissionError(f"task dependency {name!r} is not declared") from error
        task = WindowsTask.parse(name, value, manifest=self.manifest)
        visiting.append(name)
        try:
            for dependency in task.dependencies:
                if failure := self._run_task(dependency, (), action, completed, visiting, results):
                    return failure
        finally:
            visiting.pop()
        invocation, environment = task.invocation(argv)
        if invocation:
            with (
                local.cwd(str(task.cwd)),
                local.env(INIT_CWD=str(self.initial_cwd), **environment),
            ):
                executable = local[invocation[0]][invocation[1:]]
                result = action(executable)
            results.append(result)
            if not result.succeeded:
                return result
        completed.add(name)
        return None

    def _tasks(self) -> dict[str, Json]:
        """Every root and selected-feature task active in this generated environment."""
        try:
            document = cast(
                "dict[str, Json]", tomllib.loads(self.manifest.read_text(encoding="utf-8"))
            )
        except FileNotFoundError as error:
            raise MissionError(
                f"generated Pixi manifest does not exist: {self.manifest}"
            ) from error
        selected = _table(document.get("environments", {}), field="environments", task="")
        environment_table = _table(
            selected.get(self.environment, {}),
            field="environment",
            task=self.environment,
        )
        active: dict[str, Json] = {}
        if not environment_table.get("no-default-feature", False):
            active.update(_table(document.get("tasks", {}), field="tasks", task=""))
        features = _table(document.get("feature", {}), field="feature", task="")
        for feature_name in _strings(
            environment_table.get("features"), field="features", task=self.environment
        ):
            feature = _table(features.get(feature_name), field="feature", task=feature_name)
            active.update(_table(feature.get("tasks", {}), field="tasks", task=feature_name))
        return active
