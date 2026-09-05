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

# What a declared placeholder is masked with while the command is split. A placeholder is written
# with spaces inside its braces, and the split has to happen before any value is bound, so each
# one becomes this stand-in first: it carries no whitespace, no quote and no escape, so `shlex`
# moves it through as part of whichever token it was written in and the value that replaces it
# afterwards is exactly one argv element however many spaces or backslashes it holds.
_SLOT = "\x00mainboard-argument-{index}\x00"


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
        """Bind typed arguments and return plain executable argv plus the task environment.

        The declared command is what is vetted and what is split, both before a single value is
        bound, and each resulting token is rendered on its own afterwards. That order is the
        whole contract: one binding is exactly one argv element.

        Rendering first and splitting the result broke it twice over. A value with a space in it
        (`--filter {{ pattern }}` bound to `not slow`) became two arguments, and a Windows path
        lost its backslashes to the shell-quoting rules of a split that was never meant to read
        user data. Vetting the rendered string was the same mistake from the other side: an
        ordinary value carrying `*`, `&` or a glob failed the task for task-shell syntax the
        manifest never contained.
        """
        bindings, trailing = self._bound(argv)
        masked, slots = self._masked()
        if "{{" in masked or "}}" in masked:
            raise MissionError(
                f"task {self.name!r} uses a template expression the restricted Windows "
                "runner cannot reproduce"
            )
        self._refuse_task_shell(self.name, masked)
        try:
            declared = tuple(shlex.split(masked, posix=True))
        except ValueError as error:
            raise MissionError(
                f"task {self.name!r} has invalid command quoting ({error})"
            ) from error
        if self.command and not declared:
            raise MissionError(f"task {self.name!r} has an empty command")
        tokens = tuple(self._filled(token, slots, bindings) for token in declared)
        environment = {key: self._render(value, bindings) for key, value in self.env.items()}
        return (*tokens, *trailing), environment

    def _masked(self) -> tuple[str, dict[str, str]]:
        """The declared command with every placeholder replaced by a stand-in, and what each was.

        The one preparation the split needs: `{{ suite }}` is four shlex tokens and its
        stand-in is one, so masking is what lets the quoting rules apply to the manifest's own
        text while the value bound into it is never split at all.
        """
        slots: dict[str, str] = {}

        def mask(match: re.Match[str]) -> str:
            slot = _SLOT.format(index=len(slots))
            slots[slot] = match.group(1)
            return slot

        return _TEMPLATE.sub(mask, self.command), slots

    def _filled(self, token: str, slots: dict[str, str], bindings: dict[str, str]) -> str:
        """One split token with its stand-ins replaced by the values bound to them.

        token: a token of the split, masked command.
        slots: every stand-in in that command and the argument it stands for.
        bindings: the values the caller bound to the declared arguments.
        """
        for slot, argument in slots.items():
            if slot not in token:
                continue
            try:
                token = token.replace(slot, bindings[argument])
            except KeyError:
                raise MissionError(
                    f"task {self.name!r} refers to undeclared argument {argument!r}"
                ) from None
        return token

    def _bound(self, argv: Sequence[str]) -> tuple[dict[str, str], tuple[str, ...]]:
        """`argv` split into this task's declared argument values and whatever trails them.

        A task that declares no arguments binds nothing and forwards every token, which is what
        `run <task> extra` has always meant for a task with no typed surface.

        argv: the tokens the caller typed after the task name.
        """
        if not self.arguments:
            return {}, tuple(argv)
        try:
            separator = argv.index("--")
        except ValueError:
            values, trailing = tuple(argv), ()
        else:
            values, trailing = tuple(argv[:separator]), tuple(argv[separator + 1 :])
        if len(values) != len(self.arguments):
            raise MissionError(
                f"task {self.name!r} needs {len(self.arguments)} arguments "
                f"({', '.join(self.arguments)}), got {len(values)}"
            )
        return dict(zip(self.arguments, values, strict=True)), trailing

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
        """Reject syntax whose Deno task-shell meaning plain Windows argv cannot preserve.

        Only ever asked about the declared command. What a caller binds into it is data, and a
        value holding a glob or an ampersand is an argument rather than a chain, so vetting it
        would refuse the manifest for something the manifest does not say.
        """
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
