import platform
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import MissionError
from mainboard.engines.compile.backend import CommandResult, Process
from mainboard.engines.compile.backend.windows_task import WindowsTask, WindowsTaskRunner

if TYPE_CHECKING:
    from collections.abc import Callable

    from plumbum.commands.base import BaseCommand


def test_the_windows_runner_preserves_the_task_graph_cwd_environment_and_arguments(
    tmp_path: Path,
    stub_binary: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback runs shared dependencies once, then binds the selected feature task."""
    manifest = tmp_path / ".mainboard" / "envs" / "default" / "pixi.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        """[environments.default]
features = ["checks"]

[tasks.prepare]
cmd = "worker prepare.py"
cwd = "../../.."
env = { PHASE = "prepare" }

[tasks.shared]
cmd = "worker shared.py"
cwd = "../../.."
depends-on = ["prepare"]

[feature.checks.tasks.check]
cmd = "worker check.py {{ suite }}"
cwd = "../../../packages/mainboard"
env = { PHASE = "{{ suite }}" }
depends-on = ["prepare", "shared"]
args = ["suite"]
""",
        encoding="utf-8",
    )
    worker = stub_binary("worker.exe")
    monkeypatch.delenv("PHASE", raising=False)
    tmp_path.joinpath("packages", "mainboard").mkdir(parents=True)
    observed: list[tuple[list[str], Path, str, str]] = []

    def execute(command: BaseCommand) -> CommandResult:
        observed.append(
            (
                list(command.formulate()),
                Path.cwd(),
                str(local.env.get("PHASE", "")),
                str(local.env["INIT_CWD"]),
            )
        )
        return CommandResult(0, f"step-{len(observed)}\n", "")

    initial = Path.cwd()
    result = WindowsTaskRunner(manifest, "default").run(("check", "unit", "--", "--fix"), execute)

    assert result == CommandResult(0, "step-1\nstep-2\nstep-3\n", "")
    assert [item[0] for item in observed] == [
        [worker, "prepare.py"],
        [worker, "shared.py"],
        [worker, "check.py", "unit", "--fix"],
    ]
    assert [item[1] for item in observed] == [
        tmp_path,
        tmp_path,
        tmp_path / "packages" / "mainboard",
    ]
    assert [item[2] for item in observed] == ["prepare", "", "unit"]
    assert {item[3] for item in observed} == {str(initial)}


def test_the_windows_runner_stops_on_the_dependency_exit_code(
    tmp_path: Path,
    stub_binary: Callable[[str], str],
) -> None:
    """A failed prerequisite prevents the dependent command and retains its exact code."""
    manifest = tmp_path / "pixi.toml"
    manifest.write_text(
        """[tasks.prepare]
cmd = "prepare"
[tasks.check]
cmd = "check"
depends-on = ["prepare"]
""",
        encoding="utf-8",
    )
    stub_binary("prepare.exe")
    seen: list[str] = []

    def execute(command: BaseCommand) -> CommandResult:
        seen.append(Path(list(command.formulate())[0]).stem)
        return CommandResult(17, "partial\n", "failed\n")

    result = WindowsTaskRunner(manifest, "default").run(("check",), execute)

    assert result == CommandResult(17, "partial\n", "failed\n")
    assert seen == ["prepare"]


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        pytest.param(
            '[tasks.a]\ndepends-on = ["b"]\n[tasks.b]\ndepends-on = ["a"]\n',
            "task dependency cycle: a -> b -> a",
            id="cycle",
        ),
        pytest.param(
            '[tasks.a]\ndepends-on = ["missing"]\n',
            "task dependency 'missing' is not declared",
            id="missing-dependency",
        ),
    ],
)
def test_the_windows_runner_refuses_an_invalid_dependency_graph(
    tasks: str,
    message: str,
    tmp_path: Path,
) -> None:
    """An invalid graph fails before a command can be reported as successful."""
    manifest = tmp_path / "pixi.toml"
    manifest.write_text(tasks, encoding="utf-8")

    with pytest.raises(MissionError, match=message):
        WindowsTaskRunner(manifest, "default").run(
            ("a",), lambda command: CommandResult(0, "", "")
        )


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("python build.py && python check.py", id="chain"),
        pytest.param("echo $PHASE", id="posix-variable"),
        pytest.param("ruff check *.py", id="shell-glob"),
        pytest.param("echo ready > result.txt", id="output-redirection"),
        pytest.param("echo first\rsecond", id="carriage-return"),
    ],
)
def test_the_windows_runner_refuses_task_shell_syntax(
    command: str,
    tmp_path: Path,
) -> None:
    """Direct argv execution never silently changes Pixi's Deno task-shell grammar."""
    task = WindowsTask.parse("check", {"cmd": command}, manifest=tmp_path / "pixi.toml")

    with pytest.raises(MissionError, match="uses task-shell syntax unsupported"):
        task.invocation(())


@pytest.mark.skipif(platform.system() != "Windows", reason="restricted Windows runner")
def test_the_windows_runner_executes_a_real_dependency_in_its_declared_directory(
    tmp_path: Path,
) -> None:
    """The restricted Windows path executes both real commands with graph, cwd, and env intact."""
    manifest = tmp_path / ".mainboard" / "envs" / "default" / "pixi.toml"
    manifest.parent.mkdir(parents=True)
    tmp_path.joinpath("work").mkdir()
    manifest.write_text(
        "[tasks.prepare]\n"
        "cmd = '''python -c \"from pathlib import Path; "
        "Path('dependency.txt').write_text('ready')\"'''\n"
        'cwd = "../../.."\n\n'
        "[tasks.check]\n"
        "cmd = '''python -c \"import os; from pathlib import Path; "
        "Path('result.txt').write_text(Path('../dependency.txt').read_text() + ':' + "
        "os.environ['PHASE'])\"'''\n"
        'cwd = "../../../work"\n'
        'env = { PHASE = "check" }\n'
        'depends-on = ["prepare"]\n',
        encoding="utf-8",
    )

    result = WindowsTaskRunner(manifest, "default").run(("check",), Process.capture)

    assert result.succeeded
    assert tmp_path.joinpath("work", "result.txt").read_text(encoding="utf-8") == "ready:check"


def test_a_commandless_aggregator_runs_a_shorthand_string_dependency_once(
    tmp_path: Path,
    stub_binary: Callable[[str], str],
) -> None:
    """Shorthand tasks and a string dependency keep Pixi's commandless aggregator behavior."""
    manifest = tmp_path / "pixi.toml"
    manifest.write_text(
        '[tasks]\nprepare = "prepare"\n[tasks.all]\ndepends-on = "prepare"\n',
        encoding="utf-8",
    )
    prepare = stub_binary("prepare.exe")
    observed: list[list[str]] = []

    result = WindowsTaskRunner(manifest, "default").run(
        ("all",),
        lambda command: (
            observed.append(list(command.formulate())) or CommandResult(0, "ready\n", "")
        ),
    )

    assert result == CommandResult(0, "ready\n", "")
    assert observed == [[prepare]]


def test_typed_arguments_work_without_a_trailing_separator(tmp_path: Path) -> None:
    """Required typed values bind directly and malformed calls fail before execution."""
    task = WindowsTask.parse(
        "check",
        {"cmd": "check {{ suite }}", "args": ["suite"], "cwd": str(tmp_path)},
        manifest=tmp_path / "pixi.toml",
    )

    assert task.invocation(("unit",)) == (("check", "unit"), {})
    with pytest.raises(MissionError, match="needs 1 arguments"):
        task.invocation(())

    quoted = WindowsTask.parse(
        "markers",
        {"cmd": "pytest -m 'not slow'"},
        manifest=tmp_path / "pixi.toml",
    )
    assert quoted.invocation(()) == (("pytest", "-m", "not slow"), {})

    escaped = WindowsTask.parse(
        "python",
        {"cmd": r'python -c "print(\"ok\")"'},
        manifest=tmp_path / "pixi.toml",
    )
    assert escaped.invocation(()) == (("python", "-c", 'print("ok")'), {})

    nested = WindowsTask.parse(
        "nested",
        {"cmd": '''python -c "print('ok')"'''},
        manifest=tmp_path / "pixi.toml",
    )
    assert nested.invocation(()) == (("python", "-c", "print('ok')"), {})

    empty = WindowsTask.parse(
        "empty",
        {"cmd": "   "},
        manifest=tmp_path / "pixi.toml",
    )
    with pytest.raises(MissionError, match="has an empty command"):
        empty.invocation(())


@pytest.mark.parametrize(
    ("command", "message"),
    [
        pytest.param("check {{ missing }}", "refers to undeclared argument", id="unknown-name"),
        pytest.param(
            "check {{ suite | upper }}",
            "uses a template expression",
            id="unsupported-expression",
        ),
        pytest.param('python -c "print(1)', "invalid command quoting", id="unclosed-double-quote"),
    ],
)
def test_typed_arguments_refuse_templates_the_fallback_cannot_bind(
    command: str,
    message: str,
    tmp_path: Path,
) -> None:
    """Template failures name the unsupported contract instead of changing the command."""
    task = WindowsTask.parse(
        "check",
        {"cmd": command, "args": ["suite"]},
        manifest=tmp_path / "pixi.toml",
    )

    with pytest.raises(MissionError, match=message):
        task.invocation(("unit",))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param("[tasks]\ncheck = 1\n", "non-table 'definition'", id="definition"),
        pytest.param("[tasks.check]\ncmd = 1\n", "non-string 'cmd'", id="command"),
        pytest.param('[[tasks.check]]\ncmd = "check"\n', "non-table 'definition'", id="array"),
        pytest.param('[tasks.check]\ncmd = "check"\ncwd = 1\n', "non-string 'cwd'", id="cwd"),
        pytest.param(
            '[tasks.check]\ncmd = "check"\nenv = { PHASE = 1 }\n',
            "non-string environment",
            id="environment-value",
        ),
        pytest.param(
            '[tasks.check]\ncmd = "check"\ndepends-on = 1\n',
            "non-string 'depends-on'",
            id="dependencies",
        ),
        pytest.param(
            '[tasks.check]\ncmd = "check"\nargs = ["suite", 1]\n',
            "non-string 'args'",
            id="arguments",
        ),
        pytest.param('tasks = "check"\n', "non-table 'tasks'", id="tasks-table"),
        pytest.param(
            '[environments]\ndefault = "check"\n[tasks]\ncheck = "check"\n',
            "non-table 'environment'",
            id="environment-table",
        ),
    ],
)
def test_the_windows_runner_refuses_malformed_generated_task_fields(
    body: str,
    message: str,
    tmp_path: Path,
) -> None:
    """A corrupt generated manifest fails at its precise task field."""
    manifest = tmp_path / "pixi.toml"
    manifest.write_text(body, encoding="utf-8")

    with pytest.raises(MissionError, match=message):
        WindowsTaskRunner(manifest, "default").run(
            ("check",), lambda command: CommandResult(0, "", "")
        )


def test_an_isolated_environment_excludes_root_tasks(tmp_path: Path) -> None:
    """A no-default feature exposes only its own declared task set."""
    manifest = tmp_path / "pixi.toml"
    manifest.write_text(
        """[environments.isolated]
features = ["isolated"]
no-default-feature = true
[tasks]
root = "root"
[feature.isolated.tasks]
check = "check"
""",
        encoding="utf-8",
    )

    runner = WindowsTaskRunner(manifest, "isolated")

    assert set(runner.tasks) == {"check"}
    with pytest.raises(MissionError, match="received no declared task"):
        runner.run(("root",), lambda command: CommandResult(0, "", ""))


def test_the_windows_runner_requires_a_generated_manifest(tmp_path: Path) -> None:
    """The restricted path reports a missing generated manifest as a Mainboard error."""
    with pytest.raises(MissionError, match="generated Pixi manifest does not exist"):
        WindowsTaskRunner(tmp_path / "missing.toml", "default")
