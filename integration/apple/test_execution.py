import json
import string
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import JsonValue

from mainboard import MissionError
from mainboard.apple import Apple, XcodeRun
from mainboard.proc import Processes

from .builders import devices_reply, succeeded_build
from .constants import SIMCTL_LIST, XCRUN
from .fakes import ProcessLog, ResultTool

_DESTINATION = "platform=iOS Simulator,name=Example,OS=27.0"
_SIMULATORS = devices_reply(
    {
        "com.apple.CoreSimulator.SimRuntime.iOS-27-0": [
            {"name": "Example", "udid": "chosen", "state": "Booted", "isAvailable": True}
        ],
        "com.apple.CoreSimulator.SimRuntime.iOS-26-0": [
            {"name": "Example", "udid": "twin", "state": "Shutdown", "isAvailable": True}
        ],
    }
)
_LITERAL = st.text(alphabet=string.ascii_letters + string.digits + " '$()日本=:/_", max_size=12)
_RUNS = st.builds(
    XcodeRun,
    scheme=_LITERAL.filter(str.strip),
    action=st.sampled_from(["build", "test", "build-for-testing", "resolve"]),
    destination=_LITERAL,
    configuration=_LITERAL,
    derived_data=st.sampled_from([None, Path("/derived data")]),
    only_testing=st.lists(_LITERAL, max_size=3),
    skip_testing=st.lists(_LITERAL, max_size=3),
    setting=st.lists(_LITERAL, max_size=3),
    timeout=st.floats(min_value=1, max_value=1e4),
)
_URL_NEEDS_A_TEST = "URL delivery requires the test action"
_INVALID_CONFIGURATION = "invalid URL test configuration"
_ONE_SIMULATOR = "one exact simulator destination"
_LOCAL_ROUTES = "local custom-scheme paths"
_CONTAINED = "inside the runner data container"
_APP = {"scheme": "App"}
_PACKAGE_FLAGS = ["-onlyUsePackageVersionsFromResolvedFile", "-hideShellScriptEnvironment"]
_PASSED = {"build": True, "tests": True}
_FAILED = {"build": True, "tests": False}

type Reports = dict[str, dict[str, JsonValue]]


def _config(**fields: str | list[str]) -> str:
    """A URL test configuration in JSON, `fields` replacing its sound defaults."""
    sound = {
        "runner_bundle": "example.runner",
        "marker": "tmp/request.json",
        "routes": ["fixture:///reviews"],
    }
    return json.dumps({**sound, **fields})


def _project(root: Path, kind: str) -> Path:
    """A workspace, project or Swift package under `root`, named with shell metacharacters."""
    path = root / ("Package" if kind == "package" else f"App '$literal.{kind}")
    path.mkdir()
    if kind == "package":
        (path / "Package.swift").write_text("// swift-tools-version: 6.4\n")
    return path


def _leaves_a_bundle(command: Sequence[str]) -> None:
    """Make the result bundle xcodebuild was asked to write."""
    Path(command[command.index("-resultBundlePath") + 1]).mkdir()


class UrlHost:
    """A project, a simulator inventory and the process log a URL-delivered test runs against."""

    def __init__(self, project: Path, processes: ProcessLog) -> None:
        self.project = project
        self.processes = processes
        self.tool = ResultTool(reply=_SIMULATORS)
        self.descriptors: list[dict[str, str | list[str]]] = []
        processes.during = self._kept

    def _kept(self, command: Sequence[str]) -> None:
        """Keep the request the helper is handed, which is removed once the helper exits."""
        self.descriptors.append(json.loads(Path(command[3]).read_text(encoding="utf-8")))

    def execute(
        self,
        *,
        config: str | None = _config(),
        destination: str = _DESTINATION,
        only_testing: tuple[str, ...] = (),
        timeout: float = 900,
    ) -> int:
        """Run a URL-delivered test; a `config` of None leaves no configuration file."""
        configuration = self.project.parent / "url.json"
        if config is not None:
            configuration.write_text(config)
        run = XcodeRun(
            scheme="App",
            action="test",
            destination=destination,
            url_test=configuration,
            only_testing=only_testing,
            timeout=timeout,
        )
        return Apple(self.tool).execute(self.project, run, self.project.parent / "test.xcresult")


@pytest.fixture
def xcode_folder(tmp_path: Path) -> Path:
    """A folder holding an Xcode project and an old result bundle."""
    (tmp_path / "App.xcodeproj").mkdir()
    (tmp_path / "old.xcresult").mkdir()
    return tmp_path


@pytest.fixture
def host(project: Path, processes: ProcessLog) -> UrlHost:
    return UrlHost(project, processes)


@pytest.fixture
def execute_verb(
    cli: Callable[..., tuple[int, str]], project: Path, processes: ProcessLog
) -> Callable[..., tuple[int, Reports]]:
    """Run `mb apple execute` over a project; answers its exit code and the JSON reports it kept.

    The reports are keyed by kind, `build` or `tests`. `native` is Xcode's exit code and
    `evidence` whether Xcode leaves a result bundle behind.
    """

    def execute(
        *,
        action: str = "test",
        native: int = 0,
        result: str = "Passed",
        only: Sequence[str] = (),
        evidence: bool = True,
    ) -> tuple[int, Reports]:
        out = project.parent / "out"
        processes.exits = iter([native])
        if evidence:
            processes.during = _leaves_a_bundle
        selected = [argument for name in only for argument in ("--only-testing", name)]
        argv = ["execute", str(project), "--scheme", "App", "--action", action, *selected]
        code, _ = cli(Apple(ResultTool(summary={"result": result})), *argv, "--results", str(out))
        files = sorted(out.glob("*.json"))
        kept = {path.name.split(".")[-2]: json.loads(path.read_text()) for path in files}
        assert len(kept) == len(files), "one report of each kind"
        return code, kept

    return execute


@pytest.mark.parametrize(
    "action", [pytest.param("resolve", id="resolving"), pytest.param("test", id="testing")]
)
@pytest.mark.parametrize(
    ("kind", "source"),
    [
        pytest.param("xcworkspace", ["-workspace", "{project}"], id="a-workspace"),
        pytest.param("xcodeproj", ["-project", "{project}"], id="a-project"),
        pytest.param("package", [], id="a-swift-package"),
    ],
)
def test_xcodebuild_runs_beside_its_project_and_answers_the_native_exit(
    tmp_path: Path,
    processes: ProcessLog,
    kind: str,
    source: Sequence[str],
    action: Literal["resolve", "test"],
) -> None:
    project = _project(tmp_path, kind)
    bundle = tmp_path / "results" / "fresh.xcresult"
    processes.exits = iter([65])
    code = Apple().execute(project, XcodeRun(scheme="An app", action=action, timeout=123), bundle)
    [command] = processes.commands
    assert (code, processes.seconds) == (65, [123])
    assert processes.directories == [project if kind == "package" else project.parent]
    named = [part.format(project=project) for part in source]
    assert command[3 : 5 + len(source)] == [*named, "-scheme", "An app"]
    assert bundle.parent.is_dir() is (action != "resolve") and not bundle.exists()


@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(run=_RUNS, source=st.lists(_LITERAL, max_size=2))
def test_the_xcodebuild_command_keeps_every_input_literal_and_in_its_place(
    tmp_path: Path, run: XcodeRun, source: Sequence[str]
) -> None:
    bundle = tmp_path / "results" / "fresh.xcresult"
    command = run.command(source, bundle)
    pairs = set(zip(command, command[1:], strict=False))
    tail = [
        *(f"-only-testing:{name}" for name in run.only_testing),
        *(f"-skip-testing:{name}" for name in run.skip_testing),
        *run.setting,
    ]
    wanted = {
        ("-resultBundlePath", str(bundle)): not run.resolving,
        ("-configuration", run.configuration): not run.resolving,
        ("-parallel-testing-enabled", "NO"): run.action == "test",
        ("-destination", run.destination): bool(run.destination) and not run.resolving,
        ("-derivedDataPath", "/derived data"): run.derived_data is not None,
    }
    assert command[: 7 + len(source)] == [
        XCRUN,
        "xcodebuild",
        "-resolvePackageDependencies" if run.resolving else run.action,
        *source,
        "-scheme",
        run.scheme,
        *_PACKAGE_FLAGS,
    ]
    assert command[len(command) - len(tail) :] == tail
    assert command.count("-hideShellScriptEnvironment") == 1
    assert all((pair in pairs) is present for pair, present in wanted.items())


@pytest.mark.parametrize(
    ("target", "run", "bundle", "match"),
    [
        pytest.param(
            "App.xcodeproj",
            _APP,
            "old.xcresult",
            "overwrite result evidence",
            id="an-existing-result-bundle",
        ),
        pytest.param(
            "App.xcodeproj", _APP, None, "fresh result bundle path", id="no-result-bundle-path"
        ),
        pytest.param(
            "App.xcodeproj",
            {**_APP, "timeout": 0},
            "fresh.xcresult",
            "positive timeout",
            id="a-zero-timeout",
        ),
        pytest.param(
            "App.xcodeproj", {"scheme": " "}, "fresh.xcresult", "a scheme", id="a-blank-scheme"
        ),
        pytest.param(
            "App.xcodeproj",
            {**_APP, "action": "build", "url_test": Path("url.json")},
            "fresh.xcresult",
            _URL_NEEDS_A_TEST,
            id="url-delivery-outside-a-test",
        ),
        pytest.param(
            "missing", _APP, "fresh.xcresult", "no Apple project", id="a-missing-project"
        ),
        pytest.param(
            ".",
            _APP,
            "fresh.xcresult",
            "expected an .xcworkspace",
            id="a-folder-that-is-no-project",
        ),
    ],
)
def test_execution_refuses_old_evidence_and_invalid_inputs(
    xcode_folder: Path, processes: ProcessLog, target: str, run, bundle: str | None, match: str
) -> None:
    with pytest.raises(MissionError, match=match):
        Apple().execute(
            xcode_folder / target,
            XcodeRun(**run),
            xcode_folder / bundle if bundle else None,
        )
    assert processes.commands == []


@pytest.mark.parametrize(
    ("destination", "udid"),
    [
        pytest.param(_DESTINATION, "chosen", id="a-name-and-runtime"),
        pytest.param("platform=iOS Simulator,id=twin", "twin", id="a-udid"),
        pytest.param(
            "platform=iOS Simulator,name=Example,OS=26.0", "twin", id="the-other-runtime"
        ),
    ],
)
def test_url_delivery_hands_the_exact_simulator_and_command_to_the_swift_helper(
    host: UrlHost, destination: str, udid: str
) -> None:
    host.processes.exits = iter([65])
    code = host.execute(destination=destination, only_testing=("Suite/Case",), timeout=12)
    [command], [descriptor] = host.processes.commands, host.descriptors
    arguments = descriptor.pop("command")
    assert (code, host.processes.seconds) == (65, [12])
    assert host.processes.directories == [host.project.parent]
    assert command[:2] == [XCRUN, "swift"] and Path(command[2]).name == "url_tests.swift"
    assert descriptor == json.loads(_config()) | {"device": udid}
    assert (
        arguments[:3] == [XCRUN, "xcodebuild", "test"] and "-only-testing:Suite/Case" in arguments
    )
    assert arguments.count("-hideShellScriptEnvironment") == 1 and not Path(command[3]).exists()


@pytest.mark.parametrize(
    ("config", "destination", "reads", "match"),
    [
        pytest.param(
            _config(marker="../escape"),
            _DESTINATION,
            0,
            _CONTAINED,
            id="a-marker-leaving-the-container",
        ),
        pytest.param(
            _config(marker="/absolute"), _DESTINATION, 0, _CONTAINED, id="an-absolute-marker"
        ),
        pytest.param(
            _config(routes=["fixture:///reviews?credential=x"]),
            _DESTINATION,
            0,
            _LOCAL_ROUTES,
            id="a-route-with-a-query",
        ),
        pytest.param(
            _config(routes=["fixture://host/reviews"]),
            _DESTINATION,
            0,
            _LOCAL_ROUTES,
            id="a-route-with-a-host",
        ),
        pytest.param(
            _config(routes=["fixture:///reviews", "fixture:///reviews"]),
            _DESTINATION,
            0,
            "must be unique",
            id="a-repeated-route",
        ),
        pytest.param(_config(routes=[]), _DESTINATION, 0, _INVALID_CONFIGURATION, id="no-route"),
        pytest.param(
            "not JSON",
            _DESTINATION,
            0,
            _INVALID_CONFIGURATION,
            id="a-configuration-that-is-no-json",
        ),
        pytest.param(
            None, _DESTINATION, 0, _INVALID_CONFIGURATION, id="a-missing-configuration-file"
        ),
        pytest.param(
            _config(),
            "platform=macOS",
            0,
            "only supported on an iOS simulator",
            id="a-destination-that-is-no-simulator",
        ),
        pytest.param(
            _config(),
            "platform=iOS Simulator,name=Other,OS=27.0",
            1,
            _ONE_SIMULATOR,
            id="a-destination-naming-no-simulator",
        ),
        pytest.param(
            _config(),
            "platform=iOS Simulator,name=Example",
            1,
            _ONE_SIMULATOR,
            id="a-destination-naming-two-simulators",
        ),
    ],
)
def test_url_delivery_refuses_what_is_unsound_before_any_process_runs(
    host: UrlHost, config: str | None, destination: str, reads: int, match: str
) -> None:
    with pytest.raises(MissionError, match=match):
        host.execute(config=config, destination=destination)
    assert host.tool.calls == [SIMCTL_LIST] * reads
    assert host.processes.commands == []


@pytest.mark.parametrize(
    ("arrange", "expected"),
    [
        pytest.param({}, (0, _PASSED, []), id="a-passing-run-exits-zero"),
        pytest.param(
            {"result": "Failed"},
            (1, _FAILED, []),
            id="failed-tests-exit-one-despite-a-clean-native-exit",
        ),
        pytest.param({"native": 65}, (65, _PASSED, []), id="the-native-failure-wins"),
        pytest.param(
            {"only": ["Tests/Suite/landscape", "Tests/Wrong/landscape", "Tests/Suite/missing"]},
            (1, _FAILED, ["Tests/Wrong/landscape", "Tests/Suite/missing"]),
            id="selectors-xcode-ignored-fail-a-passing-run",
        ),
        pytest.param(
            {"evidence": False},
            (1, {}, []),
            id="a-run-that-left-no-bundle-fails-despite-a-clean-native-exit",
        ),
        pytest.param(
            {"action": "resolve", "evidence": False},
            (0, {}, []),
            id="resolving-packages-keeps-no-evidence",
        ),
    ],
)
def test_the_execute_verb_exits_with_the_native_code_or_the_actual_verdict_and_keeps_its_reports(
    execute_verb: Callable[..., tuple[int, Reports]], arrange, expected
) -> None:
    code, kept = execute_verb(**arrange)
    verdicts = {
        kind: doc.get("successful", doc.get("build_successful")) for kind, doc in kept.items()
    }
    missing = kept.get("tests", {}).get("missing_selectors", [])
    assert (code, verdicts, missing) == expected
    assert kept.get("build", {"summary": succeeded_build()})["summary"] == succeeded_build()
    assert len({doc["bundle"] for doc in kept.values()}) <= 1


@pytest.mark.parametrize(
    ("read", "suffix"),
    [
        pytest.param(Apple.diagnostics, ".build.json", id="build-evidence"),
        pytest.param(Apple.results, ".tests.json", id="test-evidence"),
    ],
)
def test_retained_evidence_sits_beside_its_bundle_and_is_never_overwritten(
    bundle: Path, read, suffix: str
) -> None:
    report = read(Apple(ResultTool()), bundle)
    kept = Apple.retain(report)
    assert kept == bundle.with_suffix(suffix)
    assert json.loads(kept.read_text())["bundle"] == str(report.bundle)
    with pytest.raises(MissionError, match="refusing to overwrite Apple evidence"):
        Apple.retain(report)


def test_process_timeout_runs_package_commands_in_the_requested_directory(tmp_path: Path) -> None:
    code = Processes().timeout(
        10,
        [sys.executable, "-c", "from pathlib import Path; Path('receipt').write_text('ok')"],
        cwd=tmp_path,
    )
    assert code == 0 and (tmp_path / "receipt").read_text() == "ok"
