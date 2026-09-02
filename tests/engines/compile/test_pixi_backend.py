import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import MissionError
from mainboard.engines.compile.backend import CommandResult, Process

if TYPE_CHECKING:
    from collections.abc import Callable

    from plumbum.commands.base import BaseCommand
    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi

    from .support import Record

_FINGERPRINT = ".pixi-environment-fingerprint"
_DAMAGED = "cupy-cuda13x"
_ROOT = "cupy_cuda13x"


@pytest.fixture
def installed(pixi: Pixi) -> Path:
    """An installation of `default` pixi finished and stamped, returning its site-packages."""
    fingerprint = pixi.env_prefix("default") / "conda-meta" / _FINGERPRINT
    fingerprint.parent.mkdir(parents=True)
    fingerprint.write_text("installed\n")
    site_packages = pixi.env_prefix("default") / "lib" / "python3.14" / "site-packages"
    site_packages.mkdir(parents=True)
    pixi.lock.write_text("version: 7\n")
    return site_packages


def test_install_requires_a_lock_unless_resolution_was_requested(
    fp: FakeProcess, pixi: Pixi
) -> None:
    """A missing generated lock never turns an ordinary install into an implicit solve."""
    with pytest.raises(MissionError, match=r"pixi.lock is missing.*install --resolve"):
        pixi.install("default")
    assert not fp.calls


@pytest.mark.parametrize(
    ("generated", "rule"),
    [
        pytest.param(None, "--locked", id="nothing-compiled-yet"),
        pytest.param('[dependencies]\nnumpy = "*"\n', "--locked", id="no-source-at-all"),
        pytest.param(
            '[dependencies.demo]\npath = "../demo"\n', "--locked", id="a-path-that-is-not-editable"
        ),
        pytest.param(
            '[[workspace.extra]]\npath = "../demo"\neditable = true\n',
            "--frozen",
            id="an-editable-source-nested-inside-an-array",
        ),
    ],
)
def test_a_mutable_editable_source_is_what_relaxes_locked_into_frozen(
    generated: str | None, rule: str, fp: FakeProcess, pixi: Pixi
) -> None:
    """An editable's code may move without the lock moving.

    Demanding an unchanged tree would refuse every workspace that develops one of its own
    dependencies.
    """
    if generated is not None:
        pixi.manifest.write_text(generated)
    pixi.lock.write_text("version: 7\n")
    fp.register([fp.any()], stdout="environment ready\n")

    assert pixi.install("default") is None

    relaxed = "--frozen" if rule == "--locked" else "--locked"
    assert rule in list(fp.calls[0])
    assert relaxed not in list(fp.calls[0])


@pytest.mark.parametrize(
    ("stdout", "stderr", "message"),
    [
        pytest.param(
            "pixi solver context\n",
            "pixi solver failed\n",
            "pixi install",
            id="an-ordinary-failure-stays-an-installation-failure",
        ),
        pytest.param(
            "",
            "the lock file is not up-to-date with the workspace\n",
            r"manifest drifted.*install --resolve",
            id="a-stale-lock-is-refused-with-the-explicit-escape",
        ),
        pytest.param(
            "Pixi task (build): command\n",
            "task says lock file is not up-to-date\n",
            "pixi install",
            id="a-tasks-own-output-is-not-mislabeled-as-drift",
        ),
    ],
)
def test_a_failed_install_reaches_the_callers_output_before_it_is_diagnosed(
    stdout: str,
    stderr: str,
    message: str,
    fp: FakeProcess,
    pixi: Pixi,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both native streams are tee'd, and only pixi's own lock rejection reads as drift."""
    pixi.lock.write_text("version: 7\n")
    fp.register([fp.any()], returncode=17, stdout=stdout, stderr=stderr)

    with pytest.raises(MissionError, match=message):
        pixi.install("default")

    assert len(fp.calls) == 1
    assert "--locked" in list(fp.calls[0])
    assert capsys.readouterr() == (stdout, stderr)


@pytest.mark.parametrize(
    ("generated", "verification"),
    [
        pytest.param(
            '[workspace]\nplatforms = ["linux-64"]\n',
            "--locked",
            id="a-freshly-solved-lock-is-verified-locked",
        ),
        pytest.param(
            '[pypi-dependencies.demo]\npath = "../demo"\neditable = true\n',
            "--frozen",
            id="an-editable-source-is-verified-frozen",
        ),
    ],
)
def test_a_resolve_installs_once_more_to_verify_the_lock_it_just_solved(
    generated: str,
    verification: str,
    fp: FakeProcess,
    pixi: Pixi,
    tool_paths: Mapping[str, str],
) -> None:
    """A solve counts only once pixi's own check passes.

    The resulting pair must clear it, the known double-install wart carried over from chefe.
    """
    pixi.manifest.write_text(generated)
    pixi.lock.write_text("version: 7\n")
    for _ in range(2):
        fp.register([fp.any()], stdout="environment ready\n")

    pixi.install("serving", resolve=True)

    assert len(fp.calls) == 2
    assert list(fp.calls[0]) == [
        tool_paths["pixi"],
        "install",
        "--manifest-path",
        str(pixi.manifest),
        "-e",
        "serving",
    ]
    assert verification in list(fp.calls[1])


def test_install_repairs_a_wheel_damaged_underneath_pixi(
    fp: FakeProcess, pixi: Pixi, installed: Path, record: Record, tool_paths: Mapping[str, str]
) -> None:
    """A retained `dist-info` whose files vanished is reinstalled through the locked env."""
    root = record(installed, _DAMAGED, roots=f"{_ROOT}\ncupyx\n")
    scope = ["--manifest-path", str(pixi.manifest), "--locked", "-e", "default"]
    fp.register([tool_paths["pixi"], "install", *scope], stdout="environment ready\n")
    fp.register(
        [tool_paths["pixi"], "reinstall", *scope, _DAMAGED],
        stdout="package reinstalled\n",
        callback=lambda process: root.mkdir(),
    )

    pixi.install("default")

    assert len(fp.calls) == 2
    assert list(fp.calls[1]) == [tool_paths["pixi"], "reinstall", *scope, _DAMAGED]


@pytest.mark.parametrize(
    ("finished", "importable"),
    [
        pytest.param(True, True, id="every-package-still-imports"),
        pytest.param(False, False, id="an-interrupted-install-abandoned-the-prefix"),
    ],
)
def test_install_reinstalls_nothing_the_audit_has_no_business_touching(
    finished: bool,
    importable: bool,
    fp: FakeProcess,
    pixi: Pixi,
    installed: Path,
    record: Record,
) -> None:
    """A half-written prefix reads as damaged everywhere, so only a stamped install is audited."""
    root = record(installed, _DAMAGED, roots=f"{_ROOT}\n")
    if importable:
        root.mkdir()
    if not finished:
        (pixi.env_prefix("default") / "conda-meta" / _FINGERPRINT).unlink()
    fp.register([fp.any()], stdout="environment ready\n")

    pixi.install("default")

    assert len(fp.calls) == 1


@pytest.mark.parametrize(
    ("returncode", "message"),
    [
        pytest.param(9, "pixi reinstall", id="the-targeted-reinstall-failed-outright"),
        pytest.param(0, f"{_DAMAGED} stayed incomplete", id="the-reinstall-mended-nothing"),
    ],
)
def test_install_refuses_an_environment_the_repair_left_broken(
    returncode: int,
    message: str,
    fp: FakeProcess,
    pixi: Pixi,
    installed: Path,
    record: Record,
) -> None:
    """A reinstall reporting success cannot conceal a package that still imports nothing."""
    record(installed, _DAMAGED, roots=f"{_ROOT}\n")
    fp.register([fp.any()], stdout="environment ready\n")
    fp.register([fp.any()], returncode=returncode, stderr="reinstall said so\n")

    with pytest.raises(MissionError, match=message):
        pixi.install("default")


def test_readiness_trusts_the_installation_fingerprint_over_the_directory(
    pixi: Pixi, installed: Path
) -> None:
    """An interrupted install leaves a prefix behind, so only pixi's own stamp proves one ran."""
    assert pixi.ready("default") is True
    assert pixi.ready("serving") is False

    (pixi.env_prefix("default") / "conda-meta" / _FINGERPRINT).unlink()

    assert installed.is_dir()
    assert pixi.ready("default") is False


def test_the_backend_pins_every_call_to_the_workspace_it_owns(pixi: Pixi) -> None:
    """`--manifest-path` is injected into every verb, and the lock is the manifest's own pair."""
    assert pixi.scope() == ("--manifest-path", str(pixi.manifest))
    assert pixi.lock == pixi.manifest.with_suffix(".lock")
    assert pixi.env_prefix("serving") == pixi.manifest.parent / ".pixi" / "envs" / "serving"


def test_run_preserves_each_argument_through_pixis_cross_platform_runner(
    fp: FakeProcess,
    pixi: Pixi,
    tool_paths: Mapping[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    fp.register([fp.any()], returncode=7)
    assert pixi.run(("python", "-c", "raise SystemExit(7)"), "serving") == 7
    assert list(fp.calls[0]) == [
        tool_paths["pixi"],
        "run",
        "--manifest-path",
        str(pixi.manifest),
        "--frozen",
        "-e",
        "serving",
        "python",
        "-c",
        "raise SystemExit(7)",
    ]


def test_windows_runs_explicit_argv_from_the_prefix_with_the_real_home(
    pixi: Pixi,
    installed: Path,
    stub_binary: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restricted Windows host need not initialize Pixi auth just to start an installed tool."""
    pixi.manifest.write_text(
        """[environments.default]
features = []
[target.win.activation]
scripts = ["dotenv.bat", "unset.bat"]
[target.win.activation.env]
FROM_MANIFEST = "declared"
""",
        encoding="utf-8",
    )
    pixi.manifest.parent.joinpath("values.env").write_text(
        "FROM_DOTENV=loaded\nHOME=C:/wrong\nTO_CLEAR=wrong\n", encoding="utf-8"
    )
    pixi.manifest.parent.joinpath("dotenv.bat").write_text(
        "@echo off\n"
        "rem Generated by mainboard's Provisioner (workspace.dotenv = true).\n"
        'if not exist "values.env" goto :eof\n',
        encoding="utf-8",
    )
    pixi.manifest.parent.joinpath("unset.bat").write_text(
        "@echo off\nrem Generated by mainboard from the [env] table.\nset TO_CLEAR=\n",
        encoding="utf-8",
    )
    pixi.windows_activation_cache.write_text(
        json.dumps(
            {
                "environment_variables": {
                    "FROM_MANIFEST": "declared",
                    "Path": str(local.env["PATH"]),
                    "SSL_CERT_FILE": "C:/prefix/ssl/cacert.pem",
                },
                "activation_scripts": [
                    str(pixi.manifest.parent / "dotenv.bat"),
                    str(pixi.manifest.parent / "unset.bat"),
                ],
            }
        ),
        encoding="utf-8",
    )
    executable = stub_binary("tectonic")
    prefix_scripts = pixi.env_prefix("default") / "Scripts"
    prefix_scripts.mkdir(parents=True)
    observed: list[tuple[BaseCommand, dict[str, str]]] = []

    def passthrough(command: BaseCommand) -> int:
        observed.append(
            (
                command,
                {
                    name: str(local.env.get(name, ""))
                    for name in (
                        "FROM_DOTENV",
                        "FROM_MANIFEST",
                        "HOME",
                        "PATH",
                        "SSL_CERT_FILE",
                        "TEMP",
                        "TMP",
                        "TO_CLEAR",
                    )
                },
            )
        )
        return 0

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(Process, "passthrough", passthrough)
    monkeypatch.setenv("HOME", "C:/sandbox/profile")
    monkeypatch.setenv("TO_CLEAR", "inherited")

    assert pixi.run(("tectonic", "--help")) == 0
    assert installed.is_dir()
    command, environment = observed[0]
    assert Path(command.formulate()[0]) == Path(executable)
    assert command.formulate()[1:] == ["--help"]
    assert environment["HOME"] == str(Path.home())
    assert environment["FROM_DOTENV"] == "loaded"
    assert environment["FROM_MANIFEST"] == "declared"
    assert environment["SSL_CERT_FILE"] == "C:/prefix/ssl/cacert.pem"
    assert environment["TO_CLEAR"] == ""
    assert list(map(Path, environment["PATH"].split(os.pathsep)))[:2] == [
        pixi.env_prefix("default"),
        prefix_scripts,
    ]
    assert environment["TEMP"] == environment["TMP"]
    assert Path(environment["TEMP"]).parent == pixi.manifest.parent.parent
    assert not Path(environment["TEMP"]).exists()
    assert os.environ["HOME"] == "C:/sandbox/profile"


def test_windows_refuses_an_arbitrary_activation_script_instead_of_skipping_it(
    pixi: Pixi,
    installed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The auth workaround must never silently weaken a workspace's activation contract."""
    pixi.manifest.write_text(
        '[target.win.activation]\nscripts = ["custom.bat"]\n', encoding="utf-8"
    )
    script = pixi.manifest.parent / "custom.bat"
    script.write_text("set PROJECT_MODE=unsafe\n")
    pixi.windows_activation_cache.write_text(
        json.dumps({"environment_variables": {}, "activation_scripts": [str(script)]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("platform.system", lambda: "Windows")

    assert installed.is_dir()
    with pytest.raises(MissionError, match="cannot reproduce activation script"):
        pixi.run(("tectonic", "--help"))


def test_windows_explicit_argv_requires_a_finished_prefix(
    pixi: Pixi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bypassing Pixi cannot also bypass Pixi's proof that installation completed."""
    pixi.manifest.write_text("[workspace]\n", encoding="utf-8")
    pixi.windows_activation_cache.write_text(
        json.dumps({"environment_variables": {}, "activation_scripts": []}), encoding="utf-8"
    )
    monkeypatch.setattr("platform.system", lambda: "Windows")

    with pytest.raises(MissionError, match="environment 'default' is not installed"):
        pixi.run(("tectonic", "--help"))


def test_windows_runs_declared_tasks_without_starting_pixis_auth_store(
    fp: FakeProcess,
    pixi: Pixi,
    installed: Path,
    stub_binary: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared task uses cached activation before Pixi can fail its profile lookup."""
    pixi.manifest.write_text(
        """[environments.default]
features = ["paper"]
[feature.paper.tasks.paper]
cmd = "tectonic paper.tex"
cwd = "../../.."
""",
        encoding="utf-8",
    )
    pixi.windows_activation_cache.write_text(
        json.dumps({"environment_variables": {}, "activation_scripts": []}), encoding="utf-8"
    )
    monkeypatch.setattr("platform.system", lambda: "Windows")
    command = stub_binary("tectonic.exe")
    fp.register([fp.any()], returncode=5)

    assert installed.is_dir()
    assert pixi.run(("paper", "--keep-logs")) == 5
    assert list(fp.calls[0]) == [command, "paper.tex", "--keep-logs"]


def test_windows_captures_a_declared_task_without_starting_pixi(
    fp: FakeProcess,
    pixi: Pixi,
    installed: Path,
    stub_binary: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capture sibling uses the same direct task graph and retains child output."""
    pixi.manifest.write_text(
        '[tasks.probe]\ncmd = "probe --json"\ncwd = "../../.."\n', encoding="utf-8"
    )
    pixi.windows_activation_cache.write_text(
        json.dumps({"environment_variables": {}, "activation_scripts": []}), encoding="utf-8"
    )
    monkeypatch.setattr("platform.system", lambda: "Windows")
    command = stub_binary("probe.exe")
    fp.register([fp.any()], stdout='{"ready": true}\n')

    assert installed.is_dir()
    result = pixi.capture(("probe",))

    assert result.succeeded
    assert result.stdout == '{"ready": true}\n'
    assert list(fp.calls[0]) == [command, "--json"]


def test_windows_capture_shares_one_timeout_across_the_task_graph(
    pixi: Pixi,
    installed: Path,
    stub_binary: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependencies consume the selected task's one wall-time budget, as ``pixi run`` does."""
    pixi.manifest.write_text(
        '[tasks.prepare]\ncmd = "prepare"\n[tasks.check]\ncmd = "check"\n'
        'depends-on = ["prepare"]\n',
        encoding="utf-8",
    )
    pixi.windows_activation_cache.write_text(
        json.dumps({"environment_variables": {}, "activation_scripts": []}), encoding="utf-8"
    )
    monkeypatch.setattr("platform.system", lambda: "Windows")
    stub_binary("prepare.exe")
    stub_binary("check.exe")
    readings = iter((10.0, 11.0, 13.0))
    monkeypatch.setattr("mainboard.engines.compile.backend.pixi.monotonic", lambda: next(readings))
    timeouts: list[float | None] = []

    def capture(command: BaseCommand, *, timeout: float | None = None) -> CommandResult:
        timeouts.append(timeout)
        return CommandResult(0, "", "")

    monkeypatch.setattr(Process, "capture", capture)

    assert installed.is_dir()
    assert pixi.capture(("check",), timeout=5.0).succeeded
    assert timeouts == [4.0, 2.0]


def test_windows_caches_pixis_complete_activation_after_provisioning(
    fp: FakeProcess,
    pixi: Pixi,
    tool_paths: Mapping[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pixi, not Mainboard, evaluates conda package hooks such as OpenSSL's activation."""
    payload = {
        "environment_variables": {"SSL_CERT_FILE": "C:/prefix/ssl/cacert.pem"},
        "activation_scripts": [],
    }
    monkeypatch.setattr("platform.system", lambda: "Windows")
    fp.register([fp.any()], stdout=json.dumps(payload))

    pixi.cache_windows_activation("default")

    assert json.loads(pixi.windows_activation_cache.read_text()) == payload
    assert list(fp.calls[0]) == [
        tool_paths["pixi"],
        "shell-hook",
        "--manifest-path",
        str(pixi.manifest),
        "--json",
        "-e",
        "default",
    ]


def test_shell_hook_returns_the_activation_script_pixi_prints(
    fp: FakeProcess, pixi: Pixi, tool_paths: Mapping[str, str]
) -> None:
    """The bash activation is captured as text so a generated `activate.sh` needs no pixi."""
    fp.register(
        [
            tool_paths["pixi"],
            "shell-hook",
            "-s",
            "bash",
            "-e",
            "default",
            "--manifest-path",
            str(pixi.manifest),
        ],
        stdout='export PATH="/env/bin:$PATH"\n',
    )
    assert pixi.shell_hook() == 'export PATH="/env/bin:$PATH"\n'


def test_activated_puts_the_env_bin_on_path_only_once_it_exists(pixi: Pixi) -> None:
    """A dry call before an install leaves PATH alone rather than exporting a dead entry."""
    env_bin = pixi.env_prefix("default") / ("Scripts" if os.name == "nt" else "bin")
    env_bin.mkdir(parents=True)
    with pixi.activated("default"):
        assert str(env_bin) in local.env["PATH"]
    before = local.env["PATH"]
    with pixi.activated("empty-env"):
        assert local.env["PATH"] == before


def test_the_lock_reading_names_every_pinned_package(fp: FakeProcess, pixi: Pixi) -> None:
    """The frozen listing is read rather than the lock parsed, so pixi owns its own format."""
    pixi.lock.write_text("version: 7\n")
    fp.register([fp.any()], stdout='[{"name": "torch", "version": "2.9.1"}]')
    assert pixi.locked("default") == {"torch": "2.9.1"}
    assert "--frozen" in fp.calls[0]


@pytest.mark.parametrize(
    "solved",
    [
        pytest.param(False, id="nothing-has-been-solved-yet"),
        pytest.param(True, id="the-lock-holds-no-such-environment"),
    ],
)
def test_the_lock_reading_is_empty_when_there_is_nothing_to_read(
    solved: bool, fp: FakeProcess, pixi: Pixi, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing state reads as a snapshot of nothing.

    A before-and-after reading has to survive the before, rather than raising an error
    somebody has to catch.
    """
    if solved:
        pixi.lock.write_text("version: 7\n")
        fp.register([fp.any()], returncode=1, stderr="unknown environment\n")

    assert pixi.locked("ghost") == {}
    assert capsys.readouterr().err == ""


def test_update_moves_the_lock_and_reports_its_own_failure(fp: FakeProcess, pixi: Pixi) -> None:
    """Refreshing inside the declared bounds is pixi's verb, and its failure is not swallowed."""
    fp.register([fp.any()], stdout="lock updated\n")
    assert pixi.update("default") is None
    fp.register([fp.any()], returncode=1, stderr="could not update\n")
    with pytest.raises(MissionError, match="pixi update"):
        pixi.update("default")
