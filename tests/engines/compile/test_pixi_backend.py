import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import MissionError

if TYPE_CHECKING:
    from pathlib import Path

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
    fp: FakeProcess, pixi: Pixi, tool_paths: Mapping[str, str]
) -> None:
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
