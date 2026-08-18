from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import MissionError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi

_FINGERPRINT = ".pixi-environment-fingerprint"


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


def damage(site_packages: Path, name: str) -> Path:
    """Leave ``name`` recorded as installed with the import root it declares missing.

    Returns the absent root, so a fake reinstall can put it back the way a real one would.
    """
    metadata = site_packages / f"{name}-1.0.dist-info"
    metadata.mkdir()
    metadata.joinpath("METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
    metadata.joinpath("INSTALLER").write_text("uv-pixi")
    metadata.joinpath("top_level.txt").write_text(f"{name.replace('-', '_')}\n")
    return site_packages / name.replace("-", "_")


def test_install_requires_a_lock_unless_resolution_was_requested(
    fp: FakeProcess, pixi: Pixi
) -> None:
    """A missing generated lock never turns an ordinary install into an implicit solve."""
    with pytest.raises(MissionError, match=r"pixi.lock is missing.*resolve=True"):
        pixi.install("default")
    assert not fp.calls


def test_successful_pixi_install_returns_cleanly(fp: FakeProcess, pixi: Pixi) -> None:
    """A successful locked installation has no error epilogue."""
    pixi.manifest.write_text('[workspace]\nplatforms = ["linux-64"]\n')
    pixi.lock.write_text("version: 7\n")
    fp.register([fp.any()], stdout="environment ready\n")
    assert pixi.install("default") is None


def test_failed_pixi_install_reaches_the_callers_output(
    fp: FakeProcess, pixi: Pixi, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Pixi failure is tee'd before mainboard raises, including both native output streams."""
    pixi.lock.write_text("version: 7\n")
    fp.register(
        [fp.any()], returncode=17, stdout="pixi solver context\n", stderr="pixi solver failed\n"
    )
    with pytest.raises(MissionError, match="pixi install"):
        pixi.install("default")
    captured = capsys.readouterr()
    assert captured.out == "pixi solver context\n"
    assert captured.err == "pixi solver failed\n"


def test_locked_environment_refuses_manifest_drift_with_actionable_error(
    fp: FakeProcess, pixi: Pixi
) -> None:
    """A stale lock aborts without a second solving call and explains the explicit escape."""
    pixi.lock.write_text("version: 7\n")
    fp.register(
        [fp.any()], returncode=1, stderr="the lock file is not up-to-date with the workspace\n"
    )
    with pytest.raises(MissionError, match=r"manifest drifted.*resolve=True"):
        pixi.install("default")
    assert len(fp.calls) == 1
    assert "--locked" in list(fp.calls[0])


def test_a_normal_task_failure_mentioning_lock_is_not_mislabeled_as_drift(
    fp: FakeProcess, pixi: Pixi
) -> None:
    """A pixi task's own failure output is not confused with pixi's own drift rejection."""
    pixi.lock.write_text("version: 7\n")
    fp.register(
        [fp.any()],
        returncode=9,
        stdout="Pixi task (build): command\n",
        stderr="task says lock file is not up-to-date\n",
    )
    with pytest.raises(MissionError, match="pixi install"):
        pixi.install("default")


def test_editable_path_environment_installs_frozen_after_resolving(
    fp: FakeProcess, pixi: Pixi
) -> None:
    """A mutable editable source trusts the resolved lock without demanding unchanged code."""
    pixi.manifest.write_text('[pypi-dependencies.demo]\npath = "../demo"\neditable = true\n')
    pixi.lock.write_text("version: 7\n")
    for _ in range(2):
        fp.register([fp.any()], stdout="environment ready\n")

    pixi.install("default", resolve=True)

    assert "--locked" not in list(fp.calls[0]) and "--frozen" not in list(fp.calls[0])
    assert "--frozen" in list(fp.calls[1])


def test_resolving_install_verifies_the_new_lock_through_the_locked_path(
    fp: FakeProcess, pixi: Pixi, tool_paths: Mapping[str, str]
) -> None:
    """A solve is successful only after the resulting pair passes Pixi's locked install check."""
    pixi.lock.write_text("version: 7\n")
    base = [tool_paths["pixi"], "install", "--manifest-path", str(pixi.manifest)]
    fp.register([*base, "-e", "serving"], stdout="environment ready\n")
    fp.register([*base, "--locked", "-e", "serving"], stdout="lock verified\n")

    pixi.install("serving", resolve=True)

    assert len(fp.calls) == 2
    assert "--locked" not in list(fp.calls[0])
    assert "--locked" in list(fp.calls[1])


def test_readiness_trusts_the_installation_fingerprint_over_the_directory(
    pixi: Pixi, installed: Path
) -> None:
    """An interrupted install leaves a prefix behind, so only pixi's own stamp proves one ran."""
    assert pixi.ready("default") is True
    assert pixi.ready("serving") is False

    (pixi.env_prefix("default") / "conda-meta" / _FINGERPRINT).unlink()

    assert installed.is_dir()
    assert pixi.ready("default") is False


def test_install_repairs_a_wheel_damaged_underneath_pixi(
    fp: FakeProcess, pixi: Pixi, installed: Path, tool_paths: Mapping[str, str]
) -> None:
    """A retained `dist-info` whose files vanished is reinstalled through the locked env."""
    root = damage(installed, "cupy-cuda13x")
    base = [tool_paths["pixi"], "--manifest-path", str(pixi.manifest), "--locked", "-e", "default"]
    fp.register([tool_paths["pixi"], "install", *base[1:]], stdout="environment ready\n")
    fp.register(
        [tool_paths["pixi"], "reinstall", *base[1:], "cupy-cuda13x"],
        stdout="package reinstalled\n",
        callback=lambda process: root.mkdir(),
    )

    pixi.install("default")

    assert len(fp.calls) == 2
    assert list(fp.calls[1])[1:] == ["reinstall", *base[1:], "cupy-cuda13x"]


def test_install_leaves_an_intact_environment_untouched(
    fp: FakeProcess, pixi: Pixi, installed: Path
) -> None:
    """The audit costs a scan and nothing else when every package still imports."""
    intact = damage(installed, "cupy-cuda13x")
    intact.mkdir()
    fp.register([fp.any()], stdout="environment ready\n")

    pixi.install("default")

    assert len(fp.calls) == 1


def test_install_audits_nothing_until_an_installation_has_finished(
    fp: FakeProcess, pixi: Pixi, installed: Path
) -> None:
    """A prefix an interrupted install abandoned reads as damage nobody should act on."""
    damage(installed, "cupy-cuda13x")
    (pixi.env_prefix("default") / "conda-meta" / _FINGERPRINT).unlink()
    fp.register([fp.any()], stdout="environment ready\n")

    pixi.install("default")

    assert len(fp.calls) == 1


def test_install_reports_a_failed_repair(fp: FakeProcess, pixi: Pixi, installed: Path) -> None:
    """A failed targeted reinstall stays a hard installation failure."""
    damage(installed, "cupy-cuda13x")
    fp.register([fp.any()], stdout="environment ready\n")
    fp.register([fp.any()], returncode=9, stderr="reinstall failed\n")

    with pytest.raises(MissionError, match="pixi reinstall"):
        pixi.install("default")


def test_install_refuses_an_environment_the_repair_did_not_mend(
    fp: FakeProcess, pixi: Pixi, installed: Path
) -> None:
    """A reinstall reporting success cannot conceal a package that still imports nothing."""
    damage(installed, "cupy-cuda13x")
    fp.register([fp.any()], stdout="environment ready\n")
    fp.register([fp.any()], stdout="package reinstalled\n")

    with pytest.raises(MissionError, match="cupy-cuda13x stayed incomplete"):
        pixi.install("default")


def test_pixi_scope_pins_manifest_path(pixi: Pixi) -> None:
    """The pixi backend injects `--manifest-path` so every call targets the env it owns."""
    assert pixi.scope() == ("--manifest-path", str(pixi.manifest))


def test_env_prefix_lives_under_the_generated_dirs_own_pixi_envs(pixi: Pixi) -> None:
    assert pixi.env_prefix("serving") == pixi.manifest.parent / ".pixi" / "envs" / "serving"


def test_pixi_shell_hook_returns_activation_script(
    fp: FakeProcess, pixi: Pixi, tool_paths: Mapping[str, str]
) -> None:
    """`shell_hook` asks pixi for the bash activation of an env and returns it verbatim."""
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


def test_pixi_activated_puts_the_env_bin_on_path(pixi: Pixi) -> None:
    """`activated` prepends the env's bin when it exists, and leaves PATH alone when it doesn't.

    This is what lets a provisioned toolchain resolve right after `provision()`.
    """
    env_bin = pixi.env_prefix("default") / "bin"
    env_bin.mkdir(parents=True)
    with pixi.activated("default"):
        assert str(env_bin) in local.env["PATH"]
    before = local.env["PATH"]
    with pixi.activated("empty-env"):
        assert local.env["PATH"] == before


def test_has_editable_paths_is_false_without_a_generated_manifest(pixi: Pixi) -> None:
    assert pixi._has_editable_paths() is False  # ruff:ignore[private-member-access]  reason=unit-tests the lock-rule helper since=2026-08-17


def test_has_editable_paths_is_false_with_no_matching_source(pixi: Pixi) -> None:
    pixi.manifest.write_text('[dependencies]\nnumpy = "*"\n')
    assert pixi._has_editable_paths() is False  # ruff:ignore[private-member-access]  reason=unit-tests the lock-rule helper since=2026-08-17


def test_has_editable_paths_finds_a_source_nested_inside_an_array(pixi: Pixi) -> None:
    pixi.manifest.write_text('[[workspace.extra]]\npath = "../demo"\neditable = true\n')
    assert pixi._has_editable_paths() is True  # ruff:ignore[private-member-access]  reason=unit-tests the lock-rule helper since=2026-08-17


def test_has_editable_paths_ignores_a_path_without_editable(pixi: Pixi) -> None:
    pixi.manifest.write_text('[dependencies.demo]\npath = "../demo"\n')
    assert pixi._has_editable_paths() is False  # ruff:ignore[private-member-access]  reason=unit-tests the lock-rule helper since=2026-08-17
