from typing import TYPE_CHECKING

from plumbum import local

from mainboard.engines.compile import Provisioner, task_line
from mainboard.engines.compile.state import SyncState

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.manifest import Manifest


def test_out_dir_and_backends_are_wired_under_the_projects_generated_directory(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    provisioner = Provisioner(tmp_path, manifest)
    assert provisioner.out == tmp_path / ".mainboard"
    assert provisioner.pixi.manifest == provisioner.out / "pixi.toml"
    assert provisioner.compiler.out == provisioner.out


def test_provision_compiles_and_installs_under_one_lock(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\nplatforms = ["linux-64"]\n')
    provisioner = Provisioner(tmp_path, manifest)
    # A real `pixi install --resolve` leaves a lock behind as a side effect of solving, which
    # is what its own recursive locked re-verify then checks for. The fake process never
    # writes one, so it is pre-seeded here to stand in for that first call's real effect.
    provisioner.out.mkdir()
    provisioner.pixi.lock.write_text("version: 7\n")
    for _ in range(2):
        fp.register([fp.any()], stdout="environment ready\n")

    provisioner.provision(resolve=True)

    assert provisioner.pixi.manifest.exists()
    assert SyncState.load(provisioner.out).envs.get("default") == provisioner.compiler.digest()


def test_provision_skips_recompiling_an_already_fresh_env(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\nplatforms = ["linux-64"]\n')
    provisioner = Provisioner(tmp_path, manifest)
    provisioner.out.mkdir()
    provisioner.pixi.lock.write_text("version: 7\n")
    for _ in range(2):
        fp.register([fp.any()], stdout="environment ready\n")
    provisioner.provision(resolve=True)
    written_at = provisioner.pixi.manifest.read_text()

    fp.register([fp.any()], stdout="environment ready\n")
    provisioner.provision()

    assert provisioner.pixi.manifest.read_text() == written_at


def test_activated_never_compiles_a_workspace_that_was_never_provisioned(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """First-time compilation is `provision`'s job, `activated` only catches up a stale one."""
    manifest = manifest_from('[workspace]\nname = "w"\n')
    provisioner = Provisioner(tmp_path, manifest)

    with provisioner.activated():
        pass

    assert not provisioner.pixi.manifest.exists()


def test_activated_recompiles_a_provisioned_env_that_has_gone_stale(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    """`activated` only compiles under its lock, it never touches `pixi install`."""
    fresh = manifest_from('[workspace]\nname = "w"\n')
    provisioner = Provisioner(tmp_path, fresh)
    provisioner.out.mkdir()
    provisioner.pixi.lock.write_text("version: 7\n")
    for _ in range(2):
        fp.register([fp.any()], stdout="environment ready\n")
    provisioner.provision(resolve=True)

    edited = manifest_from('[workspace]\nname = "w"\n[deps]\nripgrep = "*"\n')
    edited_provisioner = Provisioner(tmp_path, edited)
    with edited_provisioner.activated():
        pass

    assert len(fp.calls) == 2  # no further pixi invocation happened during `activated`
    assert "ripgrep" in edited_provisioner.pixi.manifest.read_text()


def test_activated_puts_the_provisioned_envs_bin_on_path(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    provisioner = Provisioner(tmp_path, manifest)
    env_bin = provisioner.pixi.env_prefix("default") / "bin"
    env_bin.mkdir(parents=True)

    with provisioner.activated():
        assert str(env_bin) in local.env["PATH"]


def test_activate_writes_the_script_without_modules(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    provisioner = Provisioner(tmp_path, manifest)
    fp.register([fp.any()], stdout="export PATH=/env/bin:$PATH\n")

    path = provisioner.activate()

    assert path == provisioner.out / "activate.sh"
    text = path.read_text()
    assert "module load" not in text
    assert "export PATH=/env/bin:$PATH" in text


def test_activated_puts_a_second_stage_toolchains_binaries_ahead_of_the_env(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """A tool npm installed is reachable by name exactly like a conda one."""
    manifest = manifest_from('[workspace]\nname = "w"\n[nodejs.deps]\nprettier = ">=3"\n')
    provisioner = Provisioner(tmp_path, manifest)
    linked = provisioner.out / "node_modules" / ".bin"
    linked.mkdir(parents=True)

    with provisioner.activated():
        assert local.env["PATH"].startswith(str(linked))


def test_activated_leaves_out_a_directory_nothing_has_installed_into(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    provisioner = Provisioner(tmp_path, manifest)
    before = local.env["PATH"]

    with provisioner.activated():
        assert local.env["PATH"] == before


def test_provision_installs_the_second_stage_after_pixi(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    fp: FakeProcess,
    stub_binary: Callable[[str], str],
) -> None:
    """Every second-stage manager ships as a conda package, so pixi has to land first."""
    npm = stub_binary("npm")
    manifest = manifest_from(
        '[workspace]\nname = "w"\nplatforms = ["linux-64"]\n[nodejs.deps]\nprettier = ">=3"\n'
    )
    provisioner = Provisioner(tmp_path, manifest)
    provisioner.out.mkdir()
    provisioner.pixi.lock.write_text("version: 7\n")
    for _ in range(3):
        fp.register([fp.any()], stdout="environment ready\n")

    provisioner.provision(resolve=True)

    assert "prettier" in (provisioner.out / "package.json").read_text()
    assert [next(iter(call)) for call in fp.calls][-1] == npm


def test_activate_carries_the_second_stage_binaries_into_the_generated_script(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    """A remote job sourcing `activate.sh` gets the same PATH `activated` builds in process."""
    manifest = manifest_from('[workspace]\nname = "w"\n[nodejs.deps]\nprettier = ">=3"\n')
    provisioner = Provisioner(tmp_path, manifest)
    linked = provisioner.out / "node_modules" / ".bin"
    linked.mkdir(parents=True)
    fp.register([fp.any()], stdout="export PATH=/env/bin:$PATH\n")

    assert str(linked) in provisioner.activate().read_text()


def test_task_line_hands_a_workspace_task_to_pixi_in_the_generated_workspace(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """The manifest path is relative because a wrapped command already changed into the root."""
    manifest = manifest_from('[workspace]\nname = "w"\n[tasks]\nlint = "ruff check"\n')
    assert task_line(manifest, "lint --fix", env="default") == (
        "pixi run --manifest-path .mainboard/pixi.toml --frozen -e default lint --fix"
    )


def test_task_line_resolves_a_task_an_environment_declares_for_itself(
    manifest_from: Callable[[str], Manifest],
) -> None:
    manifest = manifest_from(
        '[workspace]\nname = "w"\n[envs.serving.tasks]\nserve = "vllm serve"\n'
    )
    assert task_line(manifest, "serve", env="serving").endswith("-e serving serve")
    assert task_line(manifest, "serve", env="default") == "serve"


def test_task_line_leaves_an_ordinary_command_exactly_as_written(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """A command line is not a task name, and an env nobody declared resolves no tasks."""
    manifest = manifest_from('[workspace]\nname = "w"\n[tasks]\nlint = "ruff check"\n')
    assert task_line(manifest, "python -c 'print(1)'", env="default") == "python -c 'print(1)'"
    assert task_line(manifest, "lint", env="undeclared").startswith("pixi run")


def test_activate_gives_a_named_environment_its_own_script(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    """Installing one environment must not overwrite the activation another one is sourced by."""
    manifest = manifest_from('[workspace]\nname = "w"\n[envs.serving]\n')
    provisioner = Provisioner(tmp_path, manifest)
    fp.register([fp.any()], stdout="export PATH=/env/serving/bin:$PATH\n")

    path = provisioner.activate("serving")

    assert path == provisioner.out / "activate-serving.sh"
    assert not (provisioner.out / "activate.sh").exists()


def test_activate_writes_the_script_with_per_host_modules(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    provisioner = Provisioner(tmp_path, manifest)
    fp.register([fp.any()], stdout="export PATH=/env/bin:$PATH\n")

    path = provisioner.activate(modules={"singularity": "4.2.1"})

    assert "module load singularity/4.2.1" in path.read_text()
