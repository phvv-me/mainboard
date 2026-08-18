import pytest

from mainboard.dispatch.jobs import JobSpec, walltime_seconds


def test_walltime_seconds_converts_hh_mm_ss() -> None:
    assert walltime_seconds("01:02:03") == 3723
    assert walltime_seconds("00:00:00") == 0


def test_pbs_render_requires_an_explicit_walltime() -> None:
    spec = JobSpec(cmd="python -m foo", env_prefix="/repo/.mainboard/envs/default")
    with pytest.raises(ValueError, match="explicit walltime"):
        spec.render(pbs=True)


def test_pbs_render_emits_the_header_and_exit_trap() -> None:
    spec = JobSpec(
        cmd="python -m foo",
        env_prefix="/repo/.mainboard/envs/default",
        queue="short-g",
        walltime="06:00:00",
        account="xg25g007",
        mem_gb=100,
        gpus=1,
    )
    text = spec.render(pbs=True)
    assert "#PBS -q short-g" in text
    assert "#PBS -l select=1:ngpus=1:mem=100gb" in text
    assert "#PBS -l walltime=06:00:00" in text
    assert "#PBS -W group_list=xg25g007" in text
    assert "trap 'mainboard_exit $?' EXIT" in text
    assert "bash -c 'python -m foo'" in text


def test_pbs_render_omits_the_account_line_when_unset() -> None:
    spec = JobSpec(cmd="run", env_prefix="/p", walltime="00:10:00")
    text = spec.render(pbs=True)
    assert "group_list" not in text


def test_pbs_render_drops_ngpus_when_gpu_in_select_is_false() -> None:
    spec = JobSpec(cmd="run", env_prefix="/p", walltime="00:10:00", gpus=1)
    text = spec.render(pbs=True, gpu_in_select=False)
    assert "ngpus" not in text
    assert "select=1" in text


def test_bash_render_with_a_walltime_wraps_in_timeout() -> None:
    spec = JobSpec(cmd="run", env_prefix="/p", walltime="00:05:00")
    text = spec.render(pbs=False)
    assert "timeout --kill-after=30s 300 bash" in text
    assert "mainboard: killed at walltime 00:05:00" in text


def test_bash_render_without_a_walltime_is_uncapped() -> None:
    spec = JobSpec(cmd="run", env_prefix="/p")
    text = spec.render(pbs=False)
    assert "timeout" not in text
    assert "MAINBOARD_TIMED" not in text


def test_render_sources_the_workspace_activation_the_wrapped_line_uses() -> None:
    spec = JobSpec(cmd="run", env_prefix="/repo/.mainboard/.pixi/envs/default", root="/repo")
    text = spec.render(pbs=False)
    assert "if [ -f /repo/.mainboard/activate.sh ]" in text
    assert "elif [ -f /repo/.chefe/activate.sh ]" in text
    assert "export PATH=/repo/.mainboard/.pixi/envs/default/bin:$PATH" in text


def test_render_sources_the_named_environments_own_activation() -> None:
    """A queued job and an interactive run must land in the same environment."""
    spec = JobSpec(
        cmd="run", env_prefix="/repo/.mainboard/.pixi/envs/serving", env="serving", root="/repo"
    )
    text = spec.render(pbs=False)
    assert "if [ -f /repo/.mainboard/activate-serving.sh ]" in text
    assert "/repo/.mainboard/activate.sh" not in text


def test_render_refuses_a_host_with_no_environment_to_activate() -> None:
    spec = JobSpec(cmd="run", env_prefix="/repo/.mainboard/.pixi/envs/default", root="/repo")
    text = spec.render(pbs=False)
    assert "no environment at /repo/.mainboard/.pixi/envs/default" in text
    assert "exit 1" in text


def test_render_without_a_root_activates_through_the_prefix_alone() -> None:
    text = JobSpec(cmd="run", env_prefix="/p").render(pbs=False)
    assert "activate.sh" not in text
    assert "if [ -d /p/bin ]; then export PATH=/p/bin:$PATH" in text


def test_render_keeps_an_inherited_pythonpath_when_isolation_is_off() -> None:
    text = JobSpec(cmd="run", env_prefix="/p", isolate_pythonpath=False).render(pbs=False)
    assert "PYTHONPATH" not in text


def test_render_sets_pythonpath_when_given() -> None:
    spec = JobSpec(cmd="run", env_prefix="/p", pythonpath="/repo/src")
    assert "export PYTHONPATH=/repo/src" in spec.render(pbs=False)


def test_render_unsets_pythonpath_by_default() -> None:
    spec = JobSpec(cmd="run", env_prefix="/p")
    assert "unset PYTHONPATH" in spec.render(pbs=False)


def test_render_runs_the_container_command_instead_of_bare_bash() -> None:
    spec = JobSpec(
        cmd="run", env_prefix="/p", container_command="apptainer exec image.sif bash -c 'run'"
    )
    text = spec.render(pbs=False)
    assert text.splitlines()[-1] == "apptainer exec image.sif bash -c 'run'"
    assert "bash -c 'run'" not in text.replace("apptainer exec image.sif bash -c 'run'", "")


def test_render_shell_quotes_the_command() -> None:
    spec = JobSpec(cmd="python -m foo --name 'a b'", env_prefix="/p")
    text = spec.render(pbs=False)
    assert "bash -c 'python -m foo --name" in text
