import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.dispatch.jobs import JobSpec, walltime_seconds

from ..support import FieldValue, plan


def spec(**overrides: FieldValue) -> JobSpec:
    """A `JobSpec` for gold's default environment under `/repo`, overridden field by field."""
    fields: dict[str, FieldValue] = {"cmd": "run", "plan": plan(), "root": "/repo"}
    fields.update(overrides)
    return JobSpec.model_validate(fields)


@given(
    hours=st.integers(min_value=0, max_value=99),
    minutes=st.integers(min_value=0, max_value=59),
    seconds=st.integers(min_value=0, max_value=59),
)
def test_a_walltime_converts_to_the_whole_seconds_the_timeout_wrapper_counts(
    hours: int, minutes: int, seconds: int
) -> None:
    walltime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    assert walltime_seconds(walltime) == hours * 3600 + minutes * 60 + seconds


def test_a_pbs_render_needs_an_explicit_walltime_and_carries_the_full_header() -> None:
    """A silently injected site constant that kills correct work is worse than no default."""
    with pytest.raises(ValueError, match="explicit walltime"):
        spec(cmd="python -m foo").render(pbs=True)
    text = spec(
        cmd="python -m foo",
        queue="short-g",
        walltime="06:00:00",
        account="xg25g007",
        mem_gb=100,
        gpus=1,
    ).render(pbs=True)
    assert "#PBS -q short-g" in text
    assert "#PBS -l select=1:ngpus=1:mem=100gb" in text
    assert "#PBS -l walltime=06:00:00" in text
    assert "#PBS -W group_list=xg25g007" in text
    assert "trap 'mainboard_exit $?' EXIT" in text
    assert "bash -c 'python -m foo'" in text


def test_a_gpu_queue_that_rejects_an_explicit_count_keeps_it_out_of_the_select_chunk() -> None:
    bare = spec(walltime="00:10:00").render(pbs=True)
    assert "group_list" not in bare
    without_gpu = spec(walltime="00:10:00", gpus=1).render(pbs=True, gpu_in_select=False)
    assert "ngpus" not in without_gpu
    assert "select=1" in without_gpu


def test_a_bash_render_caps_the_job_only_when_a_walltime_was_chosen() -> None:
    """An invisible cap killing correct work is worse than a hung job a monitor can cancel."""
    capped = spec(walltime="00:05:00").render(pbs=False)
    assert "timeout --kill-after=30s 300 bash" in capped
    assert "mainboard: killed at walltime 00:05:00" in capped
    uncapped = spec().render(pbs=False)
    assert "timeout" not in uncapped
    assert "MAINBOARD_TIMED" not in uncapped


def test_every_render_activates_the_plans_own_environment_or_refuses_to_start() -> None:
    """A queued job and an interactive run must land in the same interpreter."""
    default = spec().render(pbs=False)
    assert "if [ -f /repo/.mainboard/activate.sh ]" in default
    assert "elif [ -f /repo/.chefe/activate.sh ]" in default
    assert "export PATH=/repo/.mainboard/envs/default/.pixi/envs/default/bin:$PATH" in default
    assert (
        "found no default environment at /repo/.mainboard/envs/default/.pixi/envs/default on gold"
        in default
    )
    assert "mainboard install default --on gold" in default
    assert "exit 1" in default
    serving = spec(plan=plan(env="serving")).render(pbs=False)
    assert "if [ -f /repo/.mainboard/activate-serving.sh ]" in serving
    assert "/repo/.mainboard/activate.sh" not in serving


def test_the_rendered_body_quotes_the_command_and_owns_its_pythonpath() -> None:
    quoted = spec(cmd="python -m foo --name 'a b'").render(pbs=False)
    assert "bash -c 'python -m foo --name" in quoted
    assert "unset PYTHONPATH" in quoted
    assert "PYTHONPATH" not in spec(isolate_pythonpath=False).render(pbs=False)
    assert "export PYTHONPATH=/repo/src" in spec(pythonpath="/repo/src").render(pbs=False)
    contained = spec(container_command="apptainer exec image.sif bash -c 'run'").render(pbs=False)
    assert "apptainer exec image.sif bash -c 'run' || status=$?" in contained
    assert contained.count("bash -c 'run'") == 1
    # The command's own status is kept and re-raised, so framing the receipts back after it
    # costs the job none of its exit code.
    assert contained.splitlines()[-1] == "exit $status"


def test_a_dispatched_job_learns_which_source_it_is_and_only_when_one_was_declared() -> None:
    stamped = spec(source="abc1234-dirty").render(pbs=False)
    assert "export MAINBOARD_SOURCE=abc1234-dirty" in stamped
    assert "unset PYTHONPATH" in stamped
    silent = spec().render(pbs=False)
    assert "MAINBOARD_SOURCE" not in silent


def test_a_dispatched_job_carries_the_provenance_a_mirror_cannot_derive() -> None:
    """A snapshot has no `.git`, so a preflight there can read neither HEAD nor a worktree.

    The dispatcher can read both, so it declares them: the commit says which revision this is,
    and the digest says these exact bytes, which is what a run seals itself against where there
    is no history to ask. Without them a runner has to be handed the number by hand at submit
    time, which is what the reproducibility campaign's own `--expect-source` was (2026-09-05).
    """
    sealed = spec(source="e975499", commit="e975499f" * 5, digest="9a" * 32).render(pbs=False)

    assert f"export MAINBOARD_SOURCE_COMMIT={'e975499f' * 5}" in sealed
    assert f"export MAINBOARD_SOURCE_DIGEST={'9a' * 32}" in sealed
    # Declared before the command, like every other fact about the run, so a preflight inside it
    # can refuse before anything is measured.
    assert sealed.index("MAINBOARD_SOURCE_DIGEST") < sealed.index("bash -c")
    # And a dispatch that could read neither says neither, rather than exporting an empty claim.
    assert "MAINBOARD_SOURCE_COMMIT" not in spec(source="e975499").render(pbs=False)


def test_an_addressed_environment_is_activated_frozen_and_first_on_the_library_path() -> None:
    """Thirty two GH200 jobs died importing sqlite3 against `/lib64`'s libstdc++.

    The prefix's own `lib` was second on the loader's path, so a moment when the environment was
    mid-reconciliation sent every one of them to the system library and a `CXXABI_1.3.15` that
    is not in it. The prefix goes first here, and the activation is the prefix's own rather than
    the workspace's, which is what stops a job asking pixi to reconcile anything at all.
    """
    prefix = "/repo/.mainboard/prefixes/default/abcd1234"
    frozen = spec(prefix=prefix).render(pbs=False)

    assert f"source {prefix}/activate.sh" in frozen
    assert f"export LD_LIBRARY_PATH={prefix}/.pixi/envs/default/lib${{LD_LIBRARY_PATH:+" in frozen
    assert "mainboard install" not in frozen
    # The workspace's own activation is what a job with no addressed environment still gets.
    assert "/repo/.mainboard/activate.sh" in spec().render(pbs=False)


def test_a_hosts_exports_are_written_before_the_command_and_quoted_as_the_shell_needs() -> None:
    """`HF_HUB_OFFLINE` on a cluster reaches the job as one export line, and a space survives."""
    body = spec(exports={"HF_HUB_OFFLINE": "1", "NOTE": "two words"}).render(pbs=False)
    assert "export HF_HUB_OFFLINE=1\n" in body
    assert "export NOTE='two words'\n" in body
    assert body.index("export HF_HUB_OFFLINE=1") < body.index("bash -c")
