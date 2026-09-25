import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.dispatch.evidence import receipts_in
from mainboard.dispatch.jobs import JobSpec
from mainboard.dispatch.shared import state_dir
from mainboard.runtime.job import PrefixActivation, ToolCall, WorkspaceActivation, walltime_seconds

from ..support import FieldValue, plan, recorded


def spec(**overrides: FieldValue | ToolCall | tuple[str, ...]) -> JobSpec:
    """A `JobSpec` for gold's default environment under `/repo`, overridden field by field."""
    return JobSpec.model_validate({"cmd": "run", "plan": plan(), "root": "/repo", **overrides})


@given(
    hours=st.integers(min_value=0, max_value=99),
    minutes=st.integers(min_value=0, max_value=59),
    seconds=st.integers(min_value=0, max_value=59),
)
def test_a_walltime_converts_to_the_whole_seconds_the_runner_counts(
    hours: int, minutes: int, seconds: int
) -> None:
    walltime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    assert walltime_seconds(walltime) == hours * 3600 + minutes * 60 + seconds


@given(
    command=st.text(min_size=1) | st.sampled_from(["a\u2028b", "x\x85y", "'\"$(rm -rf /)\"'"]),
    value=st.text(),
)
def test_the_script_hands_the_host_the_exact_record_whatever_the_command_holds(
    command: str, value: str
) -> None:
    """Quotes, newlines and shell syntax are data in the record, never script text."""
    rendered = spec(cmd=command, exports={"NOTE": value})
    assert recorded(rendered.render(pbs=False)) == rendered.job(pbs=False)


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
    assert text.splitlines()[:7] == [
        "#!/bin/sh",
        "#PBS -q short-g",
        "#PBS -l select=1:ngpus=1:mem=100gb",
        "#PBS -l walltime=06:00:00",
        "#PBS -W group_list=xg25g007",
        "#PBS -j oe",
        "# A mainboard job. The record below is what runs; this line hands it over.",
    ]
    # PBS enforces the walltime and spools the output itself, so the runner does neither.
    job = recorded(text)
    assert (job.command, job.walltime, job.logs) == (
        "python -m foo",
        "",
        "/repo/.mainboard/dispatch/logs",
    )


def test_the_handover_puts_the_per_user_tools_first_and_replaces_the_shell() -> None:
    """A batch shell is not a login shell, so the tool is found by the directories it lives in."""
    handover = spec().render(pbs=False).splitlines()[-1]
    assert handover.startswith(
        'PATH="$HOME/.local/bin:$HOME/.pixi/bin:$HOME/.cargo/bin:$PATH" exec mainboard job '
    )


def test_a_gpu_queue_that_rejects_an_explicit_count_keeps_it_out_of_the_select_chunk() -> None:
    bare = spec(walltime="00:10:00").render(pbs=True)
    assert "group_list" not in bare
    without_gpu = spec(walltime="00:10:00", gpus=1).render(pbs=True, gpu_in_select=False)
    assert "ngpus" not in without_gpu
    assert "#PBS -l select=1\n" in without_gpu


def test_a_plain_render_caps_the_job_only_when_a_walltime_was_chosen() -> None:
    """An invisible cap killing correct work is worse than a hung job a monitor can cancel."""
    capped = spec(walltime="00:05:00").render(pbs=False)
    assert "#PBS" not in capped
    assert (recorded(capped).walltime, recorded(capped).logs) == ("00:05:00", "")
    assert recorded(spec().render(pbs=False)).walltime == ""


def test_every_job_enters_the_plans_own_environment_or_refuses_to_start() -> None:
    """A queued job and an interactive run must land in the same interpreter."""
    default = spec().job(pbs=False).activation
    assert default == WorkspaceActivation(
        script="/repo/.mainboard/activate.sh",
        prefix="/repo/.mainboard/envs/default/.pixi/envs/default",
        refusal=default.refusal,
    )
    assert "found no default environment at /repo/.mainboard/envs/default" in default.refusal
    assert "mainboard setup gold --env default" in default.refusal
    serving = spec(plan=plan(env="serving")).job(pbs=False).activation
    assert isinstance(serving, WorkspaceActivation)
    assert serving.script == "/repo/.mainboard/activate-serving.sh"


def test_an_addressed_environment_is_entered_frozen_and_never_reconciled() -> None:
    """The activation is the prefix's own, which stops a job asking pixi to reconcile anything."""
    prefix = "/repo/.mainboard/prefixes/default/abcd1234"
    frozen = spec(prefix=prefix).job(pbs=False).activation
    assert frozen == PrefixActivation(prefix=prefix, env="default", refusal=frozen.refusal)
    assert "mainboard provide default` rebuilds exactly it" in frozen.refusal


def test_the_job_owns_its_pythonpath_command_and_container() -> None:
    job = spec(cmd="python -m foo --name 'a b'", pythonpath="/repo/src").job(pbs=False)
    assert (job.command, job.pythonpath, job.isolate_pythonpath) == (
        "python -m foo --name 'a b'",
        "/repo/src",
        True,
    )
    assert not spec(isolate_pythonpath=False).job(pbs=False).isolate_pythonpath
    argv = ("apptainer", "exec", "image.sif", "bash", "-c", "run")
    assert spec(container=argv).job(pbs=False).container == argv


def test_a_dispatched_job_carries_the_provenance_a_mirror_cannot_derive_and_nothing_empty() -> (
    None
):
    """A snapshot has no `.git`, so a preflight there can read neither HEAD nor a worktree.

    The dispatcher can read both, so it declares them: the commit says which revision this is,
    and the digest says these exact bytes, which is what a run seals itself against where there
    is no history to ask. A dispatch that could read neither says neither, rather than exporting
    an empty claim, and a host's exports come after every fact about the run.
    """
    sealed = spec(
        source="e975499",
        commit="e975499f" * 5,
        digest="9a" * 32,
        closure="/repo/.mainboard/dispatch/sources/k/.mainboard-closure",
        first_party="core:experiments",
        deferred="cutoken",
        exports={"HF_HUB_OFFLINE": "1", "NOTE": "two words"},
    ).job(pbs=False)
    assert list(sealed.variables.items()) == [
        ("MAINBOARD_SOURCE", "e975499"),
        ("MAINBOARD_SOURCE_COMMIT", "e975499f" * 5),
        ("MAINBOARD_SOURCE_DIGEST", "9a" * 32),
        ("MAINBOARD_CLOSURE", "/repo/.mainboard/dispatch/sources/k/.mainboard-closure"),
        ("MAINBOARD_FIRST_PARTY", "core:experiments"),
        ("MAINBOARD_DEFERRED", "cutoken"),
        ("HF_HUB_OFFLINE", "1"),
        ("NOTE", "two words"),
    ]
    assert spec(source="e975499").job(pbs=False).variables == {"MAINBOARD_SOURCE": "e975499"}


def test_the_calls_around_the_command_travel_as_this_tools_own_verbs() -> None:
    build = ToolCall(args=("provide", "default", "--source", "x"), cwd="/repo")
    watch = ToolCall(args=("sample", "s", "--job", "j"), credentials="/repo/.mainboard/t.json")
    attest = ToolCall(args=("attest", "s", "--job", "j"))
    job = recorded(spec(provide=build, sampler=watch, attestation=attest).render(pbs=False))
    assert (job.provide, job.sampler, job.attestation) == (build, watch, attest)


@pytest.mark.skipif(sys.platform == "win32", reason="the handover is a POSIX shell script")
@pytest.mark.parametrize("pbs", [False, True], ids=["a queue passing the path", "PBS on stdin"])
def test_a_rendered_script_run_by_sh_hands_over_to_the_tool_on_its_path(
    tmp_path: Path, pbs: bool
) -> None:
    """End to end: `sh` runs the script, the tool it finds runs the record, receipts come back.

    PBS may feed the script to the shell on stdin rather than pass its path, which is why the
    record travels inline, and under it everything the job says, the activation's chatter
    included, lands in the log a later poll reads.
    """
    tool = shutil.which("mainboard", path=str(Path(sys.executable).parent))
    if tool is None:
        pytest.skip("the tool's console script is not installed beside this interpreter")
    (tmp_path / ".mainboard").mkdir()
    (tmp_path / ".mainboard" / "activate.sh").write_text("echo activating\n", encoding="utf-8")
    receipt = '{"trial_receipt": {"run_id": "sh"}}'
    command = f"printf '%s\\n' {shlex.quote(receipt)} >> \"$MAINBOARD_RECEIPTS\"; exit 3"
    script = tmp_path / "job.sh"
    rendered = spec(cmd=command, root=tmp_path.as_posix(), walltime="00:10:00")
    script.write_text(rendered.render(pbs=pbs), encoding="utf-8")
    done = subprocess.run(
        ["sh"] if pbs else ["sh", str(script)],
        input=script.read_text(encoding="utf-8") if pbs else None,
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "PATH": f"{Path(tool).parent}:{os.environ['PATH']}",
            "PBS_JOBID": "42.opbs",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 3, done.stderr
    logs = tmp_path / state_dir() / "logs"
    output = (logs / "42.log").read_text(encoding="utf-8") if pbs else done.stdout
    assert output.startswith("activating\n")
    assert receipts_in(output) == (receipt,)
    assert (logs / "42.exit").is_file() is pbs
