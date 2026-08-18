import pytest

from mainboard.dispatch.schedulers import Pbs, Resources, build_qsub_flags
from mainboard.dispatch.schedulers.pbs import (
    JobInfo,
    PbsState,
    _extract_job_id,
    bare,
    parse_job_state,
    parse_qstat_full,
    parse_qstat_output,
    parse_qstat_queues,
    parse_rsc_queues,
    pbs_verdict,
)

from ..conftest import machine_with

_QSTAT_STANDARD = """
Job ID                  Username    Jobname     Time Use S Queue
------------------------ ----------- ----------- -------- - -----
2435326.opbs             pedro       train.sh    00:10:00 R short-g
""".strip("\n")

_QSTAT_WIDE = """
JOB_ID   JOB_NAME STATUS PROJECT QUEUE    START_DATE ELAPSE TOKEN NODE MIG
2435326  train.sh R      proj    short-g  Jan 1 12:00 00:10:00 tok  node1 -
""".strip("\n")


# --- bare() ---


def test_bare_strips_the_server_suffix() -> None:
    assert bare("2435326.opbs") == "2435326"
    assert bare("2435326") == "2435326"


# --- parse_job_state ---


def test_parse_job_state_accepts_single_letters_and_words() -> None:
    assert parse_job_state("R") is PbsState.RUNNING
    assert parse_job_state("RUNNING") is PbsState.RUNNING
    assert parse_job_state("BEGUN") is PbsState.ARRAY_BEGUN
    assert parse_job_state("mystery") == "mystery"


# --- parse_qstat_output ---


def test_parse_qstat_output_standard_layout() -> None:
    jobs = parse_qstat_output(_QSTAT_STANDARD)
    assert [job.job_id for job in jobs] == ["2435326.opbs"]
    assert jobs[0].queue == "short-g"
    assert jobs[0].state is PbsState.RUNNING


def test_parse_qstat_output_wide_vendor_layout() -> None:
    jobs = parse_qstat_output(_QSTAT_WIDE)
    assert jobs[0].job_id == "2435326"
    assert jobs[0].name == "train.sh"


def test_parse_qstat_output_with_no_recognizable_header_is_empty() -> None:
    assert parse_qstat_output("nothing to see here\n") == []


def test_parse_qstat_output_skips_short_standard_rows() -> None:
    output = "Job ID\n------\ntoo few cols\n"
    assert parse_qstat_output(output) == []


def test_parse_qstat_output_skips_short_wide_rows() -> None:
    output = "JOB_ID JOB_NAME\ntoo short\n"
    assert parse_qstat_output(output) == []


def test_parse_qstat_output_standard_five_column_row() -> None:
    output = "Job ID   Jobname User State Queue\n--------\n123.srv job usr R q\n"
    jobs = parse_qstat_output(output)
    assert jobs[0].job_id == "123.srv"
    assert jobs[0].state is PbsState.RUNNING
    assert jobs[0].queue == "q"


# --- parse_qstat_queues / parse_rsc_queues ---


def test_parse_qstat_queues_reads_the_flush_left_body() -> None:
    output = "Queue  Max ...\n------ ---\nshort-g  active\ndebug-g  active\n   totals 2\n"
    assert parse_qstat_queues(output) == ["short-g", "debug-g"]


def test_parse_rsc_queues_keeps_only_enabled_rows() -> None:
    output = "interact-g |--[ENABLE, START]\n  `--_n1 [DISABLE]\nmig [ENABLE, START]\n"
    assert parse_rsc_queues(output) == ["interact-g", "mig"]


# --- parse_qstat_full ---


def test_parse_qstat_full_reads_every_field() -> None:
    record = """Job Id: 2435326.opbs
    Job_Name = train.sh
    job_state = F
    Exit_status = 0
    queue = short-g
"""
    [job] = parse_qstat_full(record)
    assert job == JobInfo(
        job_id="2435326.opbs",
        name="train.sh",
        state=PbsState.FINISHED,
        queue="short-g",
        exit_status=0,
    )


def test_parse_qstat_full_handles_multiple_records_and_ignores_junk_lines() -> None:
    record = """Job Id: 1.opbs
    job_state = Q
junk line with no equals
Job Id: 2.opbs
    job_state = R
"""
    jobs = parse_qstat_full(record)
    assert [job.job_id for job in jobs] == ["1.opbs", "2.opbs"]


def test_parse_qstat_full_empty_output_yields_nothing() -> None:
    assert parse_qstat_full("") == []


def test_parse_qstat_full_ignores_an_unrecognized_key() -> None:
    record = "Job Id: 1.opbs\n    resources_used.walltime = 00:01:00\n    queue = short-g\n"
    [job] = parse_qstat_full(record)
    assert job.queue == "short-g"


# --- build_qsub_flags ---


def test_build_qsub_flags_only_set_fields() -> None:
    assert build_qsub_flags(Resources()) == []
    flags = build_qsub_flags(
        Resources(queue="short-g", walltime="06:00:00", account="xg25g007", mem_gb=100, nodes=2)
    )
    assert flags == [
        "-q",
        "short-g",
        "-l",
        "walltime=06:00:00",
        "-W",
        "group_list=xg25g007",
        "-l",
        "select=2:mem=100gb",
    ]


# --- pbs_verdict ---


@pytest.mark.parametrize(
    ("state", "exit_code", "verdict"),
    [
        (None, None, "vanished"),
        ("R", None, "running"),
        ("Q", None, "running"),
        ("F", 0, "ok"),
        ("F", 1, "failed"),
        ("E", None, "unknown"),
        ("F", None, "unknown"),
    ],
)
def test_pbs_verdict(state: str | None, exit_code: int | None, verdict: str) -> None:
    assert pbs_verdict(state, exit_code) == verdict


# --- Pbs backend ---


def test_submit_returns_the_job_id_on_success() -> None:
    remote = machine_with("2435326.opbs\n")
    handle = Pbs().submit(
        remote, "/repo", script="job.sh", args=(), resources=Resources(queue="short-g")
    )
    assert handle == "2435326"
    assert remote.calls[-1] == ["bash", "-lc", "qsub -q short-g job.sh"]


def test_submit_raises_system_exit_when_qsub_rejects() -> None:
    remote = machine_with("qsub: Resource invalid\n")
    with pytest.raises(SystemExit, match="qsub failed"):
        Pbs().submit(remote, "/repo", script="job.sh", args=(), resources=Resources())


def test_submit_raises_system_exit_on_empty_output() -> None:
    remote = machine_with("")
    with pytest.raises(SystemExit, match="qsub failed"):
        Pbs().submit(remote, "/repo", script="job.sh", args=(), resources=Resources())


def test_jobs_lists_parsed_qstat_rows() -> None:
    remote = machine_with(_QSTAT_STANDARD)
    [state] = Pbs().jobs(remote, "/repo")
    assert state.handle == "2435326.opbs"
    assert state.verdict == "running"


def test_logs_reads_the_state_dir_log_file() -> None:
    remote = machine_with("job output\n")
    assert Pbs().logs(remote, "/repo", handle="2435326.opbs") == "job output\n"


def test_state_falls_back_to_autopsy_when_absent_from_both_queries() -> None:
    remote = machine_with("", "", "exit=0\n")
    state = Pbs().state(remote, "/repo", handle="9999.opbs")
    assert state.verdict == "ok"
    assert state.exit_code == 0


def test_state_uses_the_live_query_result_when_present() -> None:
    record = "Job Id: 2435326.opbs\n    job_state = R\n    queue = short-g\n"
    remote = machine_with(record)
    state = Pbs().state(remote, "/repo", handle="2435326.opbs")
    assert state.verdict == "running"


def test_states_falls_back_to_history_for_missing_handles() -> None:
    live = "Job Id: 1.opbs\n    job_state = R\n"
    history = "Job Id: 2.opbs\n    job_state = F\n    Exit_status = 0\n"
    remote = machine_with(live, history)
    states = Pbs().states(remote, "/repo", ["1.opbs", "2.opbs"])
    assert states["1.opbs"].verdict == "running"
    assert states["2.opbs"].verdict == "ok"


def test_states_with_no_handles_makes_no_call() -> None:
    remote = machine_with()
    assert Pbs().states(remote, "/repo", []) == {}


def test_autopsy_returns_ok_or_failed_from_the_exit_artifact() -> None:
    remote = machine_with("exit=1\n")
    state = Pbs().autopsy(remote, "/repo", handle="42.opbs")
    assert state.verdict == "failed"
    assert state.exit_code == 1


def test_autopsy_without_an_artifact_reads_as_vanished() -> None:
    remote = machine_with("")
    state = Pbs().autopsy(remote, "/repo", handle="42.opbs")
    assert state.verdict == "vanished"


def test_wait_polls_state_until_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = machine_with("Job Id: 1.opbs\n    job_state = F\n    Exit_status = 0\n")
    monkeypatch.setattr("mainboard.dispatch.schedulers.pbs.poll_until_done", lambda probe: probe())
    state = Pbs().wait(remote, "/repo", handle="1.opbs")
    assert state.verdict == "ok"


def test_stream_drains_the_log_until_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = machine_with("Job Id: 1.opbs\n    job_state = F\n    Exit_status = 0\n")
    monkeypatch.setattr(
        "mainboard.dispatch.schedulers.pbs.stream_until_done", lambda probe, drain: probe()
    )
    state = Pbs().stream(remote, "/repo", handle="1.opbs")
    assert state.verdict == "ok"


def test_cancel_calls_qdel() -> None:
    remote = machine_with("")
    Pbs().cancel(remote, "/repo", handle="1.opbs")
    assert remote.calls[-1] == ["bash", "-lc", "qdel 1.opbs"]


def test_revive_is_unsupported_for_a_site_managed_scheduler() -> None:
    remote = machine_with()
    with pytest.raises(SystemExit, match="site-managed"):
        Pbs().revive(remote, "/repo")


def test_queues_prefers_the_standard_listing() -> None:
    remote = machine_with("Queue\n------\nshort-g active\n")
    assert Pbs().queues(remote, "/repo") == ["short-g"]


def test_queues_falls_back_to_the_rsc_tree() -> None:
    remote = machine_with("", "interact-g |--[ENABLE, START]\n")
    assert Pbs().queues(remote, "/repo") == ["interact-g"]


def test_extract_job_id_falls_back_to_the_raw_stripped_output() -> None:
    assert _extract_job_id("no leading digits here") == "no leading digits here"
