from mainboard.dispatch.state import RunRecord
from mainboard.fleet.reliability import reliability


def test_reliability_charges_the_machine_not_the_job() -> None:
    def run(
        target: str, *, verdict: str, exit_code: int | None = None, evidence: str = ""
    ) -> RunRecord:
        return RunRecord(
            handle=f"{target}-{verdict}-{exit_code}-{evidence}",
            target=target,
            kind="vast",
            script="job.sh",
            args="",
            submitted_at="2026-10-24T00:00:00Z",
            verdict=verdict,
            exit_code=exit_code,
            evidence=evidence,
        )

    rows = reliability(
        [
            run("vast", verdict="ok", exit_code=0),
            run(
                "vast", verdict="failed", exit_code=1
            ),  # the job's own failure: the machine delivered
            run("vast", verdict="failed", evidence="not_started"),  # the landing never started it
            run("vast", verdict="vanished"),  # taken back under it
            run("vast", verdict="cancelled"),  # judges nothing
            run("held", verdict="running"),
        ]
    )
    vast = next(row for row in rows if row.target == "vast")
    assert (vast.ran, vast.unstarted, vast.lost, vast.unsettled) == (2, 1, 1, 1)
    assert vast.delivered == 0.5 and rows[-1].target == "held" and rows[-1].delivered is None
