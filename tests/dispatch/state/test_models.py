import pytest

from mainboard.dispatch import state
from mainboard.dispatch.state import DownHost, Failed, Finished, MonitorReport, ReconcileRow


@pytest.mark.parametrize(
    ("report", "changed"),
    [
        (MonitorReport(running=2), False),
        (MonitorReport(finished=[Finished(handle="H1", target="gold")]), True),
        (MonitorReport(failed=[Failed(handle="H1", target="gold", reason="oom")]), True),
        (
            MonitorReport(unreachable_hosts=[DownHost(host="gold", reason="daemon down")]),
            False,
        ),
    ],
)
def test_a_sweep_reads_as_changed_exactly_when_it_harvested_a_terminal_job(
    report: MonitorReport, changed: bool
) -> None:
    """A host that could not be probed is not news, so it must never wake a cron tick."""
    assert report.changed is changed


def test_the_state_package_reexports_the_value_objects_a_reconcile_builds() -> None:
    assert {
        "Cache",
        "RunRecord",
        "History",
        "MonitorReport",
        "ReconcileRow",
        "connect",
    } <= set(state.__all__)
    row = ReconcileRow(handle="H1", script="job.sh", submitted_at="t0", verdict="ok")
    assert (row.name, row.state, row.exit_code) == ("", None, None)
