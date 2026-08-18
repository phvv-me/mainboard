from mainboard.dispatch.state import ReconcileRow


def test_reconcile_row_defaults() -> None:
    row = ReconcileRow(handle="H1", script="job.sh", submitted_at="t0", verdict="ok")
    assert row.name == ""
    assert row.state is None
    assert row.exit_code is None
