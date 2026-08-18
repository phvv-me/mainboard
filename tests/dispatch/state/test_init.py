from mainboard.dispatch import state


def test_state_package_reexports_the_public_surface() -> None:
    assert {
        "Cache",
        "RunRecord",
        "History",
        "HistoryEvent",
        "MonitorReport",
        "ReconcileRow",
        "connect",
    } <= set(state.__all__)
    assert state.Cache is not None
