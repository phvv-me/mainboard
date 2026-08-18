from mainboard.dispatch.state import DownHost, Failed, Finished, MonitorReport


def test_changed_is_false_with_nothing_new() -> None:
    assert MonitorReport(running=2).changed is False


def test_changed_is_true_with_a_newly_finished_job() -> None:
    report = MonitorReport(finished=[Finished(handle="H1", target="gold")])
    assert report.changed is True


def test_changed_is_true_with_a_newly_failed_job() -> None:
    report = MonitorReport(failed=[Failed(handle="H1", target="gold", reason="oom")])
    assert report.changed is True


def test_down_host_carries_the_reason() -> None:
    host = DownHost(host="gold", reason="daemon down")
    assert host.reason == "daemon down"
