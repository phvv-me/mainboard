"""Incident evidence must survive the failure modes that misled the September 30 triage."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from plumbum import local
from pydantic import JsonValue

from mainboard.core.section import Verdict
from mainboard.diagnostics import Analysis, Diagnostics, Event, Probe, ProbeState
from mainboard.engines.compile.backend.process import Process
from mainboard.engines.compile.backend.tool import windows_launcher

NOW = datetime(2026, 9, 30, 7, tzinfo=UTC)


def reported(
    record: int,
    *,
    data: dict[str, str],
    provider: str = "Windows Error Reporting",
    event_id: int = 1001,
) -> dict[str, JsonValue]:
    return Event(
        log="Application",
        provider=provider,
        id=event_id,
        record_id=record,
        reported_at=NOW,
        data=data,
    ).model_dump(mode="json")


def test_resubmitted_dump_is_historical_and_duplicate_report_ids_collapse() -> None:
    probe = Probe(
        name="events-application",
        state=ProbeState.OK,
        collected_at=NOW,
        payload={
            "events": [
                reported(
                    record,
                    data={
                        "ReportId": "same-incident",
                        "AttachedFiles": r"C:\LiveKernelReports\WATCHDOG-20260903-0825.dmp",
                    },
                )
                for record in (10, 11, 12)
            ]
        },
    )
    analysis = Analysis((probe,), NOW - timedelta(days=7))
    events = analysis.events()
    assert len(events) == 1
    assert events[0].occurred_at is not None and events[0].occurred_at < analysis.since
    assert "1 historical incidents resubmitted" in analysis.findings(events)[0].detail


def test_report_without_original_timestamp_does_not_become_a_new_crash() -> None:
    event = Event.model_validate(reported(12, data={"ReportId": "unknown-age"})).dated()
    assert event.occurred_at is None
    finding = Analysis((), NOW - timedelta(days=7)).findings((event,))[0]
    assert "1 incident times unknown" in finding.detail


def test_unexpected_restart_does_not_establish_a_cause() -> None:
    event = Event.model_validate(
        reported(
            20, provider="Microsoft-Windows-Kernel-Power", event_id=41, data={"BugcheckCode": "0"}
        )
    ).dated()
    finding = Analysis((), NOW - timedelta(days=7)).findings((event,))[0]
    assert "not its cause" in finding.detail
    assert finding.verdict is Verdict.WARN


def test_missing_storage_counters_are_preserved_as_unknown() -> None:
    probe = Probe(
        name="storage",
        state=ProbeState.OK,
        collected_at=NOW,
        payload={"reliability": [{"health": "Healthy", "counters": {"ReadErrorsTotal": None}}]},
    )
    finding = Analysis((probe,), NOW).findings(())[0]
    assert "null counters are unknown" in finding.detail
    assert probe.payload["reliability"][0]["counters"]["ReadErrorsTotal"] is None


def test_pending_io_and_corruption_are_actionable_without_inventing_a_disk_failure() -> None:
    event = Event.model_validate(reported(30, provider="ESENT", event_id=532, data={})).dated()
    integrity = Probe(
        name="integrity",
        state=ProbeState.OK,
        collected_at=NOW,
        payload={"text": "The component store is repairable."},
    )
    findings = Analysis((integrity,), NOW).findings((event,))
    assert len(findings) == 2 and all(f.verdict is Verdict.FAIL for f in findings)
    assert "pending" in findings[0].detail
    assert "DISM" in findings[1].fix


def test_disconnected_or_virtual_wifi_does_not_count_as_duplicate_connected_wifi() -> None:
    probe = Probe(
        name="network",
        state=ProbeState.OK,
        collected_at=NOW,
        payload={
            "adapters": [
                {"wireless": True, "status": "Up"},
                {"wireless": True, "status": "Disconnected"},
                {"wireless": False, "status": "Up"},
            ]
        },
    )
    assert not Analysis((probe,), NOW).findings(())
    both = probe.model_copy(
        update={"payload": {"adapters": [{"wireless": True, "status": "Up"}] * 2}}
    )
    assert "2 physical Wi-Fi" in Analysis((both,), NOW).findings(())[0].detail


def test_partial_collection_and_raw_output_survive_export(tmp_path) -> None:
    def query(name: str, since: datetime) -> Probe:
        return Probe(
            name=name,
            state=ProbeState.TIMED_OUT,
            collected_at=NOW,
            stdout="partial response",
            error="deadline reached",
        )

    report = Diagnostics(query=query).collect()
    path = report.save(tmp_path)
    assert path.exists()
    assert len(report.probes) == len(Diagnostics.NAMES)
    assert all(f.verdict is Verdict.WARN for f in report.findings)
    assert "partial response" in (tmp_path / "devices.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("days", [0, -1, 366])
def test_bad_event_windows_are_refused_before_collection(days: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 365"):
        Diagnostics(days=days)


def test_diagnose_help_does_not_require_a_manifest(mb, tmp_path) -> None:
    ran = mb("diagnose", "--help", cwd=tmp_path)
    assert ran.code == 0 and "--days" in ran.out and "--out" in ran.out


def test_bad_window_is_an_actionable_cli_error_without_a_traceback(mb) -> None:
    ran = mb("diagnose", "--days", "0")
    assert ran.code == 1 and "between 1 and 365" in ran.said


def test_pcie_root_port_maps_to_gpu_descendants_instead_of_guessing_an_ssd() -> None:
    root = r"PCI\VEN_8086&DEV_A70D&SUBSYS_50001458&REV_01\3&ROOT"
    gpu = r"PCI\VEN_10DE&DEV_2C02\4&GPU"
    probe = Probe(
        name="devices",
        state=ProbeState.OK,
        collected_at=NOW,
        payload={
            "devices": [
                {"id": root, "name": "Intel PCIe root port", "properties": {}},
                {
                    "id": gpu,
                    "name": "NVIDIA RTX 5080",
                    "properties": {"DEVPKEY_Device_Parent": root},
                },
                {"id": r"PCI\VEN_15B7&DEV_5023\NVME", "name": "WD NVMe", "properties": {}},
            ]
        },
    )
    event = Event.model_validate(
        reported(
            42,
            provider="Microsoft-Windows-WHEA-Logger",
            event_id=17,
            data={"PrimaryDeviceName": root.rsplit("\\", 1)[0]},
        )
    ).dated()
    finding = Analysis((probe,), NOW).findings((event,))[0]
    assert "Intel PCIe root port" in finding.detail and "NVIDIA RTX 5080" in finding.detail
    assert "WD NVMe" not in finding.detail


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows lookup")
def test_windows_executables_resolve_when_parent_omits_or_empties_pathext() -> None:
    with local.env(PATHEXT="", PATH=str(Path(sys.executable).parent)):
        result = Process.capture(windows_launcher("python")["-c", "print('resolved')"], timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == "resolved"


def test_restart_report_time_is_not_promoted_to_the_time_of_the_freeze() -> None:
    event = Event.model_validate(
        reported(71, provider="Microsoft-Windows-Kernel-Power", event_id=41, data={})
    ).dated()
    assert event.occurred_at is None
    shutdown = Event.model_validate(
        reported(72, provider="EventLog", event_id=6008, data={})
    ).dated()
    assert shutdown.occurred_at is None


def test_io_failures_sort_ahead_of_generic_log_noise() -> None:
    warning = Event.model_validate(
        reported(90, provider="Microsoft-Windows-DistributedCOM", event_id=10016, data={})
    ).dated()
    stalled = Event.model_validate(reported(91, provider="ESENT", event_id=532, data={})).dated()
    findings = Analysis((), NOW).findings((warning, stalled))
    assert findings[0].section == "ESENT: 532"
    assert "permission changes" in findings[1].fix


def test_dirty_filesystem_is_actionable_even_when_component_store_is_clean() -> None:
    probe = Probe(
        name="integrity",
        state=ProbeState.OK,
        collected_at=NOW,
        payload={
            "text": "No component store corruption detected.",
            "filesystems": [{"DriveLetter": "C:", "DirtyBitSet": True}],
        },
    )
    finding = Analysis((probe,), NOW).findings(())[0]
    assert finding.verdict is Verdict.FAIL and "C:" in finding.detail
    assert "queued for restart" in finding.fix


def test_shutdown_occurrence_and_original_xml_survive_normalization() -> None:
    occurred = NOW - timedelta(minutes=30)
    event = Event(
        log="System",
        provider="EventLog",
        id=6008,
        record_id=101,
        reported_at=NOW,
        occurred_at=occurred,
        data={"0": "15:26:46", "1": "30/09/2026"},
        xml="<Event><EventData><Data>15:26:46</Data></EventData></Event>",
    )
    probe = Probe(
        name="events-system",
        state=ProbeState.OK,
        collected_at=NOW,
        payload={"events": [event.model_dump(mode="json")]},
    )
    retained = Analysis((probe,), NOW).events()[0]
    assert retained.occurred_at == occurred and retained.reported_at == NOW
    assert retained.xml == event.xml and retained.data["0"] == "15:26:46"


def test_queued_disk_repair_remains_visible_without_elevated_component_store_query() -> None:
    probe = Probe(
        name="integrity",
        state=ProbeState.UNAVAILABLE,
        collected_at=NOW,
        payload={"boot_checks": [r"autocheck autochk /p \??\C:", "autocheck autochk *"]},
        error="component-store health requires an elevated terminal",
    )
    findings = Analysis((probe,), NOW).findings(())
    assert any(f.section == "filesystem repair" and "next restart" in f.detail for f in findings)
    routine = probe.model_copy(
        update={"payload": {"boot_checks": ["autocheck autochk /k:C *", "autocheck autochk *"]}}
    )
    assert all(f.section != "filesystem repair" for f in Analysis((routine,), NOW).findings(()))


def test_boot_disk_check_reports_keep_results_and_are_not_deduplicated_as_wer() -> None:
    probe = Probe(
        name="events-application",
        state=ProbeState.OK,
        collected_at=NOW,
        payload={
            "events": [
                reported(
                    record,
                    provider="Microsoft-Windows-Wininit",
                    event_id=1001,
                    data={"ReportId": "shared-data", "wininit": "Windows checked the filesystem"},
                )
                for record in (201, 202)
            ]
        },
    )
    analysis = Analysis((probe,), NOW)
    events = analysis.events()
    assert len(events) == 2 and all(e.occurred_at == NOW for e in events)
    assert "startup repair output" in analysis.findings(events)[0].detail
