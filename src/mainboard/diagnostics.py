"""Bounded operating-system evidence collection, independent of a working workspace."""

import platform
import re
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from enum import StrEnum, auto
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from patos import FrozenModel
from plumbum import local
from plumbum.commands.processes import CommandNotFound, ProcessTimedOut
from pydantic import Field, ValidationError

from .core.section import Section, Verdict
from .engines.compile.backend.process import Process
from .render.values import Node

if TYPE_CHECKING:
    from collections.abc import Callable


class ProbeState(StrEnum):
    OK = auto()
    UNAVAILABLE = auto()
    TIMED_OUT = auto()
    FAILED = auto()
    TRUNCATED = auto()


class Probe(FrozenModel):
    name: str
    state: ProbeState
    collected_at: datetime
    elapsed_s: float = 0
    payload: dict[str, Node] = Field(default_factory=dict)
    stdout: str = ""
    error: str = ""


class Event(FrozenModel):
    """Reported time remains separate from the incident time inside a requeued WER report."""

    log: str
    provider: str
    id: int
    record_id: int
    reported_at: datetime
    occurred_at: datetime | None = None
    message: str = ""
    xml: str = ""
    data: dict[str, str] = Field(default_factory=dict)

    @property
    def reference(self) -> str:
        return f"{self.log}:{self.record_id}"

    @property
    def report_key(self) -> str:
        return (
            (self.data.get("ReportId") or self.reference)
            if self.provider == "Windows Error Reporting"
            else self.reference
        )

    def dated(self) -> Event:
        if self.id == 41 and self.provider == "Microsoft-Windows-Kernel-Power":
            return self
        if self.id == 6008 and self.provider == "EventLog":
            return self
        if self.provider != "Windows Error Reporting":
            return self.model_copy(update={"occurred_at": self.reported_at})
        # WER's event timestamp is a submission time. The retained dump's local timestamp is
        # incident evidence; without it the age is unknown, never assumed to be today's crash.
        match = re.search(r"-(\d{8})-(\d{4})\.dmp", self.data.get("AttachedFiles", ""), re.I)
        if match is None:
            return self
        try:
            date = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M").astimezone(UTC)
        except ValueError:
            return self
        return self.model_copy(update={"occurred_at": date})


class Finding(Section):
    evidence: tuple[str, ...] = ()


class DiagnosticReport(FrozenModel):
    schema_version: int = 1
    host: str
    platform: str
    since: datetime
    collected_at: datetime
    probes: tuple[Probe, ...]
    events: tuple[Event, ...]
    findings: tuple[Finding, ...]

    def save(self, destination: Path) -> Path:
        """Keep the raw probe responses beside a normalized, versioned report."""
        destination.mkdir(parents=True, exist_ok=True)
        report = destination / "report.json"
        report.write_text(self.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        for probe in self.probes:
            (destination / f"{probe.name}.json").write_text(
                probe.model_dump_json(indent=2), encoding="utf-8", newline="\n"
            )
        return report


class Analysis:
    """Explain what the collected evidence establishes, including what could not be checked."""

    def __init__(self, probes: tuple[Probe, ...], since: datetime) -> None:
        self.probes = probes
        self.since = since

    def events(self) -> tuple[Event, ...]:
        unique: dict[str, Event] = {}
        for probe in self.probes:
            records = probe.payload.get("events", [])
            if not isinstance(records, list):
                continue
            for record in records:
                event = Event.model_validate(record).dated()
                previous = unique.get(event.report_key)
                if previous is None or event.reported_at < previous.reported_at:
                    unique[event.report_key] = event
        return tuple(sorted(unique.values(), key=lambda event: event.reported_at))

    def _device_path(self, group: list[Event]) -> str:
        identities = {
            name.casefold()
            for event in group
            for key in ("PrimaryDeviceName", "SecondaryDeviceName")
            if (name := event.data.get(key, "")).startswith("PCI")
        }
        names: dict[str, str] = {}
        parents: dict[str, str] = {}
        for probe in self.probes:
            records = probe.payload.get("devices", []) if probe.name == "devices" else []
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict) or not isinstance(
                    identity := record.get("id"), str
                ):
                    continue
                identity = identity.casefold()
                names[identity] = str(record.get("name") or identity)
                properties = record.get("properties", {})
                parent = (
                    properties.get("DEVPKEY_Device_Parent")
                    if isinstance(properties, dict)
                    else None
                )
                if isinstance(parent, str):
                    parents[identity] = parent.casefold()
        matched = {
            identity
            for identity in names
            if any(identity.startswith(prefix) for prefix in identities)
        }
        for _ in range(len(parents)):
            descendants = {identity for identity, parent in parents.items() if parent in matched}
            if descendants <= matched:
                break
            matched |= descendants
        if matched:
            return "; mapped PCIe path: " + ", ".join(
                dict.fromkeys(names[identity] for identity in sorted(matched))
            )
        return "; PCIe identity not mapped by the available evidence"

    def findings(self, events: tuple[Event, ...]) -> tuple[Finding, ...]:
        findings = [
            Finding(
                section=f"probe: {probe.name}",
                verdict=Verdict.WARN,
                detail=f"{probe.state}: {probe.error or 'collection limit reached'}",
                evidence=(probe.name,),
                fix="rerun from an elevated terminal"
                if probe.name in ("integrity", "dumps") and probe.state is ProbeState.UNAVAILABLE
                else "",
            )
            for probe in self.probes
            if probe.state is not ProbeState.OK
        ]
        groups: dict[tuple[str, int], list[Event]] = {}
        for event in events:
            groups.setdefault((event.provider, event.id), []).append(event)
        for (provider, event_id), group in groups.items():
            refs = tuple(event.reference for event in group)
            last = max(event.reported_at for event in group).isoformat()
            detail = f"{len(group)} distinct records; last reported {last}"
            fix = "inspect the retained event data and nearby incidents before changing settings"
            verdict = Verdict.WARN
            if provider == "Microsoft-Windows-Kernel-Power" and event_id == 41:
                detail += "; unexpected restarts record the interruption, not its cause"
                fix = (
                    "correlate storage, WHEA, exhaustion, application hangs and live "
                    "dumps before each restart"
                )
            elif provider == "Windows Error Reporting":
                old = sum(
                    event.occurred_at is not None and event.occurred_at < self.since
                    for event in group
                )
                unknown = sum(event.occurred_at is None for event in group)
                detail += (
                    f"; {old} historical incidents resubmitted; {unknown} incident times unknown"
                )
                fix = (
                    "use original incident times and report IDs; analyze current dumps "
                    "before blaming a driver"
                )
            elif provider in ("Microsoft-Windows-Wininit", "Chkdsk"):
                detail += "; disk-check result retained, including any startup repair output"
                fix = "inspect the check result and compare filesystem state after the repair"
            elif provider == "ESENT" and event_id in (508, 510, 533):
                detail += "; unusually slow disk I/O is reported"
                verdict = Verdict.FAIL
                fix = (
                    "inspect drive latency, storage drivers, firmware, filters and "
                    "dump wait chains"
                )
            elif provider == "ESENT" and event_id == 532:
                detail += "; disk I/O remained pending; this can stall multiple applications"
                verdict = Verdict.FAIL
                fix = (
                    "inspect drive latency, storage drivers, firmware, filters and "
                    "dump wait chains"
                )
            elif provider == "Microsoft-Windows-Resource-Exhaustion-Detector" and event_id == 2004:
                detail += "; Windows recorded resource exhaustion"
                verdict = Verdict.FAIL
                fix = (
                    "identify the consumers and commit limit recorded in this event; "
                    "current free RAM is insufficient evidence"
                )
            elif provider == "Microsoft-Windows-DistributedCOM" and event_id == 10016:
                detail += "; permission retries can be expected Windows behavior (KB4022522)"
                fix = (
                    "investigate an associated app failure; "
                    "the warning alone does not justify permission changes"
                )
            elif provider == "Microsoft-Windows-WHEA-Logger":
                detail += self._device_path(group)
                fix = (
                    "compare vendor/device IDs and location paths with devices.json; a "
                    "root-port warning alone does not identify an SSD"
                )
            findings.append(
                Finding(
                    section=f"{provider}: {event_id}",
                    verdict=verdict,
                    detail=detail,
                    fix=fix,
                    evidence=refs,
                )
            )
        for probe in self.probes:
            if probe.name == "storage" and probe.state is ProbeState.OK:
                findings.append(
                    Finding(
                        section="storage",
                        verdict=Verdict.WARN,
                        detail=(
                            "drive inventory and available reliability counters retained; "
                            "Healthy status does not exclude intermittent faults; null "
                            "counters are unknown"
                        ),
                        evidence=(probe.name,),
                        fix="correlate these counters with timed storage events and dumps",
                    )
                )
            if probe.name == "network" and probe.state is ProbeState.OK:
                adapters = probe.payload.get("adapters", [])
                wireless = (
                    [
                        adapter
                        for adapter in adapters
                        if isinstance(adapter, dict)
                        and adapter.get("status") == "Up"
                        and adapter.get("wireless") is True
                    ]
                    if isinstance(adapters, list)
                    else []
                )
                if len(wireless) > 1:
                    findings.append(
                        Finding(
                            section="network",
                            verdict=Verdict.WARN,
                            detail=(
                                f"{len(wireless)} physical Wi-Fi adapters connected; "
                                "default routes "
                                f"retained for comparison"
                            ),
                            fix=(
                                "test with one intended Wi-Fi adapter active; preserve a working "
                                "connection"
                            ),
                            evidence=(probe.name,),
                        )
                    )
            if probe.name == "integrity":
                volumes = probe.payload.get("filesystems", [])
                if isinstance(volumes, list):
                    for volume in volumes:
                        if isinstance(volume, dict) and volume.get("DirtyBitSet") is True:
                            findings.append(
                                Finding(
                                    section="filesystem",
                                    verdict=Verdict.FAIL,
                                    detail=(
                                        f"{volume.get('DriveLetter') or volume.get('DeviceID')} "
                                        "is marked dirty by Windows"
                                    ),
                                    fix=(
                                        "run the Windows filesystem check; complete any repair "
                                        "queued for restart and inspect its result"
                                    ),
                                    evidence=(probe.name,),
                                )
                            )
                checks = probe.payload.get("boot_checks", [])
                if isinstance(checks, list) and any(
                    isinstance(check, str)
                    and check.startswith("autocheck autochk ")
                    and "\\??\\" in check
                    for check in checks
                ):
                    findings.append(
                        Finding(
                            section="filesystem repair",
                            verdict=Verdict.WARN,
                            detail="Windows has a filesystem check queued for the next restart",
                            fix="save work, restart and inspect the completed disk check result",
                            evidence=(probe.name,),
                        )
                    )
            if probe.name == "integrity" and probe.state is ProbeState.OK:
                text = probe.payload.get("text", "")
                if isinstance(text, str) and "component store is repairable" in text.lower():
                    findings.append(
                        Finding(
                            section="Windows integrity",
                            verdict=Verdict.FAIL,
                            detail="Windows reports repairable component-store corruption",
                            fix=(
                                "DISM /Online /Cleanup-Image /RestoreHealth, then sfc /scannow; "
                                "verify both results"
                            ),
                            evidence=(probe.name,),
                        )
                    )
        order = {Verdict.FAIL: 0, Verdict.WARN: 1, Verdict.PASS: 2}
        return tuple(sorted(findings, key=lambda finding: order[finding.verdict]))


class Diagnostics:
    """Read-only incident capture. Every probe has a deadline and keeps its own failure."""

    NAMES = (
        "events-system",
        "events-application",
        "inventory",
        "storage",
        "network",
        "devices",
        "dumps",
        "integrity",
    )
    TIMEOUT = 45
    LIMIT = 2000

    def __init__(
        self, *, days: int = 7, query: Callable[[str, datetime], Probe] | None = None
    ) -> None:
        if not 1 <= days <= 365:
            raise ValueError("days must be between 1 and 365")
        self.days = days
        self.query = query or self._query

    def collect(self) -> DiagnosticReport:
        now = datetime.now(UTC)
        since = now - timedelta(days=self.days)
        if platform.system() == "Windows" or self.query != self._query:
            with ThreadPoolExecutor(max_workers=3) as pool:
                probes = tuple(pool.map(lambda name: self.query(name, since), self.NAMES))
        else:
            probes = (
                Probe(
                    name="operating-system",
                    state=ProbeState.UNAVAILABLE,
                    collected_at=now,
                    error=f"incident log collection is not implemented for {platform.system()}",
                ),
            )
        analysis = Analysis(probes, since)
        events = analysis.events()
        return DiagnosticReport(
            host=platform.node(),
            platform=platform.platform(),
            since=since,
            collected_at=now,
            probes=probes,
            events=events,
            findings=analysis.findings(events),
        )

    def _query(self, name: str, since: datetime) -> Probe:
        started = monotonic()
        collected = datetime.now(UTC)
        script = (
            f"$probe='{name}';$since=[datetime]'{since.isoformat()}';$limit={self.LIMIT};\n"
            + Path(__file__).with_suffix(".ps1").read_text(encoding="utf-8")
        )
        encoded = b64encode(script.encode("utf-16le")).decode("ascii")
        try:
            command = local["powershell.exe"][
                "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded
            ]
            result = Process.capture(command, timeout=self.TIMEOUT)
        except ProcessTimedOut:
            return Probe(
                name=name,
                state=ProbeState.TIMED_OUT,
                collected_at=collected,
                elapsed_s=monotonic() - started,
                error=f"no answer within {self.TIMEOUT}s",
            )
        except (CommandNotFound, OSError) as error:
            return Probe(
                name=name,
                state=ProbeState.UNAVAILABLE,
                collected_at=collected,
                elapsed_s=monotonic() - started,
                error=str(error),
            )
        try:
            response = Probe.model_validate_json(result.stdout)
        except ValidationError as error:
            return Probe(
                name=name,
                state=ProbeState.FAILED,
                collected_at=collected,
                elapsed_s=monotonic() - started,
                stdout=result.stdout,
                error=f"invalid probe response: {error}; {result.stderr}",
            )
        return response.model_copy(
            update={"elapsed_s": monotonic() - started, "stdout": result.stdout}
        )
