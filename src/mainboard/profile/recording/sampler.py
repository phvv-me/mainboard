import os
import threading
from typing import TYPE_CHECKING

from ...core.constants.units import MILLISECONDS_PER_SECOND
from ...log import logger
from ..result.models import ProcessReading

if TYPE_CHECKING:
    from ...probe.units.gpu import GPU
    from .recorder import SpanRecorder


class DeviceSampler:
    """Read the selected device's telemetry of this process, polling while spans are open.

    gpu: the one device a session reads, None until the session selects it.
    evidence: whether any reading so far contained this process.
    label: the unit name of the last reading that did.
    """

    evidence: bool
    gpu: GPU | None
    interval_ms: int
    label: str
    stop_event: threading.Event
    thread: threading.Thread | None

    def __init__(self, *, interval_ms: int) -> None:
        self.gpu = None
        self.interval_ms = interval_ms
        self.evidence = False
        self.label = ""
        self.stop_event = threading.Event()
        self.thread = None

    def _poll(self, recorder: SpanRecorder) -> None:
        """Poll target-process device telemetry while at least one span is open."""
        interval = self.interval_ms / MILLISECONDS_PER_SECOND
        while not self.stop_event.wait(interval):
            self._sample_once(recorder)

    def _sample_once(self, recorder: SpanRecorder) -> None:
        """Attach one device reading of the target process to every span `recorder` has open."""
        with recorder.lock:
            frames = tuple(recorder.frames.values())
            name = frames[-1].path if frames else ""
        if not frames:
            return
        snapshot = self.snapshot(name)
        if snapshot is None:
            return
        with recorder.lock:
            for frame in frames:
                frame.samples.append(snapshot)

    def snapshot(self, name: str) -> ProcessReading | None:
        """Read one process-scoped device reading, only when it contains this process."""
        gpu = self.gpu
        if gpu is None:
            return None
        try:
            raw = gpu.snapshot(name=name)
        except OSError, RuntimeError:
            logger.warning("device sampler skipped a failed snapshot", exc_info=True)
            return None
        process = next((item for item in raw.processes if item.pid == os.getpid()), None)
        if process is None:
            return None
        self.evidence = True
        self.label = raw.unit_name
        return ProcessReading(
            unit_name=raw.unit_name,
            memory_used_bytes=process.used_bytes,
            gpu_util_pct=raw.utilization.gpu_pct,
            memory_util_pct=raw.utilization.memory_pct,
            power_w=raw.energy.power_w,
            temperature_c=raw.thermal.temperature_c,
        )

    def start(self, recorder: SpanRecorder) -> None:
        """Poll in the background, each reading going to every span `recorder` has open."""
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._poll, args=(recorder,), daemon=True, name="mainboard-profiler"
        )
        self.thread.start()

    def stop(self) -> None:
        """Stop and release the polling thread, when one runs."""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
