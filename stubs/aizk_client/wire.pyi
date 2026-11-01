# The wire models of `aizk_client.wire` that `mainboard.memory.aizk` reads: what a keep answers and
# the status report.

from uuid import UUID

from patos import FrozenModel

class WriteResult(FrozenModel):
    id: UUID

class ArtifactReceipt(FrozenModel):
    artifact_id: UUID
    content_id: UUID

class UploadTicketAccepted(FrozenModel):
    upload_url: str
    expires_seconds: int

type KeepResult = WriteResult | ArtifactReceipt | UploadTicketAccepted

class UsageSummary(FrozenModel):
    finds: int
    keeps: int

class UsageStatus(FrozenModel):
    lifetime: UsageSummary

class ProcessingStatus(FrozenModel):
    state: str

class CallerStatus(FrozenModel):
    name: str | None
    username: str | None

class StatusReport(FrozenModel):
    caller: CallerStatus
    usage: UsageStatus
    processing: ProcessingStatus
