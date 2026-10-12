from datetime import UTC, datetime

from duckdb import DuckDBPyConnection
from patos import FrozenModel
from sqlalchemy import select

from ...state import schema
from ...state.lake import Lake, staged
from .models import DeviceResult, Evidence


class NotFound(FrozenModel):
    evidence_id: str


class EvidenceStore:
    def __init__(self, lake: Lake) -> None:
        self.session = lake.session()

    def ingest[T: DeviceResult](self, evidence: Evidence[T]) -> str:
        evidence_id = evidence.key()
        row = {
            "ts": datetime.now(UTC),
            "evidence_id": evidence_id,
            "artifact_sha256": evidence.artifact.sha256,
            "kind": evidence.artifact.kind.value,
            "provider": evidence.provider,
            "tool_version": evidence.tool_version,
            "operation": evidence.operation,
            "params_json": dict(evidence.params),
            "analysis_profile_digest": evidence.analysis_profile.digest(),
            "coverage": evidence.coverage.value,
            "limitations_json": [item.model_dump(mode="json") for item in evidence.limitations],
            "unresolved_json": [item.model_dump(mode="json") for item in evidence.unresolved],
            "result_json": evidence.result.model_dump(mode="json"),
            "record_json": evidence.model_dump(mode="json"),
            "dropped": False,
        }

        def append(connection: DuckDBPyConnection) -> None:
            # The existence check and append share one statement and the session's write lock.
            table, current = schema.re_evidence, schema.re_evidence_current
            with staged(table, [row]) as source:
                self.session.lake.execute(
                    connection,
                    f"INSERT INTO {table.fullname} BY NAME {source} WHERE NOT EXISTS "
                    f"(SELECT 1 FROM {current.fullname} WHERE evidence_id = ?)",
                    [evidence_id],
                )

        self.session.run(append)
        return evidence_id

    def read(self, evidence_id: str) -> Evidence[DeviceResult] | NotFound:
        current = schema.re_evidence_current
        rows = self.session.rows(
            select(current.c.record_json).where(current.c.evidence_id == evidence_id)
        )
        if not rows:
            return NotFound(evidence_id=evidence_id)
        return Evidence[DeviceResult].model_validate_json(rows[0][0])

    def query(
        self,
        *,
        artifact_sha256: str = "",
        operation: str = "",
        provider: str = "",
    ) -> tuple[Evidence[DeviceResult], ...]:
        current = schema.re_evidence_current
        filters = {
            "artifact_sha256": artifact_sha256,
            "operation": operation,
            "provider": provider,
        }
        statement = (
            select(current.c.record_json)
            .where(*(current.c[key] == value for key, value in filters.items() if value))
            .order_by(current.c.evidence_id)
        )
        return tuple(
            Evidence[DeviceResult].model_validate_json(row[0])
            for row in self.session.rows(statement)
        )
