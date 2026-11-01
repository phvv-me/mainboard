"""The OTLP JSON shape of a spool line: one `ExportLogsServiceRequest` holding one log record.

The models are the wire format itself, so a line is validated on the way in and on the way out
instead of being built or read as loose dicts.
"""

from typing import Annotated, Self

from patos import FrozenModel
from pydantic import ConfigDict, PlainSerializer
from pydantic.alias_generators import to_camel

from .identity import Attribute, Scalar

_SCHEMA = "1"
"""The line format revision; a reader refuses a line written under another."""

_SCOPE = "mb.spool"
_SERVICE = "mb"
_FIELDS: dict[type, str] = {
    bool: "bool_value",
    float: "double_value",
    int: "int_value",
    str: "string_value",
}
"""The `AnyValue` field each scalar type fills; attributes arrive here as exact types."""

type _Int64 = Annotated[int, PlainSerializer(str, return_type=str)]
"""An OTLP int64, which OTLP JSON writes as a decimal string and a reader takes either way."""


class _Wire(FrozenModel):
    """An OTLP JSON object: camelCase keys, and a key left out when it is unset."""

    model_config = ConfigDict(alias_generator=to_camel, extra="forbid")


class _Array(_Wire):
    values: list[Value]


class Value(_Wire):
    """An OTLP `AnyValue` of the kinds an attribute can hold; bytes and maps are refused."""

    string_value: str | None = None
    bool_value: bool | None = None
    int_value: _Int64 | None = None
    double_value: float | None = None
    array_value: _Array | None = None

    @classmethod
    def of(cls, value: Attribute) -> Self:
        """The `AnyValue` holding `value`."""
        if isinstance(value, str | bool | int | float):
            return cls.model_validate({_FIELDS[type(value)]: value})
        return cls(array_value=_Array(values=[cls.of(item) for item in value]))

    @staticmethod
    def _scalar(item: Value) -> Scalar:
        for held in (item.string_value, item.bool_value, item.int_value, item.double_value):
            if held is not None:
                return held
        raise ValueError("an attribute value holds nothing")

    def read(self) -> Attribute:
        """The value this holds."""
        if self.array_value is not None:
            return [self._scalar(item) for item in self.array_value.values]
        return self._scalar(self)


class KeyValue(_Wire):
    key: str
    value: Value


class Record(_Wire):
    time_unix_nano: str
    severity_number: int
    event_name: str
    body: Value
    attributes: list[KeyValue]


class _Scope(_Wire):
    name: str
    version: str


class _ScopeLogs(_Wire):
    scope: _Scope
    log_records: list[Record]


class _Resource(_Wire):
    attributes: list[KeyValue]


class _ResourceLogs(_Wire):
    resource: _Resource
    scope_logs: list[_ScopeLogs]


class Request(_Wire):
    """One `ExportLogsServiceRequest` as mb spools it: one record of mb's service and scope."""

    resource_logs: list[_ResourceLogs]

    @classmethod
    def of(cls, record: Record) -> Self:
        """The request that carries `record` alone."""
        service = KeyValue(key="service.name", value=Value.of(_SERVICE))
        scope = _Scope(name=_SCOPE, version=_SCHEMA)
        return cls(
            resource_logs=[
                _ResourceLogs(
                    resource=_Resource(attributes=[service]),
                    scope_logs=[_ScopeLogs(scope=scope, log_records=[record])],
                )
            ]
        )

    @property
    def record(self) -> Record:
        """The one record; `ValueError` for a request of another shape or schema."""
        (resource,) = self.resource_logs
        (logs,) = resource.scope_logs
        (record,) = logs.log_records
        if (logs.scope.name, logs.scope.version) != (_SCOPE, _SCHEMA):
            raise ValueError(
                f"spool line is {logs.scope.name!r} {logs.scope.version!r}, "
                f"this mb reads {_SCOPE!r} {_SCHEMA!r}"
            )
        return record

    def line(self) -> str:
        """The request as one JSON line."""
        return self.model_dump_json(by_alias=True, exclude_none=True, ensure_ascii=False)
