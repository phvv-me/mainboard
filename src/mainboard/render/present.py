import json
from typing import TYPE_CHECKING

from patos import value_dispatch

from ..core.errors import MissionError
from . import human, tabular
from .values import pairs_of, to_row

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .values import Node


def mode_of(*, json_mode: bool, human: bool) -> str | None:
    """The dispatch key `--json`/`--human` select, `None` for the compact default; not both."""
    if json_mode and human:
        raise MissionError("pass only one of --json or --human")
    return "json" if json_mode else ("human" if human else None)


def _record(
    payload: Mapping[str, Node], *, fields: Sequence[str], title: str, mode: str | None = None
) -> None:
    """Print one entity as `field<TAB>value` lines, the default: the reader is usually an agent.

    payload: the entity's fields, typically a model's `model_dump()`.
    fields: the field names to keep, every field when empty.
    """
    del mode, title
    # A record's `field value` header line says nothing its pairs do not.
    print(tabular.encode(pairs_of(to_row(payload), fields=fields or None)).partition("\n")[2])


record = value_dispatch(_record, kind="mode")


@record.register("json")
def _record_json(payload: Mapping[str, Node], *, fields: Sequence[str], title: str) -> None:
    print(json.dumps(_project(payload, fields), separators=(",", ":")))


@record.register("human")
def _record_human(payload: Mapping[str, Node], *, fields: Sequence[str], title: str) -> None:
    human.render_table(pairs_of(to_row(payload), fields=fields or None), title=title)


def _rows(
    payloads: Sequence[Mapping[str, Node]],
    *,
    fields: Sequence[str],
    title: str,
    mode: str | None = None,
) -> None:
    """Print many entities as a header line then tab-separated rows, the default.

    fields: the column names to keep, every field when empty.
    """
    del mode, title
    print(tabular.encode([to_row(payload) for payload in payloads], fields=fields or None))


rows = value_dispatch(_rows, kind="mode")


@rows.register("json")
def _rows_json(
    payloads: Sequence[Mapping[str, Node]], *, fields: Sequence[str], title: str
) -> None:
    print(json.dumps([_project(payload, fields) for payload in payloads], separators=(",", ":")))


@rows.register("human")
def _rows_human(
    payloads: Sequence[Mapping[str, Node]], *, fields: Sequence[str], title: str
) -> None:
    human.render_table(
        [to_row(payload) for payload in payloads], fields=fields or None, title=title
    )


def _project(payload: Mapping[str, Node], fields: Sequence[str]) -> dict[str, Node]:
    """`payload` narrowed to `fields`, unchanged when `fields` is empty."""
    return dict(payload) if not fields else {key: payload[key] for key in fields if key in payload}
