import json
from typing import TYPE_CHECKING

from patos import value_dispatch

from ..core.errors import MissionError
from . import human, tabular
from .values import pairs_of, to_row

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .values import Node


def mode_of(*, json_mode: bool, agent: bool) -> str | None:
    """The dispatch key `--json`/`--agent` select, `None` for the default human render.

    Raises when both flags are set, since the two compact modes are mutually exclusive.
    """
    if json_mode and agent:
        raise MissionError("pass only one of --json or --agent")
    return "json" if json_mode else ("agent" if agent else None)


def _record(
    payload: Mapping[str, Node], *, fields: Sequence[str], title: str, mode: str | None = None
) -> None:
    """Print one entity as a rich field/value table, the default when `mode` is unset.

    A single entity routinely has more fields than a terminal is wide, so it renders down
    its own rows (one field per line) rather than across columns.

    payload: the entity's field-name to value mapping (typically a model's `model_dump()`).
    fields: the field names to keep, every field when empty.
    title: the table's heading.
    """
    del mode
    human.render_table(pairs_of(to_row(payload), fields=fields or None), title=title)


record = value_dispatch(_record, kind="mode")


@record.register("json")
def _record_json(payload: Mapping[str, Node], *, fields: Sequence[str], title: str) -> None:
    print(json.dumps(_project(payload, fields), indent=2))


@record.register("agent")
def _record_agent(payload: Mapping[str, Node], *, fields: Sequence[str], title: str) -> None:
    print(tabular.encode(pairs_of(to_row(payload), fields=fields or None)))


def _rows(
    payloads: Sequence[Mapping[str, Node]],
    *,
    fields: Sequence[str],
    title: str,
    mode: str | None = None,
) -> None:
    """Print many entities as a rich table, the default when `mode` is unset.

    payloads: the entities, each a field-name to value mapping.
    fields: the column names to keep, every field when empty.
    title: the table's heading.
    """
    del mode
    human.render_table(
        [to_row(payload) for payload in payloads], fields=fields or None, title=title
    )


rows = value_dispatch(_rows, kind="mode")


@rows.register("json")
def _rows_json(
    payloads: Sequence[Mapping[str, Node]], *, fields: Sequence[str], title: str
) -> None:
    print(json.dumps([_project(payload, fields) for payload in payloads], indent=2))


@rows.register("agent")
def _rows_agent(
    payloads: Sequence[Mapping[str, Node]], *, fields: Sequence[str], title: str
) -> None:
    print(tabular.encode([to_row(payload) for payload in payloads], fields=fields or None))


def _project(payload: Mapping[str, Node], fields: Sequence[str]) -> dict[str, Node]:
    """`payload` narrowed to `fields`, unchanged when `fields` is empty."""
    return dict(payload) if not fields else {key: payload[key] for key in fields if key in payload}
