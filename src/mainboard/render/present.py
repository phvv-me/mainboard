import json
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter
from patos import FrozenModel, value_dispatch
from pydantic import ConfigDict

from ..core.errors import MissionError
from ..log import logger
from . import human
from .enums import Audience
from .terminal import Terminal
from .values import encode, pairs_of, to_row

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .values import Node


def mode_of(*, json_mode: bool) -> str | None:
    """The dispatch key: `--json`, else rich tables for a person, else `None` for an agent.

    `None` names the compact rows.
    """
    if json_mode:
        return "json"
    return "human" if Terminal.current().audience is Audience.HUMAN else None


def _record(
    payload: Mapping[str, Node], *, fields: Sequence[str], title: str, mode: str | None = None
) -> None:
    """Print one entity as `field<TAB>value` lines, the default: the reader is usually an agent.

    payload: the entity's fields, typically a model's `model_dump()`.
    fields: the field names to keep, every field when empty.
    """
    del mode, title
    # A record's `field value` header line says nothing its pairs do not.
    logger.result(encode(pairs_of(to_row(payload), fields=fields or None)).partition("\n")[2])


record = value_dispatch(_record, kind="mode")


@record.register("json")
def _record_json(payload: Mapping[str, Node], *, fields: Sequence[str], title: str) -> None:
    logger.result(json.dumps(_project(payload, fields), separators=(",", ":")))


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
    logger.result(encode([to_row(payload) for payload in payloads], fields=fields or None))


rows = value_dispatch(_rows, kind="mode")


@rows.register("json")
def _rows_json(
    payloads: Sequence[Mapping[str, Node]], *, fields: Sequence[str], title: str
) -> None:
    logger.result(
        json.dumps([_project(payload, fields) for payload in payloads], separators=(",", ":"))
    )


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


# The values a compact record leaves out: they say nothing an absent field does not.
_EMPTY: tuple[object, ...] = (None, "", [], {}, ())


@Parameter(name="*")
class Output(FrozenModel):
    """How a verb prints its document: rich tables for a person, compact rows for an agent, JSON.

    An agent's default is what it reads cheapest: a header line of field names, then one
    tab-separated line per row, no colour, no box drawing.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    json_: Annotated[bool, Parameter(name="--json")] = False
    """print canonical JSON instead of the compact rows."""
    fields: str = ""
    """a comma-separated projection over the printed fields."""

    @property
    def mode(self) -> str | None:
        """The render key, `None` for an agent's compact rows."""
        return mode_of(json_mode=self.json_)

    def print_document(self, text: str, payload: Mapping[str, Node], *, title: str) -> None:
        """Print `text` as written, or `payload` as JSON, the only thing `--fields` projects."""
        if self.json_:
            self.print_record(payload, title=title)
        elif self.fields:
            raise MissionError("--fields projects the JSON document; add --json")
        else:
            logger.result(text)

    def print_record(self, payload: Mapping[str, Node], *, title: str) -> None:
        """Print one entity; the compact rows leave out its empty fields."""
        if self.mode is None and not self.fields:
            payload = {key: value for key, value in payload.items() if value not in _EMPTY}
        record(payload, mode=self.mode, fields=self.projection(), title=title)

    def print_rows(
        self, payloads: Sequence[Mapping[str, Node]], *, title: str, columns: Sequence[str] = ()
    ) -> None:
        """Print many entities, `columns` keeping an empty table's heading unless projected.

        The compact rows leave out a column empty in every row: it tells an agent nothing
        and costs a tab per row. `--fields` keeps exactly the columns it names.
        """
        fields = self.projection(columns)
        if self.mode is None and not self.fields and payloads:
            flat = [to_row(payload) for payload in payloads]
            fields = [
                name
                for name in fields or flat[0]
                if any(row.get(name) not in (None, "") for row in flat)
            ]
        rows(payloads, mode=self.mode, fields=fields, title=title)

    def projection(self, default: Sequence[str] = ()) -> Sequence[str]:
        """The `--fields` names, trimmed and blanks dropped, `default` when none were given."""
        return tuple(part.strip() for part in self.fields.split(",") if part.strip()) or default
