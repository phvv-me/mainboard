# The one logger the tool and every workspace using it write through: `from mb import logger`.
#
# Loguru's call shape on structlog's engine. Positional arguments format the message the way
# loguru does (`logger.info("took {}s", 3)`), and keywords are fields rather than prose
# (`logger.info("fetched", host="gold", files=12)`), so a line can be filtered, timed and counted
# without a regex. Everything goes to stderr, leaving stdout to what a verb prints.
#
# THE OUTPUT. On a terminal, one readable line per event. Anywhere else (a dispatched job's
# captured log, CI, a pipe), one JSON object per line carrying where it was logged from, the form
# every reader of a job's log parses. `MB_LOG_FORMAT` (`console` or `json`) overrides the guess and
# `MB_LOG_LEVEL` (`debug` .. `critical`, `info` by default) sets the floor; legacy names are read.
#
# SINKS. An event whose context names a sink (`bound_contextvars(mb_sink=key)`) is also handed to
# the callable registered under that key, which is how a trial keeps every line logged while it
# runs, from any library, in its own record.
#
# A process that configured structlog itself keeps its configuration: this one is applied only
# when nothing else was, so importing the tool never takes over a host application's logging.

import sys
from collections.abc import Callable, MutableMapping
from typing import Any

import structlog
from structlog.processors import CallsiteParameter
from structlog.typing import FilteringBoundLogger

from .core.project import Project

type EventDict = MutableMapping[str, Any]

# The context key naming an event's sink, and every registered sink by its key.
SINK = "mb_sink"
sinks: dict[str, Callable[[EventDict], None]] = {}

# The methods whose positional arguments are formatted loguru's way.
_METHODS = ("debug", "info", "warning", "error", "critical", "exception")


class _Stderr:
    """Writes each line to whatever `sys.stderr` is at that moment (a test capture swaps it)."""

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    debug = info = warning = error = critical = msg


def _route(_: object, __: str, event: EventDict) -> EventDict:
    """Hand an event naming a sink to that sink, dropping the name so no renderer shows it."""
    if (sink := sinks.get(event.pop(SINK, ""))) is not None:
        sink(dict(event))
    return event


def _braced(level: int) -> type[FilteringBoundLogger]:
    """structlog's level-filtered logger, formatting positional arguments with `str.format`."""
    base = structlog.make_filtering_bound_logger(level)

    def method(name: str) -> Callable[..., Any]:
        def call(self: FilteringBoundLogger, event: str, *args: object, **fields: object) -> Any:
            return getattr(base, name)(self, event.format(*args) if args else event, **fields)

        return call

    return type("Logger", (base,), {name: method(name) for name in _METHODS})


def configure(level: str = "", output: str = "") -> None:
    """Configure structlog for this process: `level` and `output` (`console` or `json`), each
    read from the environment when empty, else `info` and whatever stderr is (a terminal or not).
    """
    project = Project()
    level = (level or project.variable("LOG_LEVEL").read() or "info").lower()
    output = output or project.variable("LOG_FORMAT").read()
    json = output == "json" if output else not sys.stderr.isatty()
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _route,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    rendered: list[Any] = (
        [
            structlog.processors.CallsiteParameterAdder(
                [CallsiteParameter.QUAL_MODULE, CallsiteParameter.LINENO],
                additional_ignores=[__name__],
            ),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
        if json
        else [structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())]
    )
    structlog.configure(
        processors=[*shared, *rendered],
        wrapper_class=_braced(structlog.processors.NAME_TO_LEVEL[level]),
        logger_factory=lambda *_: _Stderr(),
        cache_logger_on_first_use=False,
    )


if not structlog.is_configured():
    configure()

logger: FilteringBoundLogger = structlog.get_logger()
