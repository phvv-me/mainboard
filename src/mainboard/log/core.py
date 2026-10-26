# The one logger the tool and every workspace using it write through: `from mb import logger`.
#
# Loguru's call shape on structlog's engine. Positional arguments format the message the way
# loguru does (`logger.info("took {}s", 3)`), and keywords are fields rather than prose
# (`logger.info("fetched", host="gold", files=12)`), so a line can be filtered, timed and counted
# without a regex. Every event goes to stderr. A verb's result is the one thing on stdout, and
# it is written through the same logger (`logger.result`), as is a question to a person
# (`logger.prompt`), so nothing in the tool prints around it.
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

import logging
import sys
from collections.abc import Callable

import structlog
from patos import FrozenModel
from structlog.processors import NAME_TO_LEVEL, CallsiteParameter
from structlog.typing import EventDict, Processor

from ..core.project import Project
from .constants import SINK

sinks: dict[str, Callable[[EventDict], None]] = {}

# The method each numeric `logging` level is logged through.
_NAMES = {
    logging.CRITICAL: "critical",
    logging.ERROR: "error",
    logging.WARNING: "warning",
    logging.INFO: "info",
    logging.DEBUG: "debug",
}


class Logger(structlog.BoundLoggerBase):
    """mb's logger: structlog's levels with loguru's positional formatting, plus the two streams a
    verb speaks on besides events. The configured floor drops what is below it (`_Floor`)."""

    def _emit(
        self, method: str, *, event: str, args: tuple[object, ...], fields: dict[str, object]
    ) -> None:
        self._proxy_to_logger(method, event.format(*args) if args else event, **fields)

    def critical(self, event: str, *args: object, **fields: object) -> None:
        self._emit("critical", event=event, args=args, fields=fields)

    def debug(self, event: str, *args: object, **fields: object) -> None:
        self._emit("debug", event=event, args=args, fields=fields)

    def error(self, event: str, *args: object, **fields: object) -> None:
        self._emit("error", event=event, args=args, fields=fields)

    def exception(self, event: str, *args: object, **fields: object) -> None:
        """An error carrying the exception being handled."""
        self._emit("error", event=event, args=args, fields={"exc_info": True, **fields})

    def info(self, event: str, *args: object, **fields: object) -> None:
        self._emit("info", event=event, args=args, fields=fields)

    def log(self, level: int, event: str, *args: object, **fields: object) -> None:
        """An event at the numeric `logging` level `level`."""
        self._emit(_NAMES[level], event=event, args=args, fields=fields)

    def prompt(self, text: str) -> None:
        """Ask a person `text` on stderr, the cursor left on its line."""
        sys.stderr.write(text)
        sys.stderr.flush()

    def result(self, text: str = "", *, end: str = "\n") -> None:
        """Write a verb's result to stdout, the stream its caller reads."""
        sys.stdout.write(f"{text}{end}")
        sys.stdout.flush()

    def warning(self, event: str, *args: object, **fields: object) -> None:
        self._emit("warning", event=event, args=args, fields=fields)

    fatal = critical
    warn = warning


class _Floor(FrozenModel):
    """Drops an event below `level`, the floor `configure` set."""

    level: int

    def __call__(self, _: object, method: str, event: EventDict) -> EventDict:
        if NAME_TO_LEVEL[method] < self.level:
            raise structlog.DropEvent
        return event


class _Stderr:
    """Writes each line to whatever `sys.stderr` is at that moment (a test capture swaps it)."""

    def msg(self, message: str) -> None:
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()

    debug = info = warning = error = critical = msg


def _route(_: object, __: str, event: EventDict) -> EventDict:
    """Hand an event naming a sink to that sink, dropping the name so no renderer shows it."""
    if (sink := sinks.get(event.pop(SINK, ""))) is not None:
        sink(dict(event))
    return event


def _line(_: object, __: str, event: EventDict) -> str:
    """One short line: the event, its fields as `key=value`, a level word only above info."""
    level = event.pop("level", "info")
    event.pop("timestamp", None)
    text = str(event.pop("event", ""))
    fields = "".join(f" {key}={value}" for key, value in event.items())
    return f"{'' if level in {'debug', 'info'} else level + ': '}{text}{fields}"


def configure(level: str = "", *, output: str = "") -> None:
    """Configure structlog for this process at `level`, rendering as `output`.

    `output` is `console`, `json` or `line`; each is read from the environment when empty, else
    `info` and whatever stderr is (a terminal or not). `line` is what the CLI's own verbs print for
    an agent reading stderr: no timestamp, module or JSON braces.
    """
    project = Project()
    level = (level or project.variable("LOG_LEVEL").read() or "info").lower()
    output = output or project.variable("LOG_FORMAT").read()
    json = output == "json" if output else not sys.stderr.isatty()
    shared: list[Processor] = [
        _Floor(level=NAME_TO_LEVEL[level]),
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _route,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    rendered: list[Processor] = (
        [structlog.processors.format_exc_info, _line]
        if output == "line"
        else [
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
        wrapper_class=Logger,
        logger_factory=lambda *_: _Stderr(),
        cache_logger_on_first_use=False,
    )


if not structlog.is_configured():
    configure()

logger: Logger = structlog.get_logger()
