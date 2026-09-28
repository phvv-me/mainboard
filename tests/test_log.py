"""The one logger: loguru's call shape, structured fields, stderr output, sinks, and `mb`."""

import json
from collections.abc import Iterator
from importlib import import_module

import pytest
from structlog.contextvars import bound_contextvars

from mainboard.log import SINK, EventDict, configure, logger, sinks


@pytest.fixture
def reconfigured() -> Iterator[None]:
    """Leave the process configured from its environment again, whatever the test set."""
    yield
    configure()


def test_mb_is_mainboard_under_its_short_name() -> None:
    """One package under two names, so no module is ever loaded twice."""
    import mainboard.trials
    import mb
    from mb import logger as short

    # Imported by name: a type checker sees `mb` as `mainboard`'s exports, not its submodules.
    trials = import_module("mb.trials")
    assert mb is mainboard and trials is mainboard.trials and short is logger
    assert trials.__spec__ is not None and trials.__spec__.name == "mainboard.trials"


def test_positional_arguments_format_the_loguru_way_and_keywords_are_fields(
    logged: list[EventDict],
) -> None:
    logger.info("took {}s", 3, host="gold")
    logger.warning("a plain %s stays as written")
    assert logged == [
        {"event": "took 3s", "host": "gold", "log_level": "info"},
        {"event": "a plain %s stays as written", "log_level": "warning"},
    ]


def test_off_a_terminal_each_event_is_one_json_line_naming_where_it_was_logged(
    capsys: pytest.CaptureFixture[str], reconfigured: None
) -> None:
    configure("warning", "json")
    logger.info("below the floor")
    logger.warning("fetched", files=12)
    [line] = capsys.readouterr().err.splitlines()
    event = json.loads(line)
    assert (event["event"], event["level"], event["files"]) == ("fetched", "warning", 12)
    assert event["qual_module"] == __name__ and event["timestamp"].endswith("Z")


def test_an_event_naming_a_sink_is_handed_to_it_without_the_name() -> None:
    kept: list[EventDict] = []
    sinks["trial-1"] = kept.append
    try:
        with bound_contextvars(**{SINK: "trial-1"}):
            logger.info("inside", step=1)
        logger.info("outside")
    finally:
        del sinks["trial-1"]
    [event] = kept
    assert (event["event"], event["step"], event["level"]) == ("inside", 1, "info")
    assert SINK not in event
