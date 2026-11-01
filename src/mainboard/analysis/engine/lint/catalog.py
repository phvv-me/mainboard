"""The rule catalog a run draws on: every provider and rule module, and the rules it selects."""

from collections.abc import Sequence
from importlib import import_module

from ....core.errors import MissionError
from ....core.execution.modules import submodules
from ....log import logger
from ...catalog import Rule


def load_catalog() -> None:
    """Import every fact provider and rule module, registering its families, rules and relations.

    A module that fails to import (one being written) is reported and skipped.
    """
    for name in [
        *submodules("mainboard.analysis.providers"),
        *submodules("mainboard.analysis.rules"),
    ]:
        try:
            import_module(name)
        except (SyntaxError, ImportError, NameError, AttributeError, TypeError) as error:
            logger.result(f"skipped {name}: {type(error).__name__}: {error}")


def selected(rule: Sequence[str]) -> list[type[Rule]]:
    """The rules named by `rule`, else every rule.

    Raises MissionError for an id no rule defines.
    """
    load_catalog()
    if not rule:
        return list(Rule.defined.values())
    try:
        return [Rule.defined[r] for r in rule]
    except KeyError as unknown:
        raise MissionError(
            f"no rule {unknown.args[0]!r} to select; `mb code rules explain` lists them"
        ) from None
