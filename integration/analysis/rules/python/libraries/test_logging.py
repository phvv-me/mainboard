"""The logging rules' examples, among them the print of several values PY-LOGG0002 refuses."""

from mainboard.analysis.engine.lint import verbs
from mainboard.analysis.rules.python.libraries.logging import TextAroundTheLogger


def test_every_example_of_the_text_around_the_logger_passes() -> None:
    """Each `bad` is flagged, the fix turns it into `good` or refuses it, `good` is not flagged."""
    verbs.test(selection=verbs.Selection(rule=(TextAroundTheLogger.id,)))
