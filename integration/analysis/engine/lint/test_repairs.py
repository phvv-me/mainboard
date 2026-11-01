"""A fix's edits are described by their kinds when its rule states no change of its own."""

from mainboard.analysis import Span
from mainboard.analysis.claims import Claim
from mainboard.analysis.edits import Delete, Edit
from mainboard.analysis.engine.lint.repairs import placed
from mainboard.analysis.rules.python.libraries.logging import TextAroundTheLogger

_SPAN = Span(path="module.py", start_line=2, end_line=4)


def test_a_repair_without_a_change_template_counts_its_claim_beside_its_deletion() -> None:
    group: list[Edit] = [Delete(span=_SPAN), Claim(span=_SPAN)]

    said = placed(TextAroundTheLogger(), group, None)

    assert said == {
        "path": "module.py",
        "line": 1,
        "change": "deleted 1 span, claimed 1 block (text around the logger)",
    }
