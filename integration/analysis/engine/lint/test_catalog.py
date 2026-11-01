"""`selected` turns rule ids into rules; an id no rule defines is a MissionError."""

import pytest

from mainboard import MissionError
from mainboard.analysis import Rule
from mainboard.analysis.engine.lint import selected


def test_selected_is_every_rule_without_ids_and_the_named_rules_in_order_with_them() -> None:
    every = selected(())
    assert every == list(Rule.defined.values())
    last, first = every[-1], every[0]
    assert selected((last.id, first.id)) == [last, first]


def test_selected_refuses_an_unknown_id_with_a_one_line_message() -> None:
    with pytest.raises(MissionError, match="no rule 'ALL-NO-SUCH-RULE' to select; `mb code rules"):
        selected((selected(())[0].id, "ALL-NO-SUCH-RULE"))
