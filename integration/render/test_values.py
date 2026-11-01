"""A printed cell carries no C0 or C1 control, since a cell may hold text anyone could write."""

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.render.human import render_table
from mainboard.render.values import encode, visible

# Tab and newline stay: the compact rows quote them, and a table lays them out.
_CONTROLS = {chr(code) for code in (*range(0x20), *range(0x7F, 0xA0))} - {"\t", "\n"}
_FORGED = "ok\x1b[2J\x1b]52;c;cHduZWQ=\x07"


@given(st.text())
@example(_FORGED)
@example("ok\x9b2J\r")
def test_text_shows_each_control_but_tab_and_newline_as_a_replacement_character(
    text: str,
) -> None:
    shown = "".join("�" if character in _CONTROLS else character for character in text)
    assert visible(text) == shown


def test_the_tab_rows_and_the_table_print_a_cell_through_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    render_table([{"cell": _FORGED}])
    printed = capsys.readouterr().out + encode([{"cell": _FORGED}])
    assert _CONTROLS.isdisjoint(printed)
    assert printed.count("ok�[2J�]52;c;cHduZWQ=�") == 2
