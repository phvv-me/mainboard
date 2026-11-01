"""`Output` prints a verb's document: its text as written, or its data as JSON a program reads."""

from collections.abc import Mapping

import pytest

from mainboard import MissionError
from mainboard.render import Output

_PAYLOAD: Mapping[str, int | tuple[int, ...]] = {"a": 1, "b": (2, 3)}


@pytest.mark.parametrize(
    ("output", "printed"),
    [
        pytest.param(Output(), "words\n", id="the-text-as-written"),
        pytest.param(Output(json_=True), '{"a":1,"b":[2,3]}\n', id="the-whole-document"),
        pytest.param(Output(json_=True, fields="b"), '{"b":[2,3]}\n', id="the-projected-document"),
    ],
)
def test_a_document_prints_its_text_or_its_data_as_json(
    output: Output, printed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    output.print_document("words", _PAYLOAD, title="document")

    assert capsys.readouterr().out == printed


def test_a_document_refuses_a_projection_without_json(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(MissionError, match="add --json"):
        Output(fields="b").print_document("words", _PAYLOAD, title="document")

    assert capsys.readouterr().out == ""
