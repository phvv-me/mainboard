"""PY-EXCE0002 fixes a `try` guarding too much: the statements past the ceiling go after the
statement when every handler exits and under `else:` when one does not, with the handlers and
their comments untouched; whatever it cannot move without changing the text, it leaves."""

import re
from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ...builders import fixed
from .builders import cases

_RULE = "PY-EXCE0002"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `cases()` after one fix of the rule over them all."""
    sources = {name: source for name, (source, _) in cases().items()}
    return fixed(tmp_path_factory.mktemp("bounded"), sources, rules=(_RULE,))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("carriage_returns.py", id="a-try-in-a-file-with-carriage-returns"),
        pytest.param("two_spaces.py", id="an-indent-of-two-spaces-is-kept"),
        pytest.param("noted_handlers.py", id="comments-between-the-handlers-stay-by-them"),
        pytest.param("loop.py", id="a-handler-that-continues-a-loop"),
        pytest.param("closure.py", id="a-tail-defining-a-function"),
        pytest.param("method.py", id="a-method-of-a-class"),
        pytest.param("module.py", id="a-handler-that-falls-through-gets-an-else"),
        pytest.param("awaiting.py", id="a-coroutine"),
        pytest.param("twice.py", id="two-trys-in-one-pass"),
        pytest.param("trailing.py", id="a-comment-ending-the-last-moved-line"),
        pytest.param("tabs.py", id="a-try-indented-with-tabs"),
        pytest.param("multiline.py", id="a-tail-holding-a-string-over-two-lines"),
        pytest.param("with_else.py", id="a-try-with-an-else-already"),
        pytest.param("same_line.py", id="a-body-on-the-line-of-the-try"),
    ],
)
def test_a_wide_try_is_split_or_left_exactly_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == cases()[name][1]


@settings(max_examples=6)
@given(
    statements=st.integers(min_value=2, max_value=5),
    handler=st.sampled_from(["return None", "raise", "pass"]),
)
def test_every_statement_survives_in_order_and_the_try_keeps_one(
    tmp_path_factory: pytest.TempPathFactory, statements: int, handler: str
) -> None:
    body = "".join(f"        step{n}()\n" for n in range(statements))
    source = f"def run():\n    try:\n{body}    except OSError:\n        {handler}\n"

    result = fixed(tmp_path_factory.mktemp("kept"), {"run.py": source}, rules=(_RULE,))["run.py"]

    assert re.findall(r"step\d", result) == [f"step{n}" for n in range(statements)]
    assert result.partition("except")[0].count("step") == 1
