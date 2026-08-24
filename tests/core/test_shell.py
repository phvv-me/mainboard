import shlex
from typing import TYPE_CHECKING

import pytest
from hypothesis import given

from mainboard import script, sh

from ..strategies import TEXT

if TYPE_CHECKING:
    from collections.abc import Callable
    from string.templatelib import Template


@given(value=TEXT)
def test_a_shell_line_quotes_every_interpolation_and_leaves_its_static_text_alone(
    value: str,
) -> None:
    """Whatever a value carries, the shell reads it back as exactly one word."""
    line = sh(t"cd {value} && true")
    assert line.startswith("cd ")
    assert line.endswith(" && true")
    assert shlex.split(line) == ["cd", value, "&&", "true"]


@pytest.mark.parametrize("compose", [sh, script])
def test_a_plain_string_is_refused_where_a_t_string_belongs(
    compose: Callable[[Template], str],
) -> None:
    """Unquoted composition is unrepresentable, so the mistake fails at the call site."""
    with pytest.raises(TypeError, match="expected a t-string"):
        compose("cd /tmp && rm -rf *")  # type: ignore[arg-type]  # the refusal under test


def test_a_trusted_fragment_lands_verbatim_inside_a_larger_line() -> None:
    """`script` is the opt-out that composes an already-quoted line without quoting it twice."""
    inner = sh(t"echo {'a b'}")
    assert script(t"bash -lc {inner}") == "bash -lc echo 'a b'"
