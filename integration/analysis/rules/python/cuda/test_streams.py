"""PY-CUDA0003 and PY-CUDA0004 give a launch or a copy to the device the one stream its function
holds, turn a device-wide barrier into a wait on it, and leave what a person must choose: no
stream, several, one the framework of the call does not take, a copy back to the host and a barrier
that names a device."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ...builders import fixed
from .builders import cases

_RULES = ("PY-CUDA0003", "PY-CUDA0004")


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `cases()` after one fix of both rules over them all."""
    sources = {name: source for name, (source, _) in cases().items()}
    return fixed(tmp_path_factory.mktemp("streams"), sources, rules=_RULES)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("tuple_grid.py", id="a-grid-written-as-a-tuple"),
        pytest.param("trailing_comma.py", id="a-trailing-comma-in-the-configuration"),
        pytest.param("twice.py", id="two-launches-in-one-function"),
        pytest.param("multiline.py", id="a-configuration-over-lines"),
        pytest.param("carriage_returns.py", id="a-launch-in-a-crlf-file"),
        pytest.param("both_rules.py", id="a-launch-both-rules-flag-is-fixed-once"),
        pytest.param("copy.py", id="a-copy-to-the-device-takes-the-stream-keyword"),
        pytest.param("patos.py", id="a-patos-launch-takes-the-stream-keyword"),
        pytest.param("barrier.py", id="a-device-wide-barrier-becomes-a-wait-on-the-stream"),
        pytest.param("barrier_device.py", id="a-barrier-naming-the-device-it-waits-for"),
        pytest.param("barrier_foreign.py", id="a-barrier-of-another-framework-than-the-stream"),
        pytest.param("barrier_two.py", id="a-barrier-in-a-function-holding-two-streams"),
        pytest.param("host.py", id="a-copy-back-to-the-host"),
        pytest.param("after.py", id="a-stream-made-after-the-launch"),
        pytest.param("two_streams.py", id="a-function-holding-two-streams"),
        pytest.param("other_name.py", id="a-stream-parameter-not-named-stream"),
        pytest.param("torch_stream.py", id="a-numba-launch-beside-a-torch-stream"),
        pytest.param("one_item.py", id="a-configuration-of-one-item"),
    ],
)
def test_a_default_stream_call_gets_the_one_stream_or_stays_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == cases()[name][1]


@settings(max_examples=6)
@given(
    grid=st.sampled_from(["1", "n", "(2, 3)", "a + b", "f(x)"]),
    block=st.sampled_from(["32", "n", "(8, 8)", "a * b", "g[0]"]),
    trailing=st.booleans(),
)
def test_the_stream_follows_the_two_configuration_items_whatever_they_are(
    tmp_path_factory: pytest.TempPathFactory, grid: str, block: str, trailing: bool
) -> None:
    head = "from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\n"
    body = f"def run(x, a, b, n, f, g, stream):\n    kernel[{grid}, {block}"
    sources = {"run.py": f"{head}{body}{',' if trailing else ''}](x)\n"}

    result = fixed(tmp_path_factory.mktemp("config"), sources, rules=_RULES)

    assert result["run.py"] == f"{head}{body}, stream](x)\n"
