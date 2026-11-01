"""CU-LAUN0002 gives a kernel launch on the default stream the one `cudaStream_t` its unit holds
in scope, after a dynamic shared memory size of zero when the launch names none; whatever else it
leaves."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..builders import fixed
from .builders import cases

_RULE = "CU-LAUN0002"
_KERNEL = "__global__ void scale(float* values) {}\n\n"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `cases()` after one fix of the rule over them all."""
    sources = {name: source for name, (source, _) in cases().items()}
    return fixed(tmp_path_factory.mktemp("launch"), sources, rules=(_RULE,))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("parameter.cu", id="a-stream-taken-as-a-parameter"),
        pytest.param("local.cu", id="a-stream-declared-in-the-function"),
        pytest.param("two.cu", id="two-launches-one-naming-its-shared-memory"),
        pytest.param("global.cu", id="a-stream-declared-at-file-scope"),
        pytest.param("template.cu", id="a-launch-of-a-template-kernel"),
        pytest.param("bare.cu", id="a-launch-passing-no-arguments"),
        pytest.param("expressions.cu", id="a-configuration-of-expressions"),
        pytest.param("carriage_returns.cu", id="a-kernel-launch-in-a-crlf-file"),
        pytest.param("after.cu", id="a-stream-declared-after-the-launch"),
        pytest.param("elsewhere.cu", id="a-stream-declared-in-another-function"),
        pytest.param("two_declared.cu", id="two-streams-declared-in-the-unit"),
        pytest.param("pointer.cu", id="a-stream-behind-a-pointer"),
        pytest.param("prototype.cu", id="a-stream-in-a-prototype-only"),
        pytest.param("commented.cu", id="a-comment-in-the-configuration"),
        pytest.param("named.cu", id="a-launch-naming-the-default-stream"),
    ],
)
def test_a_default_stream_launch_gets_the_one_stream_or_stays_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == cases()[name][1]


@settings(max_examples=6)
@given(
    grid=st.sampled_from(["1", "n", "(n + 255) / 256", "dim3(4, 4)", "blocks"]),
    block=st.sampled_from(["32", "n", "dim3(8, 8)", "threads * 2", "block.x"]),
    shared=st.sampled_from(["", "16", "bytes + 4"]),
)
def test_the_stream_follows_the_configuration_with_a_zero_when_no_shared_size_is_named(
    tmp_path_factory: pytest.TempPathFactory, grid: str, block: str, shared: str
) -> None:
    head = f"{_KERNEL}void run(float* d, cudaStream_t s) {{\n    scale<<<{grid}, {block}"
    given_shared = f", {shared}" if shared else ""
    sources = {"run.cu": f"{head}{given_shared}>>>(d);\n}}\n"}

    result = fixed(tmp_path_factory.mktemp("config"), sources, rules=(_RULE,))

    assert result["run.cu"] == f"{head}{given_shared or ', 0'}, s>>>(d);\n}}\n"
