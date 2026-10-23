"""PBS resource flags: a node count reaches `qsub` with or without a memory request."""

import pytest

from mainboard.dispatch.schedulers import build_qsub_flags
from mainboard.dispatch.vocabulary import Resources


@pytest.mark.parametrize(
    ("resources", "select"),
    [
        (Resources(queue="interact-g"), None),
        (Resources(queue="interact-g", nodes=4), "select=4"),
        (Resources(queue="short-g", mem_gb=100), "select=1:mem=100gb"),
        (Resources(queue="short-g", nodes=2, mem_gb=100), "select=2:mem=100gb"),
    ],
)
def test_a_request_selects_its_nodes(resources: Resources, select: str | None) -> None:
    flags = build_qsub_flags(resources)
    selects = [
        flags[i + 1] for i, flag in enumerate(flags) if flag == "-l" and "select" in flags[i + 1]
    ]
    assert selects == ([select] if select else [])


def test_a_scripted_submit_leaves_the_headers_chunk_and_its_cards_alone() -> None:
    flags = build_qsub_flags(Resources(queue="short-g", gpus=1, mem_gb=100), scripted=True)

    assert not any("select" in flag for flag in flags)
    assert flags[:2] == ["-q", "short-g"]
