"""Compact Swift migration reports must keep their documentation links usable."""

import pytest

from mainboard.analysis.findings import Report


@pytest.mark.parametrize(
    "url",
    [
        "https://developer.apple.com/videos/play/wwdc2026/274/",
        "https://example.com/27/274?count=3#section-2",
        "http://localhost:8080/123",
    ],
)
def test_grouped_findings_preserve_links_and_variable_values(url: str) -> None:
    report = Report(
        found={
            "SW-MIG0006": [
                ("B.swift", 28, 0, f"Review `Record.value` with 15 entries; see {url}"),
                ("A.swift", 4, 0, f"Review `Other.value` with 3.5 entries; see {url}"),
            ],
            "EMPTY": [],
        }
    )

    assert report.grouped == (
        f"SW-MIG0006 Review _ with _ entries; see {url}\n"
        " A.swift 4[Other.value|3.5]\n"
        " B.swift 28[Record.value|15]"
    )
    assert report.total == 2
    assert report.counts == "    2 SW-MIG0006\n    2 findings, 1 rules"


def test_distinct_documentation_targets_do_not_merge() -> None:
    first = "https://developer.apple.com/videos/play/wwdc2026/274/"
    second = "https://developer.apple.com/videos/play/wwdc2026/275/"
    report = Report(
        found={
            "SW-MIG0006": [
                ("A.swift", 1, 0, f"See {first}"),
                ("A.swift", 2, 0, f"See {second}"),
            ]
        }
    )

    assert report.grouped == (
        f"SW-MIG0006 See {first}\n A.swift 1\nSW-MIG0006 See {second}\n A.swift 2"
    )
