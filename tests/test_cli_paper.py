import json
from pathlib import Path

import pytest

from mainboard import Board
from mainboard.cli import build
from mainboard.manuscript import Kind, Problem, Report, Section

# A build that reached page 10 with its conclusion, and one unresolved reference.
_REPORT = Report(
    paper="head",
    pdf="/p/build/paper.pdf",
    pages=12,
    limit=9,
    ends="Conclusion",
    ends_on=10,
    sections=(
        Section(number="1", title="Introduction", page=1),
        Section(title="Appendix", page=11),
    ),
    problems=(
        Problem(kind=Kind.REFERENCE, where="sections/intro.tex:4", detail="fig:x"),
        Problem(kind=Kind.LIMIT, detail="'Conclusion' ends on page 10, past the 9-page limit"),
    ),
)


class Checked:
    """A manuscript whose build answers `report` and whose pages render where they are asked."""

    def __init__(self, report: Report) -> None:
        self.report = report
        self.shown: list[tuple[str, int]] = []

    def check(self) -> Report:
        return self.report

    def show(self, text: str, *, dpi: int) -> Path:
        self.shown.append((text, dpi))
        return Path(f"/p/build/head-page{len(self.shown)}.png")


@pytest.mark.parametrize(
    ("report", "flags", "code", "fragments"),
    [
        (_REPORT, ["--json"], "1", ()),
        (_REPORT, [], "1", ("paper: head", "Introduction", "fig:x", "past the 9-page limit")),
        (
            _REPORT.model_copy(update={"problems": (), "limit": 0}),
            ["--agent"],
            "0",
            ("kind\twhere\tdetail", "number\ttitle\tpage\twithin", "Appendix\t11\tTrue"),
        ),
    ],
    ids=["the whole report as json", "the tables, failing", "a clean build, compact"],
)
def test_paper_prints_the_check_and_exits_on_whether_anything_is_wrong(
    depot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    report: Report,
    flags: list[str],
    code: str,
    fragments: tuple[str, ...],
) -> None:
    manuscript = Checked(report)
    monkeypatch.setattr(Board, "paper", lambda self, name: manuscript)
    with pytest.raises(SystemExit, match=f"^{code}$"):
        build(depot)(["center", "paper", "head", *flags, "--show", "Pareto", "--show", "Table 2"])
    out = capsys.readouterr().out
    assert manuscript.shown == [("Pareto", 110), ("Table 2", 110)]
    assert out.rstrip().endswith("/p/build/head-page2.png")
    if not fragments:
        printed = json.loads(out[: out.rindex("}") + 1])
        assert printed["ends_on"] == 10
        assert printed["problems"][0]["where"] == "sections/intro.tex:4"
        return
    assert all(fragment in out for fragment in fragments)
