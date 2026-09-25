from hypothesis import given
from hypothesis import strategies as st

from mainboard.core.section import Section, Verdict, failed, staged

from ..strategies import TEXT, WORDS

_SECTIONS = st.builds(
    Section, section=WORDS, verdict=st.sampled_from(Verdict), detail=TEXT, fix=TEXT
)


@given(sections=st.lists(_SECTIONS, max_size=6))
def test_a_report_fails_exactly_when_one_of_its_rows_is_broken(sections: list[Section]) -> None:
    """Warnings are worth a word but never a failing exit, and an empty report is fit."""
    assert failed(sections) is (Verdict.FAIL in {row.verdict for row in sections})


@given(stage=WORDS, row=_SECTIONS)
def test_a_staged_row_is_the_same_finding_under_its_stages_name(stage: str, row: Section) -> None:
    """Prefixing the stage lets one report hold every stage without losing any row's meaning."""
    named = staged(stage, row)
    assert named.section == f"{stage}: {row.section}"
    assert named.model_copy(update={"section": row.section}) == row
