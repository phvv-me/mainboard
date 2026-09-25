from enum import StrEnum, auto

from patos import FrozenModel


class Verdict(StrEnum):
    """How one section came back: fit, fit with something worth saying, or broken."""

    PASS = auto()
    WARN = auto()
    FAIL = auto()


class Section(FrozenModel):
    """One area judged, with the single command that repairs it.

    The row every report in this tool is made of, `doctor`, `center verify`, `center migrate`
    and the machine findings `facts`, `compute` and `setup` print, so a reader learns one shape.

    section: the area reported on.
    verdict: whether it is fit, worth a word, or broken.
    detail: the one line behind the verdict.
    fix: the command that repairs it, empty when nothing needs repairing.
    """

    section: str
    verdict: Verdict
    detail: str
    fix: str = ""


def failed(sections: list[Section]) -> bool:
    """Whether any of `sections` is broken, the exit status every report shares."""
    return any(section.verdict is Verdict.FAIL for section in sections)


def staged(stage: str, row: Section) -> Section:
    """`row` named after the stage that produced it, so one report can hold every stage.

    stage: the stage's name, `machine` or `verify` say.
    row: the stage's own row.
    """
    return row.model_copy(update={"section": f"{stage}: {row.section}"})
