from enum import StrEnum, auto

from patos import FrozenModel


class Verdict(StrEnum):
    """How one section came back: fit, fit with something worth saying, or broken."""

    PASS = auto()
    WARN = auto()
    FAIL = auto()


class Section(FrozenModel):
    """One area judged, with the single command that repairs it.

    The row every report is made of (`doctor`, `center verify`, `center migrate`, and the machine
    findings of `facts`, `compute` and `setup`), so a reader learns one shape.

    detail: the one line behind the verdict.
    fix: the repairing command, empty when nothing needs repairing.
    """

    section: str
    verdict: Verdict
    detail: str
    fix: str = ""


def failed(sections: list[Section]) -> bool:
    """Whether any of `sections` is broken, the exit status every report shares."""
    return any(section.verdict is Verdict.FAIL for section in sections)


def staged(stage: str, row: Section) -> Section:
    """`row` prefixed with its stage (`machine`, `verify`), so one report can hold every stage."""
    return row.model_copy(update={"section": f"{stage}: {row.section}"})
