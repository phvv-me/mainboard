from collections.abc import Iterable, Sequence

from patos import FrozenModel

from ..enums import Verdict

# Installs keyed by package, then by `platform.system()`. A system not named is read as a Linux
# distribution, the one family with no single installer.
_INSTALLS = {
    "git": {"Darwin": "xcode-select --install"},
    "git-lfs": {"Darwin": "brew install git-lfs"},
    "gh": {"Darwin": "brew install gh"},
    "ssh": {"Darwin": "ssh ships with macOS; restore it with xcode-select --install"},
}
_DISTRIBUTION = "sudo apt install {package} (or the {package} package of this distribution)"

# How many names a detail line lists before it only counts the rest.
_NAMED = 3


class Section(FrozenModel):
    """One area judged, with the single command that repairs it.

    The row every report is made of (`doctor`, `host setup --center`, and the machine
    findings of `host list` and `host setup`), so a reader learns one shape.

    detail: the one line behind the verdict.
    fix: the repairing command, empty when nothing needs repairing.
    """

    section: str
    verdict: Verdict
    detail: str
    fix: str = ""


def abbreviated(names: Sequence[str], named: int = _NAMED) -> str:
    """The first `named` of `names`, then only a count of the rest."""
    rest = len(names) - named
    return ", ".join(names[:named]) + (f" and {rest} more" if rest > 0 else "")


def failed(sections: Iterable[Section]) -> bool:
    """Whether any of `sections` is broken, the exit status every report shares."""
    return any(section.verdict is Verdict.FAIL for section in sections)


def install_command(system: str, *, package: str) -> str:
    """The command that installs `package` on a `system` machine, a distribution's when unknown.

    system: the platform as `platform.system()` spells it.
    """
    return _INSTALLS.get(package, {}).get(system, _DISTRIBUTION.format(package=package))


def staged(stage: str, row: Section) -> Section:
    """`row` prefixed with its stage (`machine`, `verify`), so one report can hold every stage."""
    return row.model_copy(update={"section": f"{stage}: {row.section}"})
