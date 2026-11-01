"""`explain`: a rule, or every rule of the engine, with what a user needs to choose, configure
and trust it.

Per rule: its id and one-line summary, the language its prefix names (`ALL-` rules read every
language), whether its fix may run unattended (`applicability`), the settings a project can
override with their defaults, the kind of its fix (`sql` over the findings, a `method` where SQL
cannot compute the text, or `none`) and how many examples `lint test` proves.
"""

from patos import FrozenModel

from ...core.errors import MissionError
from ...render import Output
from ..catalog import Bundle, Rule, RuleProvenance
from ..constants import LANGUAGES
from ..engine.facts import Setting
from ..engine.lint import load_catalog
from .report import Report

_COMPACT = Output()


class RuleSetting(FrozenModel):
    """A setting a project can override: its name, type and default."""

    name: str
    type: str
    default: Setting


class RuleEntry(FrozenModel):
    """One rule of the catalog."""

    id: str
    summary: str
    language: str
    applicability: str
    fix: str
    examples: int
    settings: list[RuleSetting]
    guidance: str
    references: tuple[str, ...]
    provenance: RuleProvenance | None = None

    @classmethod
    def of(cls, rule: type[Rule]) -> RuleEntry:
        return cls(
            id=rule.id,
            summary=rule.summary(),
            language=LANGUAGES.get(rule.id.split("-", 1)[0], "all"),
            applicability=rule.applicability,
            fix=rule.fixes(),
            examples=len(rule.examples),
            guidance=rule.guidance,
            references=rule.references,
            provenance=rule.provenance,
            settings=[
                RuleSetting(
                    name=name,
                    type=getattr(field.annotation, "__name__", str(field.annotation)),
                    default=list(field.default)
                    if isinstance(field.default, tuple)
                    else field.default,
                )
                for name, field in (
                    rule.fields() if issubclass(rule, Bundle) else rule.model_fields
                ).items()
            ],
        )


class Catalog(Report):
    """The rules, ordered by id."""

    rules: list[RuleEntry]

    @staticmethod
    def _settings(rule: RuleEntry) -> str:
        if not rule.settings:
            return ""
        return "  [" + ", ".join(f"{s.name}={s.default!r}" for s in rule.settings) + "]"

    def text(self) -> str:
        width = max((len(rule.id) for rule in self.rules), default=0)
        counts = {
            language: sum(rule.language == language for rule in self.rules)
            for language in sorted({rule.language for rule in self.rules})
        }
        header = f"{len(self.rules)} rules: " + ", ".join(f"{n} {k}" for k, n in counts.items())
        return "\n".join(
            [
                header,
                *(
                    f"{rule.id:<{width}}  {rule.language:<10} {rule.applicability:<8} "
                    f"{rule.fix:<6} {rule.examples:>2}  {rule.summary}"
                    f"{self._settings(rule)}"
                    for rule in self.rules
                ),
            ]
        )


class Explanation(Report):
    """A rule's evidence, repair policy and official migration references, available offline."""

    rule: RuleEntry

    def text(self) -> str:
        rule = self.rule
        provenance = rule.provenance
        return "\n".join(
            line
            for line in (
                f"{rule.id}: {rule.summary}",
                f"language: {rule.language}; applicability: {rule.applicability}; fix: {rule.fix}",
                rule.guidance,
                f"reviewed {provenance.reviewed_on}: {provenance.method}" if provenance else "",
                provenance.review_context if provenance else "",
                "; ".join(provenance.inspected) if provenance else "",
                provenance.limits if provenance else "",
                *rule.references,
            )
            if line
        )


def explain(
    rule: str = "",
    *,
    output: Output = _COMPACT,
) -> None:
    """Explain RULE, its repair policy and remediation references, or list every rule without one.

    The list gives each rule's language, applicability, settings, fix kind and example count.
    Nothing is fetched: the documentation ships with the engine.

    rule: the exact rule id from a finding or from the list; every rule is listed when omitted.
    """
    load_catalog()
    if not rule:
        listed = [RuleEntry.of(Rule.defined[id]) for id in sorted(Rule.defined)]
        Catalog(rules=listed).show(output)
        return
    try:
        specification = Rule.defined[rule]
    except KeyError:
        raise MissionError(f"no rule {rule!r}; `mb code rules explain` lists them") from None
    Explanation(rule=RuleEntry.of(specification)).show(output)
