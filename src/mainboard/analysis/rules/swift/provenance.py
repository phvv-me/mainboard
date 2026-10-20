"""Offline review context; this is neither live documentation freshness nor Apple endorsement."""

from datetime import date

from ...catalog import RuleProvenance


def apple_review(*inspected: str) -> RuleProvenance:
    """Record this source/guidance review without claiming a build or resolved API check."""
    return RuleProvenance(
        reviewed_on=date(2026, 10, 20),
        method="Offline rule/source review with installed public SDK declarations",
        review_context="Xcode 27.0 (27A266a); macOS 27.0 and iOS 27.0 SDK public interfaces",
        inspected=inspected,
        limits=(
            "The date records a Mainboard guidance review, not a fresh documentation fetch. "
            "Official URLs are offline references. SDK context is historical review evidence, "
            "not the analyzed project's deployment target or runtime toolchain. Syntax cannot "
            "establish resolved symbols, availability, ownership or safe API replacement. "
            "Cross-file shadows pool top-level declarations; local names conservatively "
            "suppress within their own file. Lexical visibility and target membership are "
            "not resolved, so same-file or unrelated-target shadows can still hide signals. "
            "Compiler/runtime acceptance remains separate; no automatic rewrite is offered."
        ),
    )
