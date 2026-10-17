"""Rules whose findings are an external checker's diagnostics: the lane mcmr delegated to Ruff and
ty (providers/external.py runs them)."""

from ... import lint


def wired() -> bool:
    """Whether the engine registers the tools' facts, which it does once `lint` imports `Tool`.
    Until then a rule reading their views could not run and would fail every full check, so these
    rules stay out of the catalog; delete this gate with the wiring."""
    return hasattr(lint, "Tool")
