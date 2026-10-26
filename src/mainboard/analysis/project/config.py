"""A project's per-rule configuration: paths a rule must not see and setting overrides.

Native projects declare `[analysis.rules."ID"]` in their Mainboard manifest. Existing
`pyproject.toml` `[tool.mcmr.rules."ID"]` tables remain supported; declaring the same rule in
both places is refused so precedence never hides a policy.
"""

import tomllib
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from pathspec import GitIgnoreSpec
from patos import FrozenModel
from pydantic import ConfigDict

from ...core.project import Project
from ..engine.facts import Setting


@cache
def _spec(patterns: tuple[str, ...]) -> GitIgnoreSpec:
    return GitIgnoreSpec.from_lines(patterns)


class RuleTable(FrozenModel):
    """One `[tool.mcmr.rules."ID"]` table as written: the reserved keys, and every other key a
    setting."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    exclude: tuple[str, ...] = ()
    policy: dict[str, Setting] = {}
    settings: dict[str, Setting] = {}


class RuleConfig(FrozenModel):
    """One rule's project policy: whether it runs, excluded paths (gitignore globs), settings."""

    enabled: bool = True
    exclude: tuple[str, ...] = ()
    settings: dict[str, Setting] = {}

    @classmethod
    def of(cls, table: dict[str, Setting]) -> RuleConfig:
        """One rule's table, its settings beside the reserved keys or under `settings`.

        A `policy`'s bounds become the rule's `minimum` and `maximum` settings.
        """
        written = RuleTable.model_validate(table)
        policy = {key: value for key, value in written.policy.items() if key != "kind"}
        return cls(
            enabled=written.enabled,
            exclude=written.exclude,
            settings=written.settings | (written.model_extra or {}) | policy,
        )

    def accepts(self, path: str) -> bool:
        """Whether a finding at `path` reaches the report."""
        return not self.exclude or not _spec(self.exclude).match_file(path)


class ProjectConfig(FrozenModel):
    """Every rule's configuration in one project.

    proof: the command proving a fixed copy of the project still works, run inside the copy
        (`{root}` names it); a fix that fails it is undone. Empty when the project names none.
    """

    proof: tuple[str, ...] = ()
    rules: dict[str, RuleConfig] = {}

    @classmethod
    def of(cls, root: Path) -> ProjectConfig:
        """Per-owner native or legacy policy; a rule must have one unambiguous declaration."""
        native = cls._read_rules(Project().manifest(root), ("analysis", "rules"))
        legacy = cls._read_rules(root / "pyproject.toml", ("tool", "mcmr", "rules"))
        if duplicated := sorted(native.keys() & legacy.keys()):
            raise ValueError(
                f"rule {duplicated[0]} is configured in both the Mainboard manifest and "
                "pyproject.toml; keep one declaration"
            )
        proofs = [
            proof
            for proof in (
                cls._read(Project().manifest(root), ("analysis", "proof")),
                cls._read(root / "pyproject.toml", ("tool", "mcmr", "proof")),
            )
            if proof
        ]
        if len(proofs) > 1:
            raise ValueError(
                "the fix proof is declared in both the Mainboard manifest and pyproject.toml; "
                "keep one declaration"
            )
        return cls(rules=legacy | native, proof=tuple(proofs[0]) if proofs else ())

    @staticmethod
    def _read(path: Path, keys: Sequence[str]) -> list[str]:
        """The string list at `keys` in the TOML file `path`, empty when it holds none."""
        try:
            table = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        for key in keys:
            table = table.get(key, {})
        return list(table) if isinstance(table, list) else []

    @staticmethod
    def _read_rules(path: Path, keys: Sequence[str]) -> dict[str, RuleConfig]:
        try:
            tables = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        for key in keys:
            tables = tables.get(key, {})
        return {rule: RuleConfig.of(table) for rule, table in tables.items()}

    def rule(self, rule: str) -> RuleConfig:
        return self.rules.get(rule, RuleConfig())
