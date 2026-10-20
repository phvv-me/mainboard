"""A project's per-rule configuration: paths a rule must not see and setting overrides.

Native projects declare `[analysis.rules."ID"]` in their Mainboard manifest. Existing
`pyproject.toml` `[tool.mcmr.rules."ID"]` tables remain supported; declaring the same rule in
both places is refused so precedence never hides a policy.
"""

import tomllib
from functools import cache
from pathlib import Path

from pathspec import PathSpec
from patos import FrozenModel
from pydantic import ConfigDict

from ..core.project import Project
from .facts import Setting


@cache
def _spec(patterns: tuple[str, ...]) -> PathSpec:
    return PathSpec.from_lines("gitwildmatch", patterns)


class RuleTable(FrozenModel):
    """One `[tool.mcmr.rules."ID"]` table as written: the reserved keys, and every other key a
    setting."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    exclude: tuple[str, ...] = ()
    policy: dict[str, Setting] = {}
    settings: dict[str, Setting] = {}


class RuleConfig(FrozenModel):
    """One rule's project policy: whether it runs, excluded paths (gitwildmatch), settings."""

    enabled: bool = True
    exclude: tuple[str, ...] = ()
    settings: dict[str, Setting] = {}

    @classmethod
    def of(cls, table: dict[str, Setting]) -> RuleConfig:
        """One rule's table: settings sit beside the reserved keys or under `settings`, and a
        `policy`'s bounds become the rule's `minimum` and `maximum` settings."""
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
    """Every rule's configuration in one project."""

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
        return cls(rules=legacy | native)

    @staticmethod
    def _read_rules(path: Path, keys: tuple[str, ...]) -> dict[str, RuleConfig]:
        try:
            tables = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        for key in keys:
            tables = tables.get(key, {})
        return {rule: RuleConfig.of(table) for rule, table in tables.items()}

    def rule(self, rule: str) -> RuleConfig:
        return self.rules.get(rule, RuleConfig())
