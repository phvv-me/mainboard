"""A project's per-rule configuration: paths a rule must not see and setting overrides.

Read today from the analyzed project's `pyproject.toml` `[tool.mcmr.rules."ID"]` tables, which
every existing project already carries, so a ported rule is judged under mcmr's policy.
"""

import tomllib
from functools import cache
from pathlib import Path

from pathspec import PathSpec
from patos import FrozenModel
from pydantic import ConfigDict

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
        """The configuration in `root/pyproject.toml`, empty when it declares none."""
        try:
            declared = tomllib.loads((root / "pyproject.toml").read_text())
        except FileNotFoundError:
            return cls()
        tables = declared.get("tool", {}).get("mcmr", {}).get("rules", {})
        return cls(rules={rule: RuleConfig.of(table) for rule, table in tables.items()})

    def rule(self, rule: str) -> RuleConfig:
        return self.rules.get(rule, RuleConfig())
