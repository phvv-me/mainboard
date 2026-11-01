# A TOML value as tomllib parses it. The PEP 695 alias lets pydantic resolve the recursion, and
# covariant `Sequence`/`Mapping` accept a concrete `list[str]` or `dict[str, str]`.
import json
import tomllib
from collections.abc import Mapping, Sequence

type Toml = str | int | float | bool | Sequence[Toml] | Mapping[str, Toml]


def canonical(value: Toml) -> bytes:
    """`value` as the one JSON spelling that hashes alike wherever it was parsed."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def requirements(document: dict[str, Toml], key: str) -> set[str]:
    """The names a compiled manifest requires under `key`, anywhere.

    They are spelled as the indexes normalize them.
    """
    return {
        name.lower().replace("_", "-")
        for scope in _scopes(document)
        if isinstance(declared := scope.get(key), dict)
        for name in declared
    }


def resolution_metadata(text: str) -> dict[str, Toml]:
    """The tables of a local project's `pyproject.toml` a solve reads, and nothing else.

    Other `tool` tables cannot move a resolution; hashing them refused every host setup
    after a codespell edit. `project` stays whole since all of it reaches the resolver.
    """
    parsed = tomllib.loads(text)
    tool = parsed.get("tool", {})
    tables: dict[str, Toml] = {
        name: parsed[name]
        for name in ("build-system", "project", "dependency-groups")
        if name in parsed
    }
    resolvers = {name: tool[name] for name in ("pixi", "uv") if name in tool}
    if resolvers:
        tables["tool"] = resolvers
    return tables


def stripped(document: dict[str, Toml], key: str) -> dict[str, Toml]:
    """`document` with `key` popped from the root, every feature, and their targets."""
    for scope in _scopes(document):
        scope.pop(key, None)
    return document


def _scopes(document: dict[str, Toml]) -> list[dict[str, Toml]]:
    """The root of a compiled manifest, every feature, and each of their targets."""
    return [
        scope
        for table in (document, *_tables(document.get("feature")))
        for scope in (table, *_tables(table.get("target")))
    ]


def _tables(parent: Toml | None) -> list[dict[str, Toml]]:
    """The tables under `parent`, read defensively from re-parsed TOML."""
    if not isinstance(parent, dict):
        return []
    return [table for table in parent.values() if isinstance(table, dict)]
