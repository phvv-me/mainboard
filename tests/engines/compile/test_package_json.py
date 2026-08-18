import json

import pytest

from mainboard import MissionError
from mainboard.engines.compile.package_json import PackageJson
from mainboard.manifest.schema.spec import Spec


def test_a_compiled_manifest_carries_the_fields_npm_reads() -> None:
    compiled = PackageJson.compiled(
        name="w-npm",
        deps={"prettier": Spec.model_validate(">=3")},
        dev={"eslint": Spec.model_validate("^10")},
        fields={"type": "module"},
    )
    body = json.loads(compiled.to_json())
    assert body == {
        "name": "w-npm",
        "private": True,
        "dependencies": {"prettier": ">=3"},
        "type": "module",
        "devDependencies": {"eslint": "^10"},
    }


def test_a_manifest_without_dev_requirements_declares_no_dev_table() -> None:
    compiled = PackageJson.compiled(
        name="w-npm", deps={"prettier": Spec.model_validate("*")}, dev={}, fields={}
    )
    assert "devDependencies" not in json.loads(compiled.to_json())


def test_declared_fields_win_over_the_generated_ones_they_name() -> None:
    """`[nodejs.package]` is the escape hatch, so it is merged last and overrides."""
    compiled = PackageJson.compiled(
        name="w-npm", deps={}, dev={}, fields={"private": False, "version": "1.2.3"}
    )
    body = json.loads(compiled.to_json())
    assert body["private"] is False
    assert body["version"] == "1.2.3"


def test_the_rendered_text_is_indented_json_ending_in_a_newline() -> None:
    text = PackageJson.compiled(name="w-npm", deps={}, dev={}, fields={}).to_json()
    assert text.startswith('{\n  "name"')
    assert text.endswith("\n")


def test_a_requirement_carrying_a_source_is_refused_with_what_it_carries() -> None:
    """Compiled to a bare `*`, a git source would silently install a registry package."""
    with pytest.raises(MissionError, match=r"lib.*git, path.*\[nodejs.package\]"):
        PackageJson.requirement(
            "lib", Spec.model_validate({"git": "https://example.com/l.git", "path": "../l"})
        )
