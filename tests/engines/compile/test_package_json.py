import json

import pytest

from mainboard import MissionError
from mainboard.engines.compile.package_json import PackageJson
from mainboard.manifest.schema.spec import Json, Spec


@pytest.mark.parametrize(
    ("deps", "dev", "fields", "body"),
    [
        pytest.param(
            {"prettier": ">=3"},
            {"eslint": "^10"},
            {"type": "module"},
            {
                "name": "w-npm",
                "private": True,
                "dependencies": {"prettier": ">=3"},
                "type": "module",
                "devDependencies": {"eslint": "^10"},
            },
            id="every-field-npm-reads",
        ),
        pytest.param(
            {"prettier": "*"},
            {},
            {},
            {"name": "w-npm", "private": True, "dependencies": {"prettier": "*"}},
            id="no-dev-requirements-declares-no-dev-table",
        ),
        pytest.param(
            {},
            {},
            {"private": False, "version": "1.2.3"},
            {"name": "w-npm", "private": False, "dependencies": {}, "version": "1.2.3"},
            id="declared-fields-win-over-the-generated-ones-they-name",
        ),
    ],
)
def test_a_compiled_manifest_carries_what_the_node_manager_installs_from(
    deps: dict[str, str], dev: dict[str, str], fields: dict[str, Json], body: dict[str, Json]
) -> None:
    """`[nodejs.package]` is the escape hatch, so it is merged last and overrides."""
    compiled = PackageJson.compiled(
        name="w-npm",
        deps={name: Spec.model_validate(spec) for name, spec in deps.items()},
        dev={name: Spec.model_validate(spec) for name, spec in dev.items()},
        fields=fields,
    )
    text = compiled.to_json()
    assert json.loads(text) == body
    assert text.startswith('{\n  "name"')
    assert text.endswith("\n")


def test_a_requirement_carrying_a_source_is_refused_with_what_it_carries() -> None:
    """Compiled to a bare `*`, a git source would silently install a registry package."""
    with pytest.raises(MissionError, match=r"lib.*git, path.*\[nodejs.package\]"):
        PackageJson.requirement(
            "lib", Spec.model_validate({"git": "https://example.com/l.git", "path": "../l"})
        )
