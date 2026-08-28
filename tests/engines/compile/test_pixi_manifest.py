import tomllib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.engines.compile.pixi_manifest import (
    PixiManifest,
    dependency_tables,
    pypi_options,
    rerooted,
    spec_toml,
)
from mainboard.engines.compile.platforms import PlatformMatrix
from mainboard.manifest import Manifest, Scope, Spec

# `Json` annotates a property's own arguments, which hypothesis resolves at run time.
from mainboard.manifest.schema.spec import Json

from ...strategies import PATHS, SPECS, WORDS

if TYPE_CHECKING:
    from mainboard.engines.compile.toml import Toml
    from mainboard.manifest.schema.environment import Task

_PROJECT = "mainboard"

# The keys a dependency spec really carries beside its version, a local source, a remote one,
# and the flags that qualify them.
_EXTRAS = st.dictionaries(
    st.sampled_from(["path", "git", "editable", "index"]),
    st.one_of(PATHS, st.booleans()),
    max_size=2,
)

# Dependency tables as a manifest declares them, small enough that a falsifying example is
# still readable.
_TABLE = st.dictionaries(WORDS, SPECS, max_size=3)


# Ten examples rather than the profile's thirty, because this suite is the fast gate and a
# full compile per example is the most expensive thing in it. Every branch these properties
# reach is pinned by an `@example` or by a parametrized case, never by the search.
@settings(max_examples=10)
@given(version=SPECS, extras=_EXTRAS)
@example(version=">=2.9", extras={})
@example(version="*", extras={"path": "packages/lab-core", "editable": True})
@example(version=">=1", extras={"git": "https://example.com/repo"})
def test_the_smallest_toml_for_a_spec_keeps_everything_it_declared(
    version: str, extras: dict[str, Json]
) -> None:
    """A bare version renders as a string, and an unconstrained one is dropped beside extras."""
    spec = Spec.model_validate({"version": version, **extras})
    rendered = spec_toml(spec)
    assert Spec.model_validate(rendered) == spec
    assert isinstance(rendered, str) is not bool(extras)
    if extras:
        assert ("version" in rendered) is (version != "*")


@pytest.mark.parametrize(
    ("declared", "compiled"),
    [("packages/lote", "../packages/lote"), ("/opt/lote", "/opt/lote"), ("", "..")],
)
def test_rerooted_shifts_only_a_workspace_relative_location(declared: str, compiled: str) -> None:
    """One rule for every declared location, since they all resolve from `.mainboard/`."""
    assert rerooted(declared) == compiled


@pytest.mark.parametrize(
    ("body", "tables"),
    [
        ({}, {}),
        ({"deps": {"ripgrep": "*"}}, {"dependencies": {"ripgrep": "*"}}),
        ({"python": {"deps": {"torch": ">=2.9"}}}, {"pypi-dependencies": {"torch": ">=2.9"}}),
        (
            {"python": {"deps": {"torch": "*"}, "dev": {"ruff": "*"}}},
            {"pypi-dependencies": {"torch": "*", "ruff": "*"}},
        ),
        ({"python": {}}, {}),
        ({"nodejs": {"deps": {"prettier": "*"}}}, {}),
    ],
)
def test_dependency_tables_compile_conda_and_python_and_nothing_else(
    body: dict[str, Json], tables: dict[str, Toml]
) -> None:
    """Every other declared ecosystem rides in the manifest untranslated, on purpose."""
    assert dependency_tables(Scope.model_validate(body)) == tables


@pytest.mark.parametrize(
    ("overlay", "tables"),
    [
        pytest.param(
            {"deps": {"python": "*"}},
            {"dependencies": {"python": ">=3.14.6,<3.15"}},
            id="an-unconstrained-restatement-arrives-carrying-the-scopes-own-floor",
        ),
        pytest.param(
            {"deps": {"python": ">=3.14.9"}},
            {"dependencies": {"python": ">=3.14.9"}},
            id="an-overlay-that-narrows-keeps-its-own-constraint",
        ),
        pytest.param(
            {"deps": {"libgcc": "*"}},
            {"dependencies": {"libgcc": "*"}},
            id="a-package-the-scope-never-named-rides-through-untouched",
        ),
        pytest.param(
            {"deps": {"python": {"build": "cpython_*"}}},
            {"dependencies": {"python": {"version": ">=3.14.6,<3.15", "build": "cpython_*"}}},
            id="a-build-selector-gains-the-floor-instead-of-erasing-it",
        ),
        pytest.param(
            {"python": {"deps": {"torch": "*"}}},
            {"pypi-dependencies": {"torch": ">=2.9"}},
            id="the-pypi-side-carries-its-floor-the-same-way",
        ),
    ],
)
def test_a_platform_overlay_never_widens_what_the_scope_it_overlays_pinned(
    overlay: dict[str, Json], tables: dict[str, Toml]
) -> None:
    """A pixi `[target]` dependency replaces rather than narrows, so the floor must ride along.

    Without this, `python = "*"` in an `[on.osx]` overlay compiles to "any python at all on
    osx" and that platform's solve drifts onto an interpreter nobody asked for.
    """
    scope = Scope.model_validate(
        {"deps": {"python": ">=3.14.6,<3.15"}, "python": {"deps": {"torch": ">=2.9"}}}
    )
    assert dependency_tables(Scope.model_validate(overlay), over=scope) == tables


@pytest.mark.parametrize(
    ("body", "options"),
    [
        ({}, {}),
        ({"python": {"deps": {"torch": "*"}}}, {}),
        (
            {
                "python": {
                    "deps": {"torch": "*"},
                    "dependency-overrides": {"cuda-core": ">=1.1.1, <2"},
                    "index-strategy": "unsafe-best-match",
                    "extra-index-urls": ["https://pypi.nvidia.com"],
                    "no-build-isolation": ["fastsafetensors"],
                    "indexes": {"nvidia": "https://pypi.nvidia.com"},
                }
            },
            {
                "extra-index-urls": ["https://pypi.nvidia.com"],
                "index-strategy": "unsafe-best-match",
                "no-build-isolation": ["fastsafetensors"],
                "dependency-overrides": {"cuda-core": ">=1.1.1, <2"},
            },
        ),
    ],
)
def test_pypi_options_forward_only_the_settings_pixi_defines(
    body: dict[str, Json], options: Mapping[str, Toml]
) -> None:
    """Non-solve settings pass through and overrides are emitted last.

    A setting beside the deps that configures something other than the solve stays put, and
    the trailing override sub-table keeps TOML from swallowing the scalars after it.
    """
    assert list(pypi_options(Scope.model_validate(body)).items()) == list(options.items())


@pytest.mark.parametrize(
    ("spec", "translated"),
    [
        ("pytest", {"cmd": "pytest", "cwd": ".."}),
        (
            {"run": "pytest", "env": {"PYTHONPATH": "research"}},
            {"cmd": "pytest", "env": {"PYTHONPATH": "research"}, "cwd": ".."},
        ),
        (
            {"run": "pytest", "depends": ["build"], "dir": "packages/lab-core"},
            {"cmd": "pytest", "depends-on": ["build"], "cwd": "../packages/lab-core"},
        ),
        ({"run": "pytest", "dir": "/opt/project"}, {"cmd": "pytest", "cwd": "/opt/project"}),
        ({"depends": ["build", "lint"]}, {"depends-on": ["build", "lint"]}),
    ],
)
def test_a_task_takes_pixis_own_keys_and_runs_from_the_repo_root(
    spec: Task, translated: dict[str, Toml]
) -> None:
    """pixi rejects a `cwd` without a `cmd`, so an aggregator is the one task left unrebased."""
    assert PixiManifest.task(spec) == translated


@pytest.mark.parametrize(
    ("header", "table"),
    [
        ("dotenv = false\n", {}),
        ('\n[env]\nFOO = "bar"\n', {"env": {"FOO": "bar"}, "scripts": ["dotenv.sh"]}),
        (
            'scripts = ["scripts/activate.sh", "/opt/site.sh"]\n',
            {"scripts": ["dotenv.sh", "../scripts/activate.sh", "/opt/site.sh"]},
        ),
        (
            'dotenv = false\nscripts = ["scripts/activate.sh"]\n',
            {"scripts": ["../scripts/activate.sh"]},
        ),
        # A cleared variable cannot ride in pixi's own string-to-string env table, so it becomes
        # the unset script, sourced after the dotenv loader so the clear beats a `.env` fill.
        (
            "\n[env]\nOMP_NUM_THREADS = false\n",
            {"scripts": ["dotenv.sh", "unset.sh"]},
        ),
        (
            '\n[env]\nFOO = "bar"\nPIP_CONSTRAINT = false\n',
            {"env": {"FOO": "bar"}, "scripts": ["dotenv.sh", "unset.sh"]},
        ),
    ],
)
def test_the_activation_table_sources_the_dotenv_loader_before_declared_scripts(
    header: str, table: dict[str, Toml], manifest_from: Callable[[str], Manifest]
) -> None:
    """A declared script is workspace-relative, so it is rerooted like any other location."""
    manifest = manifest_from(f'[workspace]\nname = "w"\n{header}')
    assert PixiManifest.activation_table(manifest) == table


@pytest.mark.parametrize(
    ("declared", "isolated", "shared"),
    [
        pytest.param(
            "OMP_NUM_THREADS = false",
            {"scripts": ["unset.sh"]},
            None,
            id="an-isolated-environment-carries-the-workspace-clears-itself",
        ),
        pytest.param(
            'OMP_NUM_THREADS = "8"',
            None,
            None,
            id="a-setting-stays-with-the-default-feature-an-isolated-env-opted-out-of",
        ),
    ],
)
def test_a_cleared_variable_reaches_an_environment_that_excludes_the_default_feature(
    declared: str,
    isolated: dict[str, Toml] | None,
    shared: dict[str, Toml] | None,
    manifest_from: Callable[[str], Manifest],
) -> None:
    """`no-default` is a statement about the solve, and pixi applies it to activation too.

    So an environment declaring it saw neither the workspace's settings nor its clears, and a
    `[env] X = false` meant to guarantee a clean arithmetic environment silently stopped at the
    one environment the GPU work actually runs under. A clear says the variable must not be
    present at all, which isolation is a reason to honour rather than to skip, while a setting
    is exactly what an isolated environment opted out of.
    """
    manifest = manifest_from(
        f'[workspace]\nname = "w"\n[env]\n{declared}\n'
        "[envs.vserve]\nno-default = true\n[envs.tools]\n"
    )
    feature, _ = PixiManifest.features(manifest, PlatformMatrix.from_manifest(manifest), _PROJECT)
    assert feature["vserve"].get("activation") == isolated
    assert feature["tools"].get("activation") == shared


@settings(max_examples=10)
@given(conda=_TABLE, python=_TABLE, dev=_TABLE, served=_TABLE)
def test_every_declared_dependency_reaches_exactly_one_generated_table(
    conda: dict[str, str],
    python: dict[str, str],
    dev: dict[str, str],
    served: dict[str, str],
) -> None:
    """Nothing declared is lost on the way into pixi's tables, and nothing else is invented."""
    manifest = Manifest.model_validate(
        {
            "workspace": {"name": "w"},
            "deps": conda,
            "python": {"deps": python},
            "dev": {"deps": dev},
            "envs": {"serving": {"python": {"deps": served}}},
        }
    )
    document = tomllib.loads(PixiManifest.from_manifest(manifest, project_name=_PROJECT).to_toml())
    feature = document.get("feature", {})
    assert document.get("dependencies", {}).keys() == conda.keys()
    assert document.get("pypi-dependencies", {}).keys() == python.keys()
    assert feature.get("dev", {}).get("dependencies", {}).keys() == dev.keys()
    assert feature.get("serving", {}).get("pypi-dependencies", {}).keys() == served.keys()


def test_a_dependency_literally_named_path_keeps_its_version(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """Only a `path` carried as a dependency *source* is a location to be rerooted."""
    manifest = manifest_from('[workspace]\nname = "w"\n[deps]\npath = "*"\n')
    assert PixiManifest.from_manifest(manifest, project_name=_PROJECT).dependencies["path"] == "*"
