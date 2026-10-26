import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from mainboard import Manifest
from mainboard.engines.compile.backend import Pixi
from mainboard.engines.compile.pixi.manifest import PixiManifest, pypi_options, selected_manifest
from mainboard.manifest.members import Composition

_OVERRIDE = ">=13.8.1.7,<14"
_MEMBER = """[workspace]
name = "pkg"

[envs.x]
platforms = ["osx-arm64"]

[envs.x.python.deps]
httpx = ">=0.28"

[envs.x.tasks]
serve = "python -m pkg"
"""
_ROOT = f"""[workspace]
name = "it"
members = ["pkg"]

[system]
libc = "2.34"

[deps]
python = ">=3.14"

[python.deps]
pkg = {{ path = "pkg", editable = true }}

[envs.x]
no-default = true
platforms = ["linux-64"]

[envs.x.python.deps]
pkg = {{ path = "pkg", editable = true }}

[envs.x.python.dependency-overrides]
nvidia-cublas = "{_OVERRIDE}"
"""
_EDITABLE = '[pypi-dependencies]\npkg = { path = "pkg", editable = true }\n'
_NESTED = (
    '[feature.gpu.target.linux-64.pypi-dependencies]\npkg = { path = "pkg", editable = true }\n'
)
_PLAIN = '[pypi-dependencies]\npkg = { path = "pkg", editable = false }\n'


@pytest.fixture
def composed(tmp_path: Path) -> Manifest:
    """A root with one editable member and a `no-default` env declaring only an override."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "pyproject.toml").write_text(
        '[project]\nname = "pkg"\nversion = "0.0.0"\nrequires-python = ">=3.14"\n',
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "mainboard.toml").write_text(_MEMBER, encoding="utf-8")
    (tmp_path / "mainboard.toml").write_text(_ROOT, encoding="utf-8")
    return Composition(tmp_path, Manifest.model_validate(tomllib.loads(_ROOT))).composed()


@pytest.fixture
def lowered(composed: Manifest) -> Mapping[str, str | Mapping[str, str | bool]]:
    """The dependency overrides pixi gets for the `x` environment's feature."""
    shard = PixiManifest.from_manifest(composed, project_name="it", environment="x")
    feature = shard.model_dump(mode="python", round_trip=True)["feature"]["x"]
    return feature["pypi-options"]["dependency-overrides"]


def test_a_no_default_environment_joins_its_member_keeps_the_floors_and_lowers_its_override(
    composed: Manifest, lowered: Mapping[str, str | Mapping[str, str | bool]]
) -> None:
    environment = composed.envs["x"]
    assert environment.no_default and environment.platforms == ["linux-64"], (
        "the root's settings win"
    )
    assert {"pkg", "httpx"} <= environment.model_dump(mode="python")["python"]["deps"].keys()
    assert "pkg:serve" in environment.tasks, "a member's task keeps its member's namespace"
    assert selected_manifest(composed, "x").system == {"libc": "2.34"}

    overrides = pypi_options(environment)["dependency-overrides"]
    assert isinstance(overrides, Mapping) and overrides["nvidia-cublas"] == _OVERRIDE
    assert lowered["nvidia-cublas"] == _OVERRIDE
    assert "path" in lowered["pkg"], "the member path-pin must stay unioned under the override"


@pytest.mark.parametrize(
    ("compiled", "names", "argv"),
    [
        pytest.param(
            _EDITABLE,
            (),
            ("update", "-e", "default"),
            id="an-editable-source-drops-no-install",
        ),
        pytest.param(
            _NESTED,
            ("six", "httpx"),
            ("update", "-e", "default", "six", "httpx"),
            id="an-editable-source-in-a-target-drops-it-too",
        ),
        pytest.param(
            _PLAIN,
            (),
            ("update", "--no-install", "-e", "default"),
            id="a-source-that-is-not-editable-keeps-it",
        ),
        pytest.param(
            None,
            ("six",),
            ("update", "--no-install", "-e", "default", "six"),
            id="nothing-compiled-keeps-it",
        ),
    ],
)
def test_update_drops_no_install_only_while_a_source_must_be_built_to_be_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled: str | None,
    names: tuple[str, ...],
    argv: tuple[str, ...],
) -> None:
    pixi = Pixi(tmp_path)
    if compiled is not None:
        pixi.manifest.write_text(compiled, encoding="utf-8")
    seen: list[tuple[str, ...]] = []

    def within_cwd(action: str, *args: str, **flags: str) -> SimpleNamespace:
        seen.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pixi, "within_cwd", within_cwd)
    pixi.update("default", names)

    assert seen == [argv]
