from pathlib import Path

import pytest
import tomlkit

from mainboard import Manifest, MissionError, Project, load
from mainboard.engines.compile.pixi_manifest import PixiManifest
from mainboard.manifest.loading import composition
from mainboard.manifest.members import Package

_MANIFEST = Project().manifest

_ROOT = """
[workspace]
name = "life"
members = ["packages/*", "research/*"]

[python]
index-strategy = "unsafe-best-match"

[python.deps]
lib = { extras = ["sql"] }
rich = "*"

[python.dependency-overrides]
torchaudio = "*"

[dev.python.deps]
Tool = "*"

[on.linux.python.deps]
cupy = ">=14"

[env]
LEVEL = "INFO"

[envs.serving.python.deps]
vllm = "*"

[tasks]
test = "pytest"
build = "make"

[papers.old]
dir = "papers/old"

[lint]
owners = ["apps/*"]

[lint.tools.ruff]
check = "ruff check {files}"
files = ["*.py"]
"""

_HEAD = """
[workspace]
name = "head"
platforms = ["linux-64"]

[system]
cuda = "13.0"

[python]
extra-index-urls = ["https://example.com/simple"]

[python.deps]
llm_head = { path = ".", editable = true }
lib = { path = "../../packages/lib", editable = true }
vllm = ">=0.30"
local = { path = "/opt/wheels/local" }

[dev.python.deps]
pytest = "*"

[on.linux.python.deps]
cupy = ">=13"
cuvs = "*"
llm_head = { path = ".", editable = true, extras = ["serve"] }

[on.osx.deps]
libomp = "*"

[env]
LEVEL = "DEBUG"
HEAD = "1"

[envs.cutile]
no-default = true

[envs.cutile.tasks]
kernel = { run = "python kernel.py", depends = ["figures", "warm"] }
warm = "python warm.py"

[tasks]
figures = { run = "python plot.py", dir = "papers" }
test = "pytest"
all = { depends = ["figures", "test", "build"] }

[papers.iclr]
dir = "papers/latex"
limit = 9

[hosts.gold]
kind = "ssh"

[lint]
exclude = ["datasets/", "!keep.csv", "/build/", "src/gen/*.py"]
max-kb = 10

[lint.tools.ruff]
check = "ruff check"
files = ["*.py"]

[lint.tools.clippy]
check = "cargo clippy"
files = ["*.rs"]
exclude = ["vendor/"]
"""

_TOOLS = """
[python.deps]
tool = ">=1"

[tasks]
test = "pytest -x"
lint = "ruff check"
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _pyproject(name: str, *dependencies: str) -> str:
    listed = ", ".join(f'"{dependency}"' for dependency in dependencies)
    return f'[project]\nname = "{name}"\ndependencies = [{listed}]\n'


@pytest.fixture
def life(tmp_path: Path) -> Path:
    """A root composing three libraries and two research projects.

    The root requires `lib` and, in dev, `tool`; nothing requires `extra`. `research/head`
    carries a manifest of its own and a pyproject naming `lib` by git URL; `research/tools` is a
    manifest alone requiring `tool` by version; `research/notes` is neither, so no member at all.
    """
    _write(tmp_path / _MANIFEST, _ROOT)
    _write(tmp_path / "packages" / "lib" / "pyproject.toml", _pyproject("Lib", "lib[docs]"))
    _write(tmp_path / "packages" / "tool" / "pyproject.toml", _pyproject("tool"))
    _write(tmp_path / "packages" / "extra" / "pyproject.toml", _pyproject("extra"))
    head = tmp_path / "research" / "head"
    _write(head / "pyproject.toml", _pyproject("llm_head", "lib[cli] @ git+https://x/lib"))
    _write(head / _MANIFEST, _HEAD)
    _write(tmp_path / "research" / "tools" / _MANIFEST, _TOOLS)
    (tmp_path / "research" / "notes").mkdir()
    return tmp_path


@pytest.fixture
def composed(life: Path) -> Manifest:
    return load(life / _MANIFEST)


def test_members_are_read_with_their_projects_and_root_only_settings(life: Path) -> None:
    members = {member.name: member for member in composition(life / _MANIFEST).members}

    assert list(members) == ["extra", "lib", "tool", "head", "tools"]
    assert members["lib"].manifest is None
    assert members["tools"].package is None
    head = members["head"]
    assert head.package == Package(name="llm-head", requires={"lib": frozenset({"cli"})})
    assert head.ignored == (
        "[hosts]",
        "[system]",
        "[python] extra-index-urls",
        "[lint] max-kb",
    )


def test_member_requirements_join_with_every_source_installed_editable_and_pinned(
    composed: Manifest,
) -> None:
    """Wherever a member is required it is its own source, and overrides pin it everywhere."""
    python = composed.toolchains()["python"]
    deps = {name: spec.model_dump(exclude_defaults=True) for name, spec in python.deps.items()}

    assert deps["lib"] == {"path": "packages/lib", "editable": True, "extras": ["sql"]}
    assert deps["llm-head"] == {"path": "research/head", "editable": True}
    assert "llm_head" not in deps
    assert deps["local"] == {"path": "/opt/wheels/local"}
    assert deps["Tool"] == {"path": "packages/tool", "editable": True}
    assert deps["extra"] == {"path": "packages/extra", "editable": True}
    assert python.model_extra == {
        "index-strategy": "unsafe-best-match",
        "dependency-overrides": {
            "extra": {"path": "packages/extra"},
            "lib": {"path": "packages/lib", "extras": ["cli", "sql"]},
            "llm-head": {"path": "research/head", "extras": ["serve"]},
            "Tool": {"path": "packages/tool"},
            "torchaudio": "*",
        },
    }
    dev = composed.dev.toolchains()["python"].deps
    assert dev["Tool"].model_extra == {"path": "packages/tool", "editable": True}
    assert set(dev) == {"Tool", "pytest"}
    assert composed.on["linux"].toolchains()["python"].deps["cupy"].version == ">=14"
    assert set(composed.on) == {"linux", "osx"}
    assert composed.env == {"LEVEL": "INFO", "HEAD": "1"}


def test_member_tasks_papers_and_environments_are_namespaced_and_rebased(
    composed: Manifest,
) -> None:
    """Bare names are kept for what nobody else takes; the root's own never move."""
    tasks = composed.tasks
    assert tasks["head:figures"] == {"run": "python plot.py", "dir": "research/head/papers"}
    assert tasks["figures"] == tasks["head:figures"]
    assert tasks["head:all"] == {"depends": ["head:figures", "head:test", "build"]}
    assert tasks["test"] == "pytest"
    assert tasks["tools:test"] == {"run": "pytest -x", "dir": "research/tools"}
    assert tasks["lint"] == tasks["tools:lint"]
    assert composed.papers["iclr"].dir == "research/head/papers/latex"
    assert composed.papers["head:iclr"].limit == 9
    assert composed.papers["old"].dir == "papers/old"
    cutile = composed.envs["cutile"]
    assert cutile.no_default
    assert cutile.tasks["head:kernel"] == {
        "run": "python kernel.py",
        "dir": "research/head",
        "depends": ["head:figures", "head:warm"],
    }
    assert set(composed.envs) == {"serving", "cutile"}


def test_member_lint_joins_anchored_to_the_member(composed: Manifest) -> None:
    lint = composed.lint
    assert lint.owners == (
        "apps/*",
        "packages/extra",
        "packages/lib",
        "packages/tool",
        "research/head",
        "research/tools",
    )
    assert lint.exclude == (
        "/research/head/**/datasets/",
        "!/research/head/**/keep.csv",
        "/research/head/build/",
        "/research/head/src/gen/*.py",
    )
    assert list(lint.tools) == ["ruff", "head:clippy"]
    assert lint.tools["ruff"].files == ("*.py",)
    assert lint.tools["head:clippy"].files == ("/research/head/**/*.rs",)
    assert lint.tools["head:clippy"].exclude == ("/research/head/**/vendor/",)


def test_the_composed_manifest_compiles_member_sources_tasks_and_overrides(
    composed: Manifest,
) -> None:
    compiled = tomlkit.parse(
        PixiManifest.from_manifest(composed, project_name="mainboard").to_toml()
    ).unwrap()
    assert compiled["pypi-dependencies"]["llm-head"] == {
        "path": "../research/head",
        "editable": True,
    }
    overrides = compiled["pypi-options"]["dependency-overrides"]
    assert overrides["lib"] == {"path": "../packages/lib", "extras": ["cli", "sql"]}
    assert compiled["tasks"]["head:figures"]["cwd"] == "../research/head/papers"


def test_a_workspace_without_members_is_its_own_manifest(tmp_path: Path) -> None:
    _write(tmp_path / _MANIFEST, '[workspace]\nname = "solo"\n[tasks]\nx = "echo"\n')
    loaded = composition(tmp_path / _MANIFEST)
    assert loaded.composed() is loaded.manifest


def test_members_without_packages_leave_the_overrides_alone(tmp_path: Path) -> None:
    _write(tmp_path / _MANIFEST, '[workspace]\nname = "life"\nmembers = ["tools"]\n')
    _write(tmp_path / "tools" / _MANIFEST, '[tasks]\ntest = "pytest -x"\nlint = "ruff check"\n')
    composed = load(tmp_path / _MANIFEST)
    assert composed.toolchains() == {}
    assert set(composed.tasks) == {"tools:test", "tools:lint", "test", "lint"}


@pytest.mark.parametrize(
    ("layout", "match"),
    [
        ({"a/x/pyproject.toml": _pyproject("x"), "b/x/pyproject.toml": _pyproject("y")}, "both"),
        ({f"a/x/{_MANIFEST}": "[envs.serving]\n"}, "which the root"),
        (
            {f"a/x/{_MANIFEST}": "[envs.gpu]\n", f"b/y/{_MANIFEST}": "[envs.gpu]\n"},
            "which a/x",
        ),
        ({"a/x/pyproject.toml": "[project"}, "not valid TOML"),
    ],
)
def test_composition_refuses_what_would_collide(
    tmp_path: Path, layout: dict[str, str], match: str
) -> None:
    root = '[workspace]\nname = "w"\nmembers = ["a/*", "b/*"]\n[envs.serving]\n'
    _write(tmp_path / _MANIFEST, root)
    for path, text in layout.items():
        _write(tmp_path / path, text)
    with pytest.raises(MissionError, match=match):
        load(tmp_path / _MANIFEST)


def test_a_package_reads_every_dependency_list_and_skips_what_is_not_a_project(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pyproject.toml"
    assert Package.read(path) is None
    _write(path, "[tool.ruff]\nline-length = 99\n")
    assert Package.read(path) is None
    _write(
        path,
        '[project]\nname = "Head.Lib"\nrequires-python = ">=3.14"\ndependencies = ["Torch>=2"]\n'
        "[project.optional-dependencies]\nserve = [\"vllm[audio]; sys_platform == 'linux'\"]\n"
        '[dependency-groups]\ndev = ["pytest", "VLLM[cli]", { include-group = "serve" }]\n',
    )
    assert Package.read(path) == Package(
        name="head-lib",
        python=">=3.14",
        requires={
            "torch": frozenset(),
            "vllm": frozenset({"audio", "cli"}),
            "pytest": frozenset(),
        },
    )
