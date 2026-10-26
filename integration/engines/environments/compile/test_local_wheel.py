from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard import Manifest
from mainboard.engines.compile import Provisioner
from mainboard.engines.compile.pixi.paths import local_sources, self_installed

type _Source = tuple[str, str | None, bool]

_SHARD = PurePosixPath(".mainboard/envs/default")
_HEADERS = [
    "pypi-dependencies",
    "dependencies",
    "pypi-options.dependency-overrides",
    "feature.extra.pypi-dependencies",
    "feature.extra.target.linux-64.pypi-dependencies",
    "target.linux-64.dependencies",
]
_INSIDE = st.one_of(st.just(""), st.from_regex(r"[a-z]{1,4}(/[a-z]{1,4}){0,2}", fullmatch=True))
_SOURCE = st.one_of(
    st.tuples(_INSIDE, st.booleans()).map(
        lambda at: (f"../../../{at[0]}".rstrip("/"), at[0], at[1])
    ),
    st.tuples(st.sampled_from(["/outside/example.whl", "../short", "./near"]), st.booleans()).map(
        lambda at: (at[0], None, at[1])
    ),
)
_TABLES = st.fixed_dictionaries(
    {header: st.dictionaries(st.sampled_from("abc"), _SOURCE, max_size=3) for header in _HEADERS}
).map(lambda tables: {header: tables[header] for header in _HEADERS if tables[header]})
_WHEEL = "wheels/example.whl"
_LOCAL = "packages/local"
_ONCE = {
    "pypi-dependencies": {
        "a": (f"../../../{_WHEEL}", _WHEEL, False),
        "b": (f"../../../{_LOCAL}", _LOCAL, True),
    },
    "feature.extra.target.linux-64.pypi-dependencies": {
        "a": (f"../../../{_WHEEL}", _WHEEL, False),
        "c": ("/outside/example.whl", None, False),
    },
}


@pytest.fixture
def wheel(tmp_path: Path) -> str:
    """A local wheel file in the workspace, as the workspace-relative path that names it."""
    path = tmp_path / "wheels/example-1.0-py3-none-any.whl"
    path.parent.mkdir()
    path.write_bytes(b"wheel fixture")
    return path.relative_to(tmp_path).as_posix()


@pytest.fixture
def compiled(tmp_path: Path, wheel: str) -> Provisioner:
    """A workspace depending on `wheel`, compiled."""
    manifest = Manifest.model_validate(
        {
            "workspace": {"name": "wheel-test", "platforms": ["linux-64"]},
            "python": {"deps": {"example": {"path": wheel}}},
        }
    )
    provisioner = Provisioner(tmp_path, manifest)
    provisioner.recompiled()
    return provisioner


def test_local_files_are_shipped_without_becoming_editable_roots(
    compiled: Provisioner, wheel: str
) -> None:
    text = compiled.pixi.manifest.read_text(encoding="utf-8")
    shard = compiled.environment_dir().relative_to(compiled.root)

    assert wheel in compiled.artifact
    # Linux raises NotADirectoryError for wheel/pyproject.toml; no metadata is owed there.
    assert len(compiled.compiler.resolution_digest()) == 64
    assert local_sources(text, generated_dir=shard) == [wheel]
    assert self_installed(text, generated_dir=shard) == []


@settings(max_examples=200)
@example(tables=_ONCE)
@given(tables=_TABLES)
def test_workspace_sources_are_listed_once_across_every_table_and_only_editable_ones_install(
    tables: Mapping[str, Mapping[str, _Source]],
) -> None:
    manifest = "".join(
        f"[{header}]\n"
        + "".join(
            f'{name} = {{path = "{spelled}", editable = {str(editable).lower()}}}\n'
            for name, (spelled, _, editable) in entries.items()
        )
        for header, entries in tables.items()
    )
    sources = [source for entries in tables.values() for source in entries.values()]
    inside = [(rel, editable) for _, rel, editable in sources if rel is not None]

    assert local_sources(manifest, generated_dir=_SHARD) == list(
        dict.fromkeys(r for r, _ in inside)
    )
    assert self_installed(manifest, generated_dir=_SHARD) == list(
        dict.fromkeys(r for r, editable in inside if editable)
    )
