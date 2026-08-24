from pathlib import Path

import pytest
import tomlkit
from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard import Manifest, MissionError, Project, load
from mainboard.manifest import EnvMode, Guardrail, Header, HostProfile, Spec

from ..strategies import PATHS, SPECS, WORDS

# A manifest built from its own schema, over the tables a round trip can compare field by field.
# Every alphabet is tame on purpose, since a `{{` in a generated value would render at load and
# the identity under test is the one a workspace with no templates already relies on.
_MANIFESTS = st.builds(
    Manifest,
    workspace=st.builds(Header, name=WORDS, platforms=st.lists(WORDS, max_size=2, unique=True)),
    vars=st.dictionaries(WORDS, WORDS, max_size=3),
    deps=st.dictionaries(WORDS, st.builds(Spec, version=SPECS), max_size=3),
    hosts=st.dictionaries(WORDS, st.builds(HostProfile, kind=WORDS, scratch=PATHS), max_size=2),
)


def test_load_renders_and_validates_the_full_fixture(loaded: Manifest) -> None:
    """One pass through tomllib, the `{{ }}` rendering, and the schema, in that order."""
    assert loaded.workspace.name == "lab"
    assert loaded.vars["scratch"] == "/scratch/lab"
    assert loaded.vars["station"].startswith(("linux-", "macos-"))
    assert loaded.containers["ngc"].binds == ["/scratch/lab"]
    assert loaded.containers["ngc"].env_mode is EnvMode.VENV_SYSTEM_SITE
    assert set(loaded.containers["ngc"].guardrails) == {
        Guardrail.UNSET_PIP_CONSTRAINT,
        Guardrail.PIN_SYSTEM_PACKAGES,
    }
    assert loaded.envs["serving"].system == {"cuda": "13.0"}
    assert loaded.profile("miyabi-g").defaults.mem_gb == "min(100, attempt * 50)"


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (None, "no manifest"),
        ("workspace = [broken", "not valid TOML"),
        ('[workspace]\nname = "lab"\n\n[vars]\nbroken = "{{ nope }}"\n', r"vars\.broken"),
        ('[workspace]\nname = "lab"\n\n[hosts.gold]\nkind = 3\n', "failed validation"),
    ],
)
def test_a_manifest_that_cannot_be_read_names_what_went_wrong(
    tmp_path: Path, body: str | None, match: str
) -> None:
    """A missing file, broken TOML, a template error and a schema error each name their spot."""
    path = tmp_path / Project().manifest
    if body is not None:
        path.write_text(body, encoding="utf-8")
    with pytest.raises(MissionError, match=match):
        load(path)


@settings(max_examples=15)
@given(manifest=_MANIFESTS)
def test_a_manifest_dumped_to_toml_loads_back_as_the_same_model(
    tmp_path: Path, manifest: Manifest
) -> None:
    """Every example writes and reloads a file, so the budget stays small on purpose."""
    path = tmp_path / Project().manifest
    path.write_text(tomlkit.dumps(manifest.model_dump(exclude_defaults=True)), encoding="utf-8")
    assert load(path) == manifest
