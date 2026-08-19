import pytest

from mainboard import Manifest, MissionError
from mainboard.deps import candidates, declared


def headings(ecosystem: str, env: str, dev: bool) -> list[str]:
    """The candidate table headings for one set of flags, in preference order."""
    return [slot.table for slot in candidates(ecosystem=ecosystem, env=env, dev=dev)]


@pytest.mark.parametrize(
    ("ecosystem", "env", "dev", "expected"),
    [
        ("conda", "", False, ["[deps]"]),
        ("conda", "", True, ["[dev.deps]"]),
        ("conda", "serving", False, ["[envs.serving.deps]"]),
        ("python", "", False, ["[python.deps]"]),
        ("python", "", True, ["[dev.python.deps]", "[python.dev]"]),
        ("python", "serving", False, ["[envs.serving.python.deps]"]),
        ("nodejs", "serving", True, ["[envs.serving.nodejs.dev]"]),
    ],
)
def test_candidates_address_every_table_shape_a_manifest_writes(
    ecosystem: str, env: str, dev: bool, expected: list[str]
) -> None:
    """Each combination of resolver, environment and dev reaches the tables house style uses."""
    assert headings(ecosystem, env, dev) == expected


def test_an_environment_has_no_conda_development_table() -> None:
    """The schema gives an env no dev scope, so the refusal names where the entry belongs."""
    with pytest.raises(MissionError, match=r"no conda development table.*envs.serving.deps"):
        candidates(ecosystem="conda", env="serving", dev=True)


def test_declared_finds_every_table_across_every_scope(manifest: Manifest) -> None:
    """Root, dev, platform overlay, environment and an environment's own overlay all report."""
    found = {slot.table: names for slot, names in declared(manifest).items()}
    assert found["[deps]"] == ("python", "pueue")
    assert found["[dev.deps]"] == ("protobuf",)
    assert found["[dev.python.deps]"] == ("pytest",)
    assert found["[nodejs.dev]"] == ("@puppeteer/browsers",)
    assert found["[on.linux-64.deps]"] == ("cuda-version",)
    assert found["[envs.serving.nodejs.dev]"] == ("vite",)
    assert found["[envs.serving.on.linux-64.python.deps]"] == ("flashinfer",)


def test_declared_carries_the_resolver_that_reads_each_table(manifest: Manifest) -> None:
    """A slot knows its own ecosystem, which is what tells `upgrade` which index to ask."""
    by_table = {slot.table: slot.ecosystem for slot in declared(manifest)}
    assert by_table["[deps]"] == "conda"
    assert by_table["[envs.serving.python.deps]"] == "python"
    assert by_table["[nodejs.deps]"] == "nodejs"


def test_declared_leaves_out_tables_carrying_nothing() -> None:
    """A table declaring no requirement is not a place anything is declared."""
    bare = Manifest.model_validate(
        {"workspace": {"name": "bare"}, "deps": {}, "python": {"deps": {}, "dev": {}}}
    )
    assert declared(bare) == {}
