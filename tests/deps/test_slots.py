import pytest

from mainboard import Manifest, MissionError
from mainboard.deps import candidates, declared


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
    assert [slot.table for slot in candidates(ecosystem=ecosystem, env=env, dev=dev)] == expected


def test_an_environment_has_no_conda_development_table() -> None:
    """The schema gives an env no dev scope, so the refusal names where the entry belongs."""
    with pytest.raises(MissionError, match=r"no conda development table.*envs.serving.deps"):
        candidates(ecosystem="conda", env="serving", dev=True)


def test_declared_reports_every_table_that_carries_a_requirement_with_its_own_resolver(
    manifest: Manifest,
) -> None:
    """Root, dev, platform overlay, environment and an environment's own overlay all report."""
    found = {slot.table: (slot.ecosystem, names) for slot, names in declared(manifest).items()}
    assert found["[deps]"] == ("conda", ("python", "pueue"))
    assert found["[dev.deps]"] == ("conda", ("protobuf",))
    assert found["[dev.python.deps]"] == ("python", ("pytest",))
    assert found["[nodejs.deps]"] == ("nodejs", ("es-toolkit",))
    assert found["[nodejs.dev]"] == ("nodejs", ("@puppeteer/browsers",))
    assert found["[on.linux-64.deps]"] == ("conda", ("cuda-version",))
    assert found["[envs.serving.python.deps]"] == ("python", ("vllm",))
    assert found["[envs.serving.nodejs.dev]"] == ("nodejs", ("vite",))
    assert found["[envs.serving.on.linux-64.python.deps]"] == ("python", ("flashinfer",))
    bare = Manifest.model_validate(
        {"workspace": {"name": "bare"}, "deps": {}, "python": {"deps": {}, "dev": {}}}
    )
    assert declared(bare) == {}
