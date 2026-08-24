from collections.abc import Callable

import pytest

from mainboard import MissionError
from mainboard.engines import Apptainer, ContainerRuntime, Docker, Podman
from mainboard.engines.runtimes import resolve


@pytest.mark.parametrize(
    ("installed", "declared", "runtime"),
    [
        pytest.param(
            ("podman",), "auto", Podman, id="auto-picks-the-first-runtime-this-host-exposes"
        ),
        pytest.param((), "docker", Docker, id="an-explicit-name-ignores-availability"),
    ],
)
def test_resolve_names_the_runtime_a_manifest_asks_for(
    installed: tuple[str, ...],
    declared: str,
    runtime: type[ContainerRuntime],
    which: Callable[..., None],
) -> None:
    which(*installed)
    assert resolve(declared) is runtime


@pytest.mark.parametrize(
    ("declared", "refusal"),
    [
        pytest.param(
            "bogus",
            r"known runtimes are \['apptainer', 'docker', 'podman'\]",
            id="a-name-nobody-declared",
        ),
        pytest.param(
            "auto", "no container runtime available for 'auto'", id="a-host-running-none-of-them"
        ),
    ],
)
def test_resolve_refuses_with_the_declared_roster(
    declared: str, refusal: str, which: Callable[..., None]
) -> None:
    which()
    with pytest.raises(MissionError, match=refusal):
        resolve(declared)


@pytest.mark.parametrize(
    "runtime",
    [
        pytest.param(Docker, id="docker"),
        pytest.param(Podman, id="podman"),
        pytest.param(Apptainer, id="apptainer"),
    ],
)
def test_availability_is_whether_the_binary_is_on_path(
    runtime: type[ContainerRuntime], which: Callable[..., None]
) -> None:
    which()
    assert not runtime.is_available()
    which(runtime.binary)
    assert runtime.is_available()
