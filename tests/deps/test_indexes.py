import io
import json
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import Request

import pytest

from mainboard import MissionError
from mainboard.deps import Index

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_subprocess import FakeProcess

    from mainboard.manifest.schema.spec import Json

_SEARCH = {
    "linux-64": [{"name": "tqdm", "version": "4.38.0"}, {"name": "tqdm", "version": "4.70.0"}],
    "noarch": [{"name": "tqdm", "version": "4.69.1"}],
}


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> Callable[[Json], list[str]]:
    """Serve one JSON body to every index request, collecting the urls that asked."""
    asked: list[str] = []

    def install(body: Json) -> list[str]:
        def opened(request: Request, timeout: float = 0.0) -> io.BytesIO:
            asked.append(request.full_url)
            return io.BytesIO(json.dumps(body).encode())

        monkeypatch.setattr("urllib.request.urlopen", opened)
        return asked

    return install


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.2.3", ">=1.2.3, <2"),
        ("0.1.0", ">=0.1.0, <0.2"),
        ("0.0.5", ">=0.0.5, <0.0.6"),
        ("0.0.0", ">=0.0.0, <0.0.1"),
    ],
)
def test_a_pin_carries_the_release_and_the_compatible_ceiling_above_it(
    version: str, expected: str
) -> None:
    """pixi's own semver pinning, which is the shape the manifest already writes."""
    assert Index.of("python").pin(version) == expected


def test_a_release_the_version_grammar_cannot_read_gets_a_floor_and_no_ceiling() -> None:
    """A wider pin still resolves, and refusing to write one would not."""
    assert Index.of("conda").pin("nightly") == ">=nightly"


def test_npm_and_go_write_the_pin_their_own_resolvers_read() -> None:
    """A comma-separated range is invalid npm and no range at all is resolvable by Go."""
    assert Index.of("nodejs").pin("1.2.3") == "^1.2.3"
    assert Index.of("go").pin("1.2.3") == "1.2.3"


def test_an_ecosystem_with_no_index_refuses_with_the_roster() -> None:
    """The refusal names the way out, an explicit version, and the keys that do work."""
    with pytest.raises(MissionError, match=r"no release index for 'elixir'.*'python'"):
        Index.of("elixir")


def test_conda_reads_the_newest_release_across_every_subdir_pixi_answers_with(
    fp: FakeProcess,
) -> None:
    """One package is published per platform, and the newest of them all is the answer."""
    fp.register([fp.any()], stdout=json.dumps(_SEARCH))
    index = Index.of("conda")
    index.sources = ("rapidsai", "conda-forge")
    assert index.latest("tqdm") == "4.70.0"
    assert "--channel" in fp.calls[0]


def test_a_failing_conda_search_reaches_the_caller_as_a_refusal(fp: FakeProcess) -> None:
    """A search that cannot answer is a refusal, never a silently missing version."""
    fp.register([fp.any()], returncode=1, stderr="no such package\n")
    with pytest.raises(MissionError, match="tqdm"):
        Index.of("conda").latest("tqdm")


def test_python_reads_the_declared_index_before_falling_back_to_pypi(
    answers: Callable[[Json], list[str]],
) -> None:
    """A workspace pointing at a mirror is asked about the mirror, not about PyPI."""
    asked = answers({"versions": ["4.69.1", "4.70.0", "not-a-version"]})
    index = Index.of("python")
    index.sources = ("https://mirror.internal/simple/",)
    assert index.latest("tqdm") == "4.70.0"
    assert asked == ["https://mirror.internal/simple/tqdm/"]
    assert Index.of("python").latest("tqdm") == "4.70.0"
    assert asked[1] == "https://pypi.org/simple/tqdm/"


def test_npm_rust_and_go_each_read_their_own_registry_document(
    answers: Callable[[Json], list[str]],
) -> None:
    """Three registries, three shapes, one answer each."""
    answers({"dist-tags": {"latest": "1.49.0"}})
    assert Index.of("nodejs").latest("es-toolkit") == "1.49.0"
    answers({"crate": {"max_stable_version": "14.1.1"}})
    assert Index.of("rust").latest("ripgrep") == "14.1.1"
    asked = answers({"Version": "v1.6.0"})
    assert Index.of("go").latest("github.com/x/y") == "1.6.0"
    assert asked[-1].endswith("/github.com/x/y/@latest")


def test_an_index_that_will_not_answer_says_so_with_its_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry that is down is reported as a registry, not as a stack trace."""

    def refuse(request: Request, timeout: float = 0.0) -> None:
        raise URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(MissionError, match=r"pypi.org/simple/tqdm/ would not answer"):
        Index.of("python").latest("tqdm")


def test_an_index_publishing_nothing_readable_refuses_by_name(
    answers: Callable[[Json], list[str]],
) -> None:
    """An empty listing is a fact about the name, and the refusal says which name."""
    answers({"versions": []})
    with pytest.raises(MissionError, match=r"publishes no readable release of 'ghost'"):
        Index.of("python").latest("ghost")
