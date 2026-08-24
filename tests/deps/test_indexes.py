import io
import json
from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import Request

import pytest

from mainboard import MissionError
from mainboard.deps import Index

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess

    from mainboard.manifest.schema.spec import Json

_SEARCH = {
    "linux-64": [{"name": "tqdm", "version": "4.38.0"}, {"name": "tqdm", "version": "4.70.0"}],
    "noarch": [{"name": "tqdm", "version": "4.69.1"}],
}


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> Callable[[Json | OSError], list[str]]:
    """Serve one JSON body, or one refusal, to every index request, collecting the urls asked."""
    asked: list[str] = []

    def install(body: Json | OSError) -> list[str]:
        def opened(request: Request, timeout: float = 0.0) -> io.BytesIO:
            asked.append(request.full_url)
            if isinstance(body, OSError):
                raise body
            return io.BytesIO(json.dumps(body).encode())

        monkeypatch.setattr("urllib.request.urlopen", opened)
        return asked

    return install


@pytest.mark.parametrize(
    ("ecosystem", "version", "expected"),
    [
        ("python", "1.2.3", ">=1.2.3, <2"),
        ("python", "0.1.0", ">=0.1.0, <0.2"),
        ("python", "0.0.5", ">=0.0.5, <0.0.6"),
        ("python", "0.0.0", ">=0.0.0, <0.0.1"),
        # A release the version grammar cannot read gets a floor and no ceiling, since a wider
        # pin still resolves and a refusal does not.
        ("conda", "nightly", ">=nightly"),
        # npm separates comparators by space and Go resolves one version, so neither reads the
        # comma-separated range conda and Python both take.
        ("nodejs", "1.2.3", "^1.2.3"),
        ("go", "1.2.3", "1.2.3"),
    ],
)
def test_each_ecosystem_writes_the_pin_its_own_resolver_reads(
    ecosystem: str, version: str, expected: str
) -> None:
    """pixi's own semver pinning, which is the shape the manifest already writes."""
    assert Index.of(ecosystem).pin(version) == expected


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
    answers: Callable[[Json | OSError], list[str]],
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
    answers: Callable[[Json | OSError], list[str]],
) -> None:
    """Three registries, three shapes, one answer each."""
    answers({"dist-tags": {"latest": "1.49.0"}})
    assert Index.of("nodejs").latest("es-toolkit") == "1.49.0"
    answers({"crate": {"max_stable_version": "14.1.1"}})
    assert Index.of("rust").latest("ripgrep") == "14.1.1"
    asked = answers({"Version": "v1.6.0"})
    assert Index.of("go").latest("github.com/x/y") == "1.6.0"
    assert asked[-1].endswith("/github.com/x/y/@latest")


@pytest.mark.parametrize(
    ("served", "match"),
    [
        (URLError("connection refused"), r"pypi\.org/simple/ghost/ would not answer"),
        ({"versions": []}, r"publishes no readable release of 'ghost'"),
    ],
)
def test_an_index_that_cannot_answer_refuses_by_name(
    answers: Callable[[Json | OSError], list[str]], served: Json | OSError, match: str
) -> None:
    """A registry that is down and one that lists nothing are both facts, not stack traces."""
    answers(served)
    with pytest.raises(MissionError, match=match):
        Index.of("python").latest("ghost")
