import json
from unittest.mock import patch

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from mainboard.deps import Index

_FILES = ["tool-{}.tar.gz", "tool-{}-py3-none-any.whl"]
_PYPI = ((), "https://pypi.org/simple")
_ELSEWHERE = (("https://example.invalid/simple/",), "https://example.invalid/simple")
_RELEASE = st.builds(
    lambda parts, pre: ".".join(map(str, parts)) + pre,
    st.tuples(st.integers(0, 9), st.integers(0, 9), st.integers(0, 9)),
    st.sampled_from(["", "", "rc1", "b2"]),
)
_FLAGS = st.lists(st.sampled_from([False, True, "withdrawn"]), max_size=3)
_LISTINGS = st.dictionaries(_RELEASE, _FLAGS, min_size=1, max_size=6).filter(
    lambda releases: any(not flags or not all(flags) for flags in releases.values())
)


@pytest.mark.parametrize(
    ("ecosystem", "version", "pin"),
    [
        pytest.param("python", "2.4.0", ">=2.4.0, <3", id="python-keeps-the-major"),
        pytest.param("conda", "1.2.3", ">=1.2.3, <2", id="conda-keeps-the-major"),
        pytest.param("python", "0.1.0", ">=0.1.0, <0.2", id="a-zero-major-keeps-the-minor"),
        pytest.param("python", "0.0.3", ">=0.0.3, <0.0.4", id="a-zero-minor-keeps-the-patch"),
        pytest.param("python", "20260107", ">=20260107", id="a-lone-date-has-only-a-floor"),
        pytest.param("python", "build-7", ">=build-7", id="an-unreadable-release-has-a-floor"),
        pytest.param("nodejs", "1.2.3", "^1.2.3", id="npm-takes-a-caret"),
        pytest.param("rust", "0.3.14", "0.3.14", id="cargo-takes-the-exact-release"),
        pytest.param("go", "1.2.3", "1.2.3", id="go-takes-the-exact-release"),
    ],
)
def test_each_ecosystem_spells_a_pin_the_way_its_resolver_reads_one(
    ecosystem: str, version: str, pin: str
) -> None:
    assert Index.of(ecosystem).pin(version) == pin


@settings(max_examples=300)
@example(release=[2, 4, 0])
@example(release=[0, 1, 0])
@example(release=[0, 0, 0])
@example(release=[0, 0, 3, 7])
@given(release=st.lists(st.integers(0, 999), min_size=1, max_size=4))
def test_a_pin_admits_the_release_and_what_stays_compatible_with_it_but_not_the_next_break(
    release: list[int],
) -> None:
    version = ".".join(map(str, release))
    cut = next((at for at, part in enumerate(release) if part), len(release) - 1)
    bumped = [[*release[:at], release[at] + 1] for at in range(len(release))]
    admitted = SpecifierSet(Index.of("python").pin(version))

    assert Version(version) in admitted
    assert all(".".join(map(str, later)) in admitted for later in bumped[cut + 1 :])
    assert ".".join(map(str, bumped[cut])) not in admitted


@settings(max_examples=200)
@example(
    releases={
        "2.4.0": [False],
        "2.5.0": ["accidental breaking change", True],
        "2.6.0rc1": [False],
    },
    named=("tool", "tool"),
    where=_PYPI,
)
@given(
    releases=_LISTINGS,
    named=st.sampled_from([("tool", "tool"), ("tool[cu13]", "tool")]),
    where=st.sampled_from([_PYPI, _ELSEWHERE]),
)
def test_the_newest_release_is_the_newest_final_one_that_keeps_an_unyanked_file(
    releases: dict[str, list[bool | str]],
    named: tuple[str, str],
    where: tuple[tuple[str, ...], str],
) -> None:
    files = [
        {"filename": "notes.zip", "yanked": True},
        *(
            {"filename": _FILES[at % 2].format(version), "yanked": yanked}
            for version, flags in releases.items()
            for at, yanked in enumerate(flags)
        ),
    ]
    body = json.dumps({"versions": list(releases), "files": files}).encode()
    offered = [Version(v) for v, flags in releases.items() if not flags or not all(flags)]
    (sources, base), (requested, listed) = where, named
    python = Index.of("python")
    python.sources = sources

    with patch("urllib.request.urlopen", **{"return_value.read.return_value": body}) as opened:
        newest = python.latest(requested)

    assert newest == str(max([v for v in offered if not v.is_prerelease] or offered))
    assert [
        (call.args[0].full_url, call.args[0].get_header("Accept"))
        for call in opened.call_args_list
    ] == [(f"{base}/{listed}/", "application/vnd.pypi.simple.v1+json")]
