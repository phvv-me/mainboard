from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.manifest.render.interpolate import Interpolator


def test_vocabulary_covers_the_mise_names(tmp_path: Path) -> None:
    tree = {
        "vars": {"home": "{{ config_root }}", "cpus": "{{ num_cpus() }}"},
        "line": "{{ os_name() }}/{{ arch() }} at {{ vars.home }}",
    }
    rendered = Interpolator(tmp_path).rendered(tree)
    assert rendered["vars"]["home"] == str(tmp_path)
    assert int(rendered["vars"]["cpus"]) >= 1
    assert str(rendered["line"]).endswith(str(tmp_path))


def test_env_reads_the_environment_with_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MC_RENDER_TEST", "live")
    tree = {"a": "{{ env('MC_RENDER_TEST') }}", "b": "{{ env('MC_RENDER_MISSING', 'fb') }}"}
    rendered = Interpolator(tmp_path).rendered(tree)
    assert rendered == {"a": "live", "b": "fb", "vars": {}}


def test_exec_returns_stdout_and_fails_loudly(tmp_path: Path) -> None:
    rendered = Interpolator(tmp_path).rendered({"who": "{{ exec('echo mission') }}"})
    assert rendered["who"] == "mission"
    with pytest.raises(MissionError, match="exec"):
        Interpolator(tmp_path).rendered({"bad": "{{ exec('false') }}"})


def test_lists_and_scalars_walk_untouched(tmp_path: Path) -> None:
    tree = {"vars": {"x": "1"}, "list": ["{{ vars.x }}", 2, True], "n": 3}
    rendered = Interpolator(tmp_path).rendered(tree)
    assert rendered["list"] == ["1", 2, True]
    assert rendered["n"] == 3


def test_vars_must_be_a_table(tmp_path: Path) -> None:
    with pytest.raises(MissionError, match=r"\[vars\] must be a table"):
        Interpolator(tmp_path).rendered({"vars": "nope"})
