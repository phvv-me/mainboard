import json
import shlex
import sys
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.manifest.render.interpolate import Interpolator


def test_the_vocabulary_covers_the_mise_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`[vars]` renders in declaration order and reaches every string the tree carries."""
    monkeypatch.setenv("MC_RENDER_TEST", "live")
    tree = {
        "vars": {"home": "{{ config_root }}", "cpus": "{{ num_cpus() }}"},
        "line": "{{ os_name() }}/{{ arch() }} at {{ vars.home }}",
        "read": "{{ env('MC_RENDER_TEST') }}/{{ env('MC_RENDER_MISSING', 'fb') }}",
        "list": ["{{ vars.home }}", 2, True],
        "plain": 3,
    }
    rendered = Interpolator(tmp_path).rendered(tree)
    assert rendered["vars"]["home"] == str(tmp_path)
    assert int(rendered["vars"]["cpus"]) >= 1
    assert str(rendered["line"]).endswith(str(tmp_path))
    assert rendered["read"] == "live/fb"
    assert rendered["list"] == [str(tmp_path), 2, True]
    assert rendered["plain"] == 3


def test_exec_returns_stdout_and_a_failure_names_the_command(tmp_path: Path) -> None:
    """A shell-free command is in the vocabulary, so a failure still has to be readable."""
    command = shlex.join((sys.executable, "-c", "print('mission')"))
    template = f"{{{{ exec({json.dumps(command)}) }}}}"
    assert Interpolator(tmp_path).rendered({"who": template}) == {
        "who": "mission",
        "vars": {},
    }
    with pytest.raises(MissionError, match="exec"):
        Interpolator(tmp_path).rendered(
            {"bad": "{{ exec('mainboard-command-that-does-not-exist') }}"}
        )


def test_vars_must_be_a_table(tmp_path: Path) -> None:
    """Everything else reads `vars.*` as a mapping, so a scalar here is caught at the source."""
    with pytest.raises(MissionError, match=r"\[vars\] must be a table"):
        Interpolator(tmp_path).rendered({"vars": "nope"})
