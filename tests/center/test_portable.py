import os
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from mainboard.center.portable import RULES, Portability
from mainboard.core.section import Verdict
from mainboard.git.repo import Repo

from ..git.conftest import Forge

# One command line per rule that fires that rule and no other, the forms scripts really use.
_SAMPLES = {
    "sed -i": "sed -i 's/a/b/' notes.txt",
    "find -printf": "find . -name '*.py' -printf '%p'",
    "grep -P": "grep -P 'a+' notes.txt",
    "timeout": "timeout 5 make",
    "xargs -r": "ls | xargs -r echo",
    "readlink -f": "readlink -f here",
    "stat -c/-f": "stat -c %s notes.txt",
    "date -d": "date -d yesterday",
    "pkill/killall": "pkill python",
    "ps aux": "ps aux",
    "jq": "cat a.json | jq .name",
    "flock": "flock /tmp/lock make",
    "nc -z": "nc -z localhost 22",
    "sleep loop": "while true; do sleep 1; done",
}


def tracked(root: Path, files: Mapping[str, str | bytes], name: str = ".") -> Repo:
    """A git repository at `root` tracking `files`, as the tree would hand it to a scan."""
    root.mkdir(parents=True, exist_ok=True)
    Forge.git(root, "init", "-q")
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    Forge.git(root, "add", "-A")
    return Repo(root, name=name, owns=lambda owner: True)


def test_every_rule_has_a_sample_that_fires_it_alone() -> None:
    """The samples below name every rule, so the property over them reaches each one."""
    assert set(_SAMPLES) == {rule.name for rule in RULES}


@given(chosen=st.sets(st.sampled_from(sorted(_SAMPLES))))
def test_a_scan_warns_once_per_divergent_form_it_finds_and_passes_when_there_is_none(
    tmp_path: Path, chosen: set[str]
) -> None:
    """Each form found is one warning naming every place and its portable replacement.

    A script holding exactly the chosen forms is warned about exactly those, each at the line
    it sits on, in the order the rules are declared, and one with none of them is a single pass.
    """
    lines = ["#!/bin/sh", *(_SAMPLES[name] for name in sorted(chosen))]
    repo = tracked(tmp_path / "work", {"run.sh": "\n".join(lines) + "\n"})

    rows = Portability([repo]).sections()

    if not chosen:
        assert [(row.section, row.verdict) for row in rows] == [("portable", Verdict.PASS)]
        return
    fixes = {rule.name: rule.replacement for rule in RULES}
    order = [rule.name for rule in RULES if rule.name in chosen]
    assert [row.section for row in rows] == [f"portable: {name}" for name in order]
    for row, name in zip(rows, order, strict=True):
        assert row.verdict == Verdict.WARN
        assert row.detail == f"1 uses: run.sh:{lines.index(_SAMPLES[name]) + 1}"
        assert row.fix == fixes[name]


def test_only_live_command_lines_in_tracked_scripts_of_owned_repos_are_read(
    tmp_path: Path,
) -> None:
    """What a scan must not read, all in one tree, beside what it must.

    Comment lines of every script language, files that are not scripts, excluded paths (named
    from the workspace root, submodule prefix included), text that is not UTF-8, files too big
    to be scripts, links and untracked files are all skipped. A submodule's places carry its
    prefix, and a form found often is cut to three places and a count.
    """
    kill = "pkill python\n"
    root = tracked(
        tmp_path / "work",
        {
            "run.sh": "# pkill python\n  // pkill\nREM pkill\n:: pkill\n" + kill * 4,
            "sub/Makefile": kill,
            ".github/workflows/ci.yml": kill,
            "README.md": kill,
            "frozen/old.sh": kill,
            "latin.sh": "pkill caf\xe9\n".encode("latin-1"),
            "huge.sh": kill + "#" * (1 << 20),
        },
    )
    with suppress(OSError):  # a Windows account without Developer Mode cannot link
        os.symlink("run.sh", tmp_path / "work" / "linked.sh")
        Forge.git(tmp_path / "work", "add", "linked.sh")
    (tmp_path / "work" / "untracked.sh").write_text(kill, encoding="utf-8")
    lib = tracked(
        tmp_path / "work" / "packages" / "lib",
        {"tools/go.sh": kill, "vendor/x.sh": kill},
        name="packages/lib",
    )

    scan = Portability([root, lib], exclude=["frozen/", "packages/lib/vendor/"])
    rows = scan.sections()

    assert [(row.section, row.verdict) for row in rows] == [
        ("portable: pkill/killall", Verdict.WARN)
    ]
    assert rows[0].detail == ("7 uses: .github/workflows/ci.yml:1, run.sh:5, run.sh:6 and 4 more")
    assert scan.found() == {
        "pkill/killall": [
            ".github/workflows/ci.yml:1",
            "run.sh:5",
            "run.sh:6",
            "run.sh:7",
            "run.sh:8",
            "sub/Makefile:1",
            "packages/lib/tools/go.sh:1",
        ]
    }
