"""Every group and command of the CLI is declared in `TREE`, answers `--help`, documents every
parameter, and every command the docs name exists: help that says nothing and docs naming a removed
verb both slipped through before (no parameter description reached `--help` until 2026-09-29).
The declaration answers `mb --help` and a mistyped word without importing a verb module, and
`--help` and `--version` import no heavy library: on Python 3.15, where `__lazy_modules__` defers
the facades, that held `--help` at 0.17 s (0.57 s before, 2026-10-29). PY-IMPO0010 reads the same
`heavy` list statically."""

import inspect
import io
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from functools import reduce
from pathlib import Path

import pytest
from cyclopts import App
from hypothesis import given
from hypothesis import strategies as st
from rich.console import Console

from mainboard.analysis.rules.python.imports.startup import HeavyStartupImport
from mainboard.verbs.constants import TREE
from mainboard.verbs.node import Command, Typed

_PACKAGE = Path(__file__).parents[2]
# The package's README, and the workspace's agent skills when the package sits in its monorepo.
_DOCS = [
    doc
    for doc in (
        _PACKAGE / "README.md",
        _PACKAGE.parents[1] / "AGENTS.md",
        *sorted(_PACKAGE.parents[1].glob(".agents/skills/*/SKILL.md")),
    )
    if doc.is_file()
]
# A call as a doc spells it in code: a shell line in a fenced block (`$ ` optional) or an inline
# code span, `mb` or `mainboard` then the words of a command path.
_WORDS = r"(?:mb|mainboard) ((?:[a-z][a-z-]*)(?: [a-z][a-z-]*)*)"
_SHELL_LINE = re.compile(rf"^\s*(?:\$ )?{_WORDS}")
_SPAN = re.compile(rf"`{_WORDS}")

# A fresh process that prints, at its exit, every module it imported: `mb ARGV` when its first
# argument is `mb`, or the library facade alone when given no argument.
_IMPORTED = """
import atexit, json, sys
atexit.register(lambda: print(json.dumps(sorted(sys.modules)), file=sys.__stderr__))
if sys.argv[1:]:
    sys.argv = sys.argv[1:]
    from mainboard.cli import main
    main()
else:
    import mainboard
"""
# A verb home module, the kind `mb --help` must not import: `verbs/<group>/<module>.py`.
_HOME = re.compile(r"mainboard\.verbs\.\w+\.\w+")
_HEAVY = set(HeavyStartupImport().heavy)
# The libraries of the memory extra, which no listing or help may import.
_CLIENTS = {"aizk_client", "fastmcp", "mcp"}

# Every path the declaration holds, hidden ones included, and what each is.
_DECLARED = dict(TREE.entries())

# Where each leaf of 0.5.7 that moved answers now, an alias's old spelling running it as typed.
# `refactor compare` retired.
_MOVED = {
    **{verb: f"dep {verb}" for verb in ("add", "install", "list", "lock", "remove", "shell")},
    **{verb: f"dep {verb}" for verb in ("tree", "update", "upgrade")},
    **{verb: f"code {verb}" for verb in ("ci", "doctor", "lint", "new")},
    "completion": "--completion",
    "help": "--help",
    "pack": "host pack",
    "plot": "data paper plot",
    "query": "data query",
    "job collect": "data lake collect",
    **{f"lake {verb}": f"data lake {verb}" for verb in ("check", "compact", "dedup", "evict")},
    **{f"lake {verb}": f"data lake {verb}" for verb in ("ingest", "materialize", "replicate")},
    **{f"lake {verb}": f"data lake {verb}" for verb in ("serve", "upgrade")},
    **{f"paper {verb}": f"data paper {verb}" for verb in ("build", "plot")},
    "self update": "host update",
    "self version": "--version",
    "probe device": "data probe",
    **{f"agents {verb}": f"code agents {verb}" for verb in ("check", "sync", "update")},
    "binary sync": "dep binary",
    **{f"apple {verb}": "code app inspect" for verb in ("bundle", "diagnostics", "results")},
    "apple font-previews": "code app fonts",
    **{f"apple {verb}": f"code app {verb}" for verb in ("execute", "launch", "simulators")},
    "apple toolchain": "code app toolchain",
    **{f"app {verb}": f"code app {verb}" for verb in ("build", "list", "test")},
    "refactor catalog": "code rules explain",
    **{f"refactor {verb}": f"code rules {verb}" for verb in ("check", "explain", "fix", "gate")},
    **{f"refactor {verb}": f"code rules {verb}" for verb in ("preview", "sql", "test", "trial")},
    **{f"refactor {verb}": f"code refactor {verb}" for verb in ("diagram", "impact", "matrix")},
    **{f"refactor {verb}": f"code refactor {verb}" for verb in ("move", "proof", "uses")},
    **{
        f"refactor swift-{verb}": f"code refactor swift-{verb}"
        for verb in ("compare", "inventory")
    },
    **{f"refactor {verb}": f"data paper {verb}" for verb in ("evidence", "prose")},
}


def paths(node: App | None = None, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Every command path the CLI registers, groups and hidden verbs included, from the root."""
    node = node or TREE.cli()
    found: list[tuple[str, ...]] = []
    for name in sorted(name for name in node if not name.startswith("-")):
        found.append((*prefix, name))
        found += paths(node[name], (*prefix, name))
    return found


def _at(app: App, path: tuple[str, ...]) -> App:
    """The command `path` names in `app`, imported if registered lazily."""
    return reduce(App.__getitem__, path, app)


def _heavy(modules: Iterable[str]) -> set[str]:
    return {name for name in modules if name.partition(".")[0] in _HEAVY}


def _imported(*argv: str) -> set[str]:
    """The modules `ARGV` imported, `mb` its first word; the library facade's alone without."""
    done = subprocess.run(
        [sys.executable, "-c", _IMPORTED, *argv],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return set(json.loads(done.stderr.splitlines()[-1]))


def _silent(path: tuple[str, ...]) -> list[str]:
    """The parameters `path`'s help lists without a word on what they are."""
    rendered = io.StringIO()
    TREE.cli().help_print(
        list(path), console=Console(file=rendered, width=4000, color_system=None)
    )
    # A parameter row is `  --name: <description> [default: ...]`; a bare one says nothing.
    return re.findall(r"^  (\S+): (?:\[default|$)", rendered.getvalue(), re.MULTILINE)


@pytest.mark.parametrize("path", paths(), ids=" ".join)
def test_every_command_answers_the_help_flag(mb, path: tuple[str, ...]) -> None:
    ran = mb(*path, "--help")
    assert ran.code == 0, ran.said
    assert "Usage:" in ran.out


@pytest.mark.parametrize("path", paths(), ids=" ".join)
def test_every_parameter_says_what_it_is(path: tuple[str, ...]) -> None:
    silent = _silent(path)
    assert not silent, f"{' '.join(path)}: no description for {silent}"


def test_every_registered_command_is_declared_in_the_tree_as_registered() -> None:
    """Each registered path is declared in `TREE`, shown and trailing as declared.

    So no `App.command` registers a verb outside the declaration.
    """
    app = TREE.cli()
    registered = {path: _at(app, path) for path in paths(app)}
    declared = {
        path: (entry.show, isinstance(entry, Command) and entry.trailing)
        for path, entry in _DECLARED.items()
    }
    assert {path: (at.show, not at.version_flags) for path, at in registered.items()} == declared


def test_every_lazily_registered_command_lists_its_docstrings_first_line() -> None:
    """A root command is listed from `TREE` without importing it, so its line must be its own."""
    app = TREE.cli()
    listed = {
        path: entry.help
        for path, entry in _DECLARED.items()
        if isinstance(entry, Command) and entry.module
    }
    first = {path: inspect.getdoc(_at(app, path).default_command) for path in listed}
    assert {path: doc.partition("\n")[0] for path, doc in first.items() if doc} == listed


@pytest.mark.parametrize(
    ("argv", "homes"),
    [
        pytest.param([], set(), id="the-bare-listing"),
        pytest.param(["--help"], set(), id="the-root-help"),
        pytest.param(["--tree", "--json"], set(), id="the-declaration"),
        pytest.param(["nosuchcmd"], set(), id="a-word-naming-nothing"),
        pytest.param(["lint"], set(), id="a-moved-word"),
        pytest.param(
            ["job", "--help"],
            {"mainboard.verbs.work.jobs", "mainboard.verbs.work.submit"},
            id="one-group-listing",
        ),
    ],
)
def test_a_command_line_imports_only_the_verb_modules_it_types(
    argv: list[str], homes: set[str]
) -> None:
    loaded = _imported("mb", *argv)
    assert {name for name in loaded if _HOME.fullmatch(name)} == homes
    assert not [name for name in loaded if name.split(".")[1:2] in (["serve"], ["memory"])]
    assert not _CLIENTS & {name.partition(".")[0] for name in loaded}


def test_the_memory_group_listing_imports_its_verbs_and_no_client_library() -> None:
    loaded = _imported("mb", "memory", "--help")
    assert {name for name in loaded if _HOME.fullmatch(name)} == {
        "mainboard.verbs.evidence.memory"
    }
    assert not _CLIENTS & {name.partition(".")[0] for name in loaded}


@pytest.mark.parametrize(
    "argv",
    [pytest.param("--help", id="the-root-listing"), pytest.param("--version", id="the-version")],
)
def test_a_listing_imports_no_verb_module_and_no_heavy_library_past_the_facade(
    argv: str,
) -> None:
    """3.14 runs the facade eagerly, so there the listing may import what the facade imports."""
    loaded = _imported("mb", argv)
    assert not {name for name in loaded if _HOME.fullmatch(name)}
    assert _heavy(loaded) <= _heavy(_imported())


@pytest.mark.skipif(
    not hasattr(sys, "set_lazy_imports_filter"),
    reason="M00-PY315: the tool's Python 3.14 runs every facade eagerly",
)
def test_the_library_facade_imports_no_heavy_library_where_imports_are_lazy() -> None:
    assert not _heavy(_imported())


@pytest.mark.parametrize(
    ("typed", "said"),
    [
        *(
            pytest.param(old, f"`mb {new}`", id=f"moved-{old.replace(' ', '-')}")
            for old, new in sorted(_MOVED.items())
        ),
        pytest.param("ingets", "`mb data lake ingest`", id="a-leaf-typed-at-the-root"),
        pytest.param("job sumbit", "`mb job submit`", id="a-misspelled-verb"),
        pytest.param("nosuchcmd", "`mb --help` lists them", id="a-word-near-nothing"),
        pytest.param("host --bogus", "`mb host --help` lists them", id="an-unknown-group-option"),
    ],
)
def test_a_line_of_0_5_7_runs_as_typed_or_is_refused_naming_its_new_path(
    typed: str, said: str
) -> None:
    """An alias runs its path; any other moved spelling, or a typo, gets the path to type."""
    argv = TREE.rewritten(typed.split())
    assert f"`mb {' '.join(argv)}`" == said or said in Typed(node=TREE).refusal(argv)


@given(path=st.sampled_from(sorted(_DECLARED)), rest=st.lists(st.text()))
def test_a_declared_path_is_neither_refused_nor_rewritten_whatever_follows(
    path: tuple[str, ...], rest: list[str]
) -> None:
    typed = [*path, *(rest if isinstance(_DECLARED[path], Command) else ["--help", *rest])]
    assert Typed(node=TREE).refusal(typed) == ""
    assert TREE.rewritten(typed) == typed


@given(alias=st.sampled_from(TREE.spellings), rest=st.lists(st.text()))
def test_an_alias_is_spelled_as_its_path_whatever_follows(
    alias: tuple[tuple[str, ...], tuple[str, ...]], rest: list[str]
) -> None:
    words, path = alias
    assert TREE.rewritten([*words, *rest]) == [*path, *rest]
    assert Typed(node=TREE).refusal(list(path)) == ""


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(("execute",), id="the-job-script-handover"),
        pytest.param(("provide",), id="the-environment-a-dispatch-builds"),
        pytest.param(("attest",), id="the-receipt-a-job-records"),
        pytest.param(("sample",), id="the-readings-a-job-publishes"),
        pytest.param(("host", "serve"), id="the-pbs-keeper-line"),
        pytest.param(("job", "list"), id="the-settling-pass"),
    ],
)
def test_the_paths_generated_scripts_spell_stay_declared(path: tuple[str, ...]) -> None:
    """Another host's release runs these spellings, so a regroup keeps them where they are."""
    assert path in _DECLARED


def _calls(doc: Path) -> list[list[str]]:
    """The words of each call `doc` spells in an inline code span or a fenced shell line."""
    calls: list[list[str]] = []
    fenced = False
    for line in doc.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        calls += [span.split() for span in _SPAN.findall(line)]
        if fenced and (shell := _SHELL_LINE.match(line)):
            calls.append(shell.group(1).split())
    return calls


def test_every_command_the_docs_name_exists() -> None:
    known = {" ".join(path) for path in paths()} | {option.name for option in TREE.options}
    # Some known command must start the words, as an alias spells it; a tool's own argument
    # (`mb run pytest`) follows a known command and is not a command itself.
    missing = {
        f"{doc.name}: mb {' '.join(words)}"
        for doc in _DOCS
        for words in map(TREE.rewritten, _calls(doc))
        if not any(" ".join(words[:n]) in known for n in range(1, len(words) + 1))
    }
    assert not missing, sorted(missing)
