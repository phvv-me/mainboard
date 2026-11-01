"""A node operation reads the site of its node, which the engine gathers for every row of a rule at
once: the node's subtree, siblings, uncles and ancestors beside its file's text."""

import time
from collections.abc import Callable, Mapping, Sequence
from operator import attrgetter
from pathlib import Path
from unittest.mock import Mock

import duckdb
import pytest
from duckdb import DuckDBPyConnection
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mainboard.analysis import Rule
from mainboard.analysis.edits import Delete, Edit, Replace
from mainboard.analysis.engine.apply import Plan
from mainboard.analysis.engine.lint import Lint
from mainboard.analysis.engine.sources import Source
from mainboard.analysis.operations import Operation
from mainboard.analysis.operations.nodes import Node, NodeOperation, Site

from ...engine.builders import written as write
from .builders import fact_nodes

_PATHS = [
    pytest.param("module.py", id="python"),
    pytest.param("unicode.py", id="python-with-crlf-and-multibyte-text"),
    pytest.param("module.ts", id="typescript"),
    pytest.param("lib.rs", id="rust"),
    pytest.param("unit.c", id="c"),
    pytest.param("unit.cpp", id="cpp"),
    pytest.param("kernel.cu", id="cuda"),
    pytest.param("apply.py", id="a-real-module-of-the-engine"),
]
_MODULE = {"module.py": "x = 1\n"}
_HELD = "SELECT finding_key, 'probe-node' AS operation, path, node FROM finding"
_DROP = "SELECT finding_key, 'probe-drop' AS operation, path, node FROM finding"
_GONE = "SELECT finding_key, 'probe-node' AS operation, path, 999999 AS node FROM finding"
# The delimiters the facts keep: brackets, quotes and the opening of a C++ raw string.
_DELIMITERS = {b"(", b")", b"[", b"]", b"{", b"}", b'"', b"'", b"`", b'R"'}


def _expected(nodes: Sequence[Node]) -> dict[int, dict[str, tuple[Node, ...]]]:
    """Each node's subtree, siblings, uncles and ancestors found from the parent links alone."""
    by_id = {node.node: node for node in nodes}
    kids = {
        parent: tuple(sorted((n for n in nodes if n.parent == parent), key=attrgetter("index")))
        for parent in {node.parent for node in nodes}
    }

    def subtree(node: Node) -> tuple[Node, ...]:
        return (node, *(below for kid in kids.get(node.node, ()) for below in subtree(kid)))

    def above(node: Node) -> tuple[Node, ...]:
        parent = by_id.get(node.parent) if node.parent is not None else None
        return () if parent is None else (parent, *above(parent))

    def beside(node: Node | None) -> tuple[Node, ...]:
        return kids.get(node.parent, ()) if node is not None and node.parent is not None else ()

    return {
        node.node: {
            "tree": subtree(node),
            "siblings": beside(node),
            "uncles": beside(next(iter(above(node)), None)),
            "ancestors": above(node),
        }
        for node in nodes
    }


@pytest.fixture
def scratch(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], type[Rule]]:
    """Define a rule fixed by an SQL text, and the node operation that text names, for one test.

    The rule flags every expression statement. One operation rewrites a node to its type, the
    number of its siblings and of its ancestors, the other deletes it. The registries are left as
    found.
    """
    monkeypatch.setattr(Rule, "defined", dict(Rule.defined))
    monkeypatch.setattr(Operation, "registry_entries", [*Operation.registry_entries])

    class ProbeNode(NodeOperation):
        def lowered(self, site: Site, row: Mapping) -> list[Edit] | None:
            text = f"{site.target.type} {len(site.siblings)} {len(site.ancestors)}"
            return [Replace(span=site.target.span(site.path), text=text)]

    class ProbeDrop(NodeOperation):
        def lowered(self, site: Site, row: Mapping) -> list[Edit] | None:
            return [Delete(span=site.target.span(site.path))]

    def define(sql: str) -> type[Rule]:
        class Statements(Rule, id="PY-PROB0001", applicability="review"):
            """Every expression statement."""

            query = (
                "SELECT path, start_line, start_column, 'statement' AS message, node_id AS node "
                "FROM ast_nodes WHERE type = 'expression_statement'"
            )
            fix = sql

        return Statements

    return define


@pytest.mark.parametrize("path", _PATHS)
def test_a_site_holds_the_subtree_siblings_uncles_and_ancestors_of_its_node(
    cursor: DuckDBPyConnection, sites: Mapping[str, Mapping[int, Site]], path: str
) -> None:
    gathered = sites[path]

    expected = {
        node: gathered[node].model_copy(update=kin)
        for node, kin in _expected(fact_nodes(cursor, path)).items()
    }

    assert dict(gathered) == expected


@pytest.mark.parametrize("path", _PATHS)
def test_a_sites_parent_is_the_node_its_nodes_parent_link_names(
    cursor: DuckDBPyConnection, sites: Mapping[str, Mapping[int, Site]], path: str
) -> None:
    nodes = fact_nodes(cursor, path)
    by_id: dict[int | None, Node] = {node.node: node for node in nodes}

    parents = {node: site.parent for node, site in sites[path].items()}

    assert parents == {node.node: by_id.get(node.parent) for node in nodes}


@pytest.mark.parametrize("path", _PATHS)
def test_a_sites_node_is_the_first_of_its_subtree_and_its_file_text_the_files_bytes(
    project: Path, sites: Mapping[str, Mapping[int, Site]], path: str
) -> None:
    gathered = sites[path].values()

    held = {(site.target.node == site.node, site.content.encode()) for site in gathered}

    assert held == {(True, (project / path).read_bytes())}


@pytest.mark.parametrize("path", _PATHS)
def test_a_nodes_span_covers_the_bytes_of_its_range_in_every_corpus_file(
    project: Path, cursor: DuckDBPyConnection, path: str
) -> None:
    source = Source((project / path).read_bytes())
    nodes = fact_nodes(cursor, path)

    covered = [source.covers(node.span(path)) for node in nodes]

    assert covered == [(node.start, node.end) for node in nodes]


@pytest.mark.parametrize(
    ("path", "comments"),
    [
        pytest.param("module.py", ["# a comment"], id="a-hash-comment"),
        pytest.param("unicode.py", ["# 日本\r"], id="a-comment-that-holds-its-carriage-return"),
        pytest.param("module.ts", ["// note"], id="a-slash-comment"),
        pytest.param("lib.rs", ["// c"], id="a-line-comment-with-a-token-inside"),
        pytest.param("unit.c", [], id="a-file-without-comments"),
    ],
)
def test_the_comment_nodes_of_a_file_are_exactly_its_comments(
    sites: Mapping[str, Mapping[int, Site]], path: str, comments: list[str]
) -> None:
    root = sites[path][0]

    found = [root.content.encode()[n.start : n.end].decode() for n in root.tree if n.comment]

    assert found == comments


@pytest.mark.parametrize("path", _PATHS)
def test_trivial_nodes_are_only_the_delimiters_the_facts_keep(
    sites: Mapping[str, Mapping[int, Site]], path: str
) -> None:
    root = sites[path][0]

    texts = {root.content.encode()[n.start : n.end] for n in root.tree if n.trivial}

    assert texts <= _DELIMITERS


@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@example(drawn=[("module.py", 1), ("module.py", 1), ("lib.rs", 10**6)])
@given(
    drawn=st.lists(
        st.tuples(st.sampled_from(["module.py", "lib.rs", "kernel.cu"]), st.integers(0, 60))
    )
)
def test_sites_of_many_nodes_come_back_from_one_query_keyed_by_path_and_node(
    cursor: DuckDBPyConnection, drawn: list[tuple[str, int]]
) -> None:
    spy = Mock(wraps=cursor)

    batch = Site.gathered(spy, drawn)

    alone = {key: Site.gathered(cursor, [key]).get(key) for key in set(drawn)}
    assert spy.execute.call_count == min(1, len(set(drawn)))
    assert {key: batch.get(key) for key in set(drawn)} == alone
    assert all(key == (site.path, site.node) for key, site in batch.items())


def test_forty_sites_of_a_real_module_are_gathered_at_once(cursor: DuckDBPyConnection) -> None:
    rows = cursor.execute(
        "SELECT path, node_id FROM ast_nodes WHERE path = 'apply.py' "
        "AND type = 'expression_statement' ORDER BY node_id LIMIT 40"
    ).fetchall()
    started = time.perf_counter()

    found = Site.gathered(cursor, rows)

    assert (len(rows), set(found), time.perf_counter() - started < 2.0) == (40, set(rows), True)


def test_a_fixing_lint_holds_the_files_and_sources_a_site_reads(
    tmp_path: Path, scratch: Callable[[str], type[Rule]]
) -> None:
    write(tmp_path, _MODULE)

    with Lint(tmp_path, [scratch(_HELD)]) as lint:
        found = Site.gathered(lint.facts(tmp_path).cursor(None), [("module.py", 0)])

    assert [site.target.type for site in found.values()] == ["module"]


def test_no_wanted_node_gathers_no_site_without_asking_the_facts() -> None:
    with duckdb.connect() as bare:
        assert Site.gathered(bare, set()) == {}


def test_a_checking_lint_leaves_the_sources_unread(
    tmp_path: Path, scratch: Callable[[str], type[Rule]]
) -> None:
    write(tmp_path, _MODULE)

    with Lint(tmp_path, [scratch(_HELD)], fixing=False) as lint:
        cursor = lint.facts(tmp_path).cursor(None)
        with pytest.raises(duckdb.CatalogException, match="source_files"):
            Site.gathered(cursor, [("module.py", 0)])


@pytest.mark.parametrize(
    ("fix", "written"),
    [
        pytest.param(
            _HELD, {"module.py": b"expression_statement 1 1\n"}, id="a-node-the-facts-hold"
        ),
        pytest.param(_DROP, {"module.py": b""}, id="another-operation-on-a-node-the-facts-hold"),
        pytest.param(_GONE, {}, id="a-node-the-facts-do-not-hold"),
        pytest.param(
            f"{_HELD} UNION ALL {_GONE}",
            {},
            id="a-missing-node-withholds-the-other-rows-of-its-finding",
        ),
    ],
)
def test_a_node_the_facts_no_longer_hold_gives_its_operation_a_refusal(
    tmp_path: Path, scratch: Callable[[str], type[Rule]], fix: str, written: dict[str, bytes]
) -> None:
    rule, plan = scratch(fix), Plan(tmp_path)
    write(tmp_path, {"module.py": "x = 1\n"})

    with Lint(tmp_path, [rule]) as lint:
        found = lint.repairs(lint.facts(tmp_path), ["review"])
    plan.add([edit for repair in found[rule.id] for edit in repair])

    assert plan.written() == written
