"""A tree reads a subtree over its bytes: children in order, trivia set apart, indents, ranges."""

from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mainboard.analysis.operations.nodes import Dialect, Node, Site, Tree
from mainboard.analysis.operations.nodes.dialects import Python, TypeScript

_DRAWN = settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
_PATHS = ["module.py", "unicode.py", "module.ts", "lib.rs", "unit.cpp", "kernel.cu", "apply.py"]
_PATTERN = "try:\n    µµµBODY\nexcept µE:\n    µINERT\n"


def _drawn(sites: Mapping[str, Mapping[int, Site]], data: st.DataObject) -> Site:
    """A site of a corpus file, drawn."""
    path = data.draw(st.sampled_from(_PATHS))
    return sites[path][data.draw(st.sampled_from(sorted(sites[path])))]


def _indent(data: bytes, offset: int) -> int:
    """The blanks that open the line holding `offset`, found by walking the lines."""
    start = 0
    for line in data.split(b"\n"):
        if offset <= start + len(line):
            return len(line) - len(line.lstrip(b" \t"))
        start += len(line) + 1
    raise ValueError(f"offset {offset} is past the text")


@_DRAWN
@given(data=st.data())
def test_the_children_of_a_node_lie_inside_it_in_order_and_apart(
    sites: Mapping[str, Mapping[int, Site]], data: st.DataObject
) -> None:
    site = _drawn(sites, data)

    spans = [(kid.start, kid.end) for kid in Tree.of(site).children(site.target)]

    assert spans == sorted(spans)
    assert all(a[1] <= b[0] for a, b in pairwise(spans))
    assert all(site.target.start <= start and end <= site.target.end for start, end in spans)


@_DRAWN
@given(data=st.data())
def test_the_text_of_a_node_is_the_files_bytes_over_its_range(
    project: Path, sites: Mapping[str, Mapping[int, Site]], data: st.DataObject
) -> None:
    site = _drawn(sites, data)

    text = Tree.of(site).text(site.target)

    assert text == (project / site.path).read_bytes()[site.target.start : site.target.end]


@_DRAWN
@given(data=st.data())
def test_the_range_of_a_node_holds_its_subtree_and_only_equal_ancestors_or_zero_width_nodes(
    sites: Mapping[str, Mapping[int, Site]], data: st.DataObject
) -> None:
    site = _drawn(sites, data)
    whole = Tree.of(sites[site.path][0])

    inside = whole.within(site.target.start, end=site.target.end)

    extra = set(inside) - set(site.tree) - set(site.ancestors)
    assert set(site.tree) <= set(inside)
    assert all(node.start == node.end for node in extra)


@_DRAWN
@given(data=st.data(), place=st.floats(0, 1))
def test_the_indent_of_an_offset_is_the_blanks_that_open_its_line(
    sites: Mapping[str, Mapping[int, Site]], data: st.DataObject, place: float
) -> None:
    root = sites[data.draw(st.sampled_from(_PATHS))][0]
    text = root.content.encode()

    offset = int(place * len(text))

    assert Tree.of(root).indent(offset) == _indent(text, offset)


@pytest.mark.parametrize(
    ("path", "block", "comment"),
    [
        pytest.param("module.py", "block", b"# a comment", id="python"),
        pytest.param("module.ts", "statement_block", b"// note", id="typescript"),
        pytest.param("lib.rs", "block", b"// c", id="rust"),
    ],
)
def test_comments_and_delimiters_are_children_of_a_block_but_not_significant_ones(
    sites: Mapping[str, Mapping[int, Site]], path: str, block: str, comment: bytes
) -> None:
    body = next(site for site in sites[path].values() if site.target.type == block)
    tree = Tree.of(body)

    kept = [tree.text(kid) for kid in tree.significant(body.target)]
    every = [tree.text(kid) for kid in tree.children(body.target)]

    assert (comment in every, comment in kept) == (True, False)
    assert set(kept) < set(every)


@pytest.mark.parametrize(
    ("dialect", "pattern", "statement", "parts"),
    [
        pytest.param(
            Python,
            _PATTERN,
            "try_statement",
            ["try", "µµµBODY", "except µE:\n    µINERT"],
            id="a-python-pattern-with-an-except-clause",
        ),
        pytest.param(
            TypeScript,
            "if (µC) { µµµBODY }\n",
            "if_statement",
            ["if", "(µC)", "{ µµµBODY }"],
            id="a-typescript-pattern-with-a-block",
        ),
    ],
)
def test_a_tree_of_parsed_pattern_text_reads_its_own_bytes(
    grammar: Callable[[type[Dialect], str], list[Node]],
    dialect: type[Dialect],
    pattern: str,
    statement: str,
    parts: list[str],
) -> None:
    nodes = grammar(dialect, pattern)
    tree = Tree(nodes, pattern.encode())
    top = next(node for node in nodes if node.type == statement)

    significant = [tree.text(node).decode() for node in tree.significant(top)]

    assert significant == parts
