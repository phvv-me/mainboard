# What a node operation reads of the node a fix row names, gathered for every row of a rule in one
# query, and the base of the operations that read it.

import re
from abc import ABC, abstractmethod
from operator import attrgetter
from typing import TYPE_CHECKING, ClassVar

from patos import FrozenModel

from ...findings import Span
from ..base import Operation
from .dialects import Dialect

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from duckdb import DuckDBPyConnection

    from ...edits import Edit

# The semantic types of tokens that carry no meaning of their own.
_TRIVIA = frozenset({"PARSER_PUNCTUATION", "PARSER_DELIMITER"})
_BLANKS = re.compile(rb"[ \t]*")

# One `ast_nodes` row as the struct `Node` validates.
_NODE = (
    "{'node': n.node_id, 'parent': n.parent_id, 'index': n.sibling_index, 'type': n.type, "
    "'semantic': n.semantic_type, 'start': n.start_byte, 'end': n.end_byte, "
    "'line': n.start_line, 'column': n.start_column, 'end_line': n.end_line, "
    "'end_column': n.end_column}"
)
# A node's descendants follow it in id order, so its subtree and its ancestors are id ranges.
_SITES = f"""
    WITH wanted AS (SELECT unnest($sites, recursive := true))
    SELECT w.path, w.node, a.language, s.content,
           (SELECT list({_NODE} ORDER BY n.node_id) FROM ast_nodes n
            WHERE n.path = w.path
              AND n.node_id BETWEEN t.node_id AND t.node_id + t.descendant_count) AS tree,
           (SELECT list({_NODE} ORDER BY n.sibling_index) FROM ast_nodes n
            WHERE n.path = w.path AND n.parent_id = t.parent_id) AS siblings,
           (SELECT list({_NODE} ORDER BY n.sibling_index) FROM ast_nodes n, ast_nodes p
            WHERE p.path = w.path AND p.node_id = t.parent_id
              AND n.path = w.path AND n.parent_id = p.parent_id) AS uncles,
           (SELECT list({_NODE} ORDER BY n.node_id DESC) FROM ast_nodes n
            WHERE n.path = w.path AND n.node_id < t.node_id
              AND n.node_id + n.descendant_count >= t.node_id) AS ancestors
    FROM wanted w
    JOIN ast_nodes t ON t.path = w.path AND t.node_id = w.node
    JOIN ast_files a ON a.path = w.path
    JOIN source_files s ON s.path = w.path
"""


class Node(FrozenModel):
    """One `ast_nodes` row: ids within its file, kinds, byte range and span as the facts give them.

    node: the id within the file, in document order.
    parent: the parent's id, None for the file's root.
    index: the place among the parent's children.
    type: the grammar's name for the node.
    semantic: the semantic type as text (`METADATA_COMMENT`, `PARSER_DELIMITER`).
    start: the first byte.
    end: the byte after the last one.
    line: the first line, one-based.
    column: the first column in bytes, zero-based.
    end_line: the last line, one-based.
    end_column: the column after the last byte, zero-based.
    """

    node: int
    parent: int | None
    index: int
    type: str
    semantic: str
    start: int
    end: int
    line: int
    column: int
    end_line: int
    end_column: int

    @property
    def comment(self) -> bool:
        """Whether the node is a comment."""
        return self.semantic == "METADATA_COMMENT"

    @property
    def trivial(self) -> bool:
        """Whether the node is punctuation or a delimiter."""
        return self.semantic in _TRIVIA

    def span(self, path: str) -> Span:
        """Where the node stands in the file at `path`."""
        return Span(
            path=path,
            start_line=self.line,
            start_column=self.column,
            end_line=self.end_line,
            end_column=self.end_column,
        )


class Site(FrozenModel):
    """What a node operation reads of one node: its file's text and language, its subtree (the
    node first), its parent's children, its grandparent's children and its ancestors (nearest
    first). The root has no siblings, uncles or ancestors, and a child of the root no uncles."""

    SQL: ClassVar[str] = _SITES

    path: str
    node: int
    language: str
    content: str
    tree: tuple[Node, ...]
    siblings: tuple[Node, ...] = ()
    uncles: tuple[Node, ...] = ()
    ancestors: tuple[Node, ...] = ()

    @classmethod
    def gathered(
        cls, cursor: DuckDBPyConnection, wanted: Collection[tuple[str, int]]
    ) -> dict[tuple[str, int], Site]:
        """The site of each wanted `(path, node)` the facts hold, from one query.

        A node the facts do not hold has no site.
        """
        if not wanted:
            return {}
        sites = [{"path": path, "node": node} for path, node in sorted(set(wanted))]
        result = cursor.execute(cls.SQL, {"sites": sites})
        columns = [column for column, *_ in result.description]
        rows = (dict(zip(columns, row, strict=True)) for row in result.fetchall())
        found = [
            cls(**{name: value for name, value in row.items() if value is not None})
            for row in rows
        ]
        return {(site.path, site.node): site for site in found}

    @property
    def dialect(self) -> Dialect:
        """The syntax of the file's language."""
        return Dialect.find(self.language, attr="language")()

    @property
    def parent(self) -> Node | None:
        """The node's parent, None for the root."""
        return next(iter(self.ancestors), None)

    @property
    def target(self) -> Node:
        """The node the site is of."""
        return self.tree[0]


class Tree:
    """A subtree over a text's bytes: children in order, trivia set apart.

    It is built from a site, or from a parsed pattern and the pattern's own bytes.
    """

    @classmethod
    def of(cls, site: Site) -> Tree:
        """The subtree of `site` over its file's bytes."""
        return cls(site.tree, site.content.encode())

    def __init__(self, nodes: Sequence[Node], data: bytes) -> None:
        self.nodes, self.data = nodes, data
        self._kids: dict[int | None, list[Node]] = {}
        for node in sorted(nodes, key=attrgetter("index")):
            self._kids.setdefault(node.parent, []).append(node)

    def children(self, node: Node) -> list[Node]:
        """Every child of `node` in order, comments included."""
        return list(self._kids.get(node.node, ()))

    def indent(self, offset: int) -> int:
        """The leading blanks of the line holding byte `offset`."""
        line = self.data.rfind(b"\n", 0, offset) + 1
        return next(_BLANKS.finditer(self.data, line)).end() - line

    def significant(self, node: Node) -> list[Node]:
        """The children of `node` that are neither trivia nor comments."""
        return [child for child in self.children(node) if not (child.trivial or child.comment)]

    def text(self, node: Node) -> bytes:
        """The bytes of `node`."""
        return self.data[node.start : node.end]

    def within(self, start: int, *, end: int) -> list[Node]:
        """The nodes lying wholly inside bytes `[start, end)`, in document order.

        A zero-width node at either end counts: error recovery inserts such tokens (a CUDA launch
        reads as an identifier of no width).
        """
        return [node for node in self.nodes if start <= node.start and node.end <= end]


class NodeOperation(Operation, ABC):
    """An operation on the node a fix row names by `path` and `node`.

    The engine sets the row's `site` before `edits` runs; a node the facts no longer hold has
    none, and the operation refuses.
    """

    @abstractmethod
    def lowered(self, site: Site, row: Mapping) -> list[Edit] | None:
        """The edits that change the node of `site`, None when they cannot be made safely."""

    def edits(self, row: Mapping) -> list[Edit] | None:
        try:
            site = row["site"]
        except KeyError:
            self.refused("the node is gone")
            return None
        return self.lowered(site, row)
