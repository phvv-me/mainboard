from pathlib import Path

from duckdb import DuckDBPyConnection

from mainboard.analysis.engine import apply
from mainboard.analysis.operations.nodes import Dialect, Node
from mainboard.analysis.providers.ducks import Ducks

_FACT_NODES_SQL = """
    SELECT node_id AS node, parent_id AS parent, sibling_index AS "index", type,
           semantic_type AS semantic, start_byte AS start, end_byte AS "end",
           start_line AS line, start_column AS "column", end_line, end_column
    FROM ast_nodes WHERE path = ? ORDER BY node_id
"""
_GRAMMAR_NODES_SQL = """
    SELECT node_id AS node, parent_id AS parent, sibling_index AS "index", type,
           semantic_type_to_string(semantic_type) AS semantic, start_byte AS start,
           end_byte AS "end", start_line AS line, start_column - 1 AS "column", end_line,
           end_column - 1 AS end_column
    FROM parse_ast(?, ?, source := 'full') ORDER BY node_id
"""

_PYTHON = """\
import os


@decorator
def f(x, *args):
    \"\"\"doc
    more\"\"\"
    # a comment
    y = x.a(1)[0] + -x * 2 - 1.5
    z = not x and y or x == y
    s = "a" "b"
    t = [1, 2], {1: 2}, {1, 2}, (1, 2), (x + 1)
    u = True, False, None
    return y


async def g(x):
    return await x
"""
_TYPESCRIPT = """\
export function g(x: number): number {
  // note
  const a = x.y(1)[0] + -x * 2;
  const b = await x;
  const c = x as number;
  const d = x!;
  const e = x satisfies number;
  let f = x++;
  const s = 'a', t = `a
b`, r = /x/g, n = null, u = undefined, o = { a: 1 }, l = [1, 2], p = (a + b);
  const th = this, tr = true, fa = false;
  return a;
}
declare function h(): void;
const w = "a\\
b";
"""
_RUST = """\
pub fn one(a: i32) -> i32 {
    // c
    let b = a.max(1) + -a * 2;
    let c = [1, 2][0];
    let d = (1, 2);
    let e = a as i64;
    let f = &b;
    let g = *f;
    let s = "a
b";
    let r = r"raw
more";
    let ch = 'c';
    let t = true;
    let n = 1.5;
    let m = vec![1];
    let p = self::one;
    let u = ();
    let q = foo()?;
    let w = bar().await;
    let x = (a + b);
    b
}
"""
_C = """\
int add(int a, int b) {
  int c = a + -b * 2;
  c = *(&a) + a.x + a->y + b[0] + f(c);
  ++c;
  c++;
  int d = (int)c + sizeof c;
  const char *s = "a" "b";
  char ch = 'c';
  int t = true, f = false, n = NULL;
  struct S v = (struct S){1};
  return (c);
}
extern "C" void h(void) {}
const char *m = "a\\
b";
"""
_CPP = (
    _C
    + """\
template <typename T>
T twice(T a) { return a + a; }
auto raw = R"(a
b)";
int *p = nullptr;
auto me = this;
auto q = std::move(raw);
"""
)
_CUDA = (
    _CPP
    + """\
__global__ void __launch_bounds__(128) bounded(float* p) { p[0] = 1; }

static __global__ void st(float* p) { p[0] = 2; }

template <int N>
__global__ void tmpl(float* p) { p[0] = N; }

__global__ void decl_only(float* p);

namespace ns {
__global__ void inner(float* p) { p[0] = 3; }
}

struct S {
  __device__ int method(int a) { return a; }
};

__device__ int S::out_of_line(int a) { return a; }

void launch(float* p) {
  bounded<<<1, 128>>>(p);
}
"""
)
# A module whose lines end in CRLF and whose strings hold multibyte characters, which tree-sitter
# measures in bytes.
_UNICODE = 'x = "é"; y = 1\r\nz = "日本"  # 日本\r\n\r\n\r\ndef w():\r\n    return z + "é"\r\n'


def sources() -> dict[str, str]:
    """The files of the corpus the node tests parse, by project path.

    The last is a real module of the engine, long enough that a batch of forty sites is no toy.
    """
    return {
        "module.py": _PYTHON,
        "unicode.py": _UNICODE,
        "module.ts": _TYPESCRIPT,
        "lib.rs": _RUST,
        "unit.c": _C,
        "unit.cpp": _CPP,
        "kernel.cu": _CUDA,
        "apply.py": Path(apply.__file__).read_text(encoding="utf-8"),
    }


def _nodes(rows: DuckDBPyConnection) -> list[Node]:
    """The rows `rows` answers, which name the columns of a `Node`, as nodes."""
    columns = [column for column, *_ in rows.description]
    return [Node(**dict(zip(columns, row, strict=True))) for row in rows.fetchall()]


def fact_nodes(cursor: DuckDBPyConnection, path: str) -> list[Node]:
    """The nodes of the file at `path` as the facts hold them, one plain row each."""
    return _nodes(cursor.execute(_FACT_NODES_SQL, [path]))


def grammar_nodes(db: DuckDBPyConnection, dialect: type[Dialect], text: str) -> list[Node]:
    """The nodes of `text` as the grammar of `dialect` parses it, pruning nothing.

    The facts parse with the same grammars (`Ducks.LANGUAGES` maps a language to its grammar),
    but drop punctuation; a parsed pattern keeps it.
    """
    return _nodes(db.execute(_GRAMMAR_NODES_SQL, [text, Ducks.LANGUAGES[dialect.language][0]]))
