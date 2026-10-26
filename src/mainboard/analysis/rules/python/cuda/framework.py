"""PY-CUDA rules that stay specific to one framework because the hazard is: the numba-cuda kernel
that returns a value, the grid index nothing bounds, the debug build, Triton's dot precision and
Helion's hard-coded config."""

from ....catalog import Example, Rule


class KernelReturnsValue(Rule, id="PY-CUDA0014", applicability="review"):
    """An explicit value returned from a numba-cuda or patos kernel: a kernel's return value never
    reaches the host, and numba refuses to compile it."""

    query = """
        SELECT r.path, r.start_line, r.start_column, r.end_line, r.end_column, d.qualname
        FROM cuda_nodes() r
        JOIN cuda_nodes() v ON v.fact_id = r.fact_id AND v.parent_id = r.node_id AND v.sibling_index = 1
        JOIN cuda_device_code() d ON d.fact_id = r.fact_id AND d.function_id = r.scope."function"
          AND d.role = 'kernel' AND d.framework IN ('numba', 'patos')
        WHERE r.type = 'return_statement'
    """
    message = (
        "Kernel `{qualname}` returns a value: a kernel has no caller to take it and numba refuses "
        "to compile it. Fix: keep the early exit and drop the value (`return`), writing results "
        "to an output array"
    )
    change = "dropped the value of the `return` in kernel `{qualname}`"
    fix = "SELECT *, 'return' AS text FROM finding"
    examples = (
        Example(
            bad=(
                """from numba import cuda


@cuda.jit
def kernel(x):
    if x[0] == 0:
        return 1
    x[0] = 1
"""
            ),
            good=(
                """from numba import cuda


@cuda.jit
def kernel(x):
    if x[0] == 0:
        return
    x[0] = 1
"""
            ),
        ),
    )


class GridIndexUnguarded(Rule, id="PY-CUDA0015", applicability="review"):
    """A numba-cuda grid index no branch or loop tests: threads past the end write out of
    bounds."""

    query = """
        WITH grids AS (
          SELECT c.fact_id, c.path, c.start_line, c.start_column, c.end_line, c.end_column,
                 c.scope_id, c.assigned_target AS index_name
          FROM cuda_calls() c
          WHERE c.qualified_name = 'numba.cuda.grid'
            AND regexp_matches(c.assigned_target, '^[A-Za-z_]\\w*$')
        )
        SELECT g.path, g.start_line, g.start_column, g.end_line, g.end_column, g.index_name,
               d.qualname
        FROM grids g
        JOIN cuda_device_code() d ON d.fact_id = g.fact_id AND d.function_id = g.scope_id
          AND d.role = 'kernel'
        WHERE NOT EXISTS (
          SELECT 1 FROM cuda_identifier_reads() u
          JOIN cuda_nodes() b ON b.fact_id = u.fact_id
            AND b.type IN ('if_statement', 'while_statement', 'for_statement',
                           'conditional_expression')
            AND u.node_id BETWEEN b.node_id + 1 AND b.node_id + b.descendant_count
          WHERE u.fact_id = g.fact_id AND u.scope_id = g.scope_id AND u.name = g.index_name)
    """
    message = (
        "Grid index `{index_name}` is never tested in `{qualname}`: the launch rounds the grid "
        "up, so the threads past the end read and write out of bounds. Fix: guard the body with "
        "`if {index_name} < n:` or use a grid-stride loop"
    )
    examples = (
        Example(
            bad=(
                """from numba import cuda


@cuda.jit
def kernel(x):
    i = cuda.grid(1)
    x[i] = 1
"""
            ),
            good=(
                """from numba import cuda


@cuda.jit
def kernel(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = 1
"""
            ),
        ),
    )


class DebugBuildCommitted(Rule, id="PY-CUDA0016", applicability="review"):
    """A numba-cuda function compiled with `debug=True` or `opt=False`: slow, and `assert` and
    `raise` behave differently than in the optimized build."""

    query = """
        SELECT d.path, d.start_line, d.start_column, d.end_line, d.end_column, d.name,
               kn.name || '=' || CASE kv.type WHEN 'true' THEN 'True' ELSE 'False' END AS flag
        FROM cuda_device_code() d
        JOIN cuda_nodes() f ON f.fact_id = d.fact_id AND f.node_id = d.function_id
        JOIN cuda_nodes() w ON w.fact_id = f.fact_id AND w.node_id = f.parent_id
          AND w.type = 'decorated_definition'
        JOIN cuda_nodes() dc ON dc.fact_id = w.fact_id AND dc.parent_id = w.node_id
          AND dc.type = 'decorator'
        JOIN cuda_nodes() kw ON kw.fact_id = dc.fact_id AND kw.type = 'keyword_argument'
          AND kw.node_id BETWEEN dc.node_id + 1 AND dc.node_id + dc.descendant_count
        JOIN cuda_nodes() kn ON kn.fact_id = kw.fact_id AND kn.parent_id = kw.node_id
          AND kn.sibling_index = 0
        JOIN cuda_nodes() kv ON kv.fact_id = kw.fact_id AND kv.parent_id = kw.node_id
          AND kv.sibling_index = 2
        WHERE d.framework = 'numba' AND d.role IN ('kernel', 'device')
          AND ((kn.name = 'debug' AND kv.type = 'true') OR (kn.name = 'opt' AND kv.type = 'false'))
    """
    message = (
        "CUDA function `{name}` is compiled with `{flag}`: the unoptimized build is much slower "
        "and its `assert` and `raise` differ from the shipped one. Fix: drop the flag before "
        "committing"
    )
    examples = (
        Example(
            bad="from numba import cuda\n\n\n@cuda.jit(debug=True)\ndef kernel(x):\n    x[0] = 1\n",
            good="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n",
        ),
    )


class UnpinnedDotPrecision(Rule, id="PY-CUDA0017", applicability="review"):
    """A Triton `tl.dot` that leaves its precision to the default (TF32 for fp32 inputs on NVIDIA)
    or states it with the deprecated `allow_tf32`. `input_precision` is the fourth parameter, so a
    call with four positional arguments, a keyword or a splat has pinned it."""

    query = r"""
        SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown,
               c.node_text, regexp_extract(c.node_text, 'allow_tf32\s*=\s*(\w+)', 1) AS allow_tf32
        FROM cuda_calls() c
        WHERE c.qualified_name = 'triton.language.dot' AND NOT EXISTS (
          SELECT 1 FROM cuda_arguments() k WHERE k.fact_id = c.fact_id AND k.call_id = c.node_id
            AND (k.keyword = 'input_precision' OR k.position >= 3))
          AND NOT EXISTS (
          SELECT 1 FROM cuda_nodes() s WHERE s.fact_id = c.fact_id AND s.parent_id = c.arguments_id
            AND s.type IN ('list_splat', 'dictionary_splat'))
    """
    message = (
        "`{shown}` does not pin `input_precision`: the default silently rounds fp32 inputs to "
        "TF32 on NVIDIA, so results differ by device and Triton version. Fix: name the "
        "precision"
    )
    change = "pinned `input_precision` on `{shown}`"
    # Behavior kept: `allow_tf32=False` is "ieee", `True` and the default are "tf32"; a
    # flag set from a variable is left for a person.
    fix = r"""
        SELECT *, CASE allow_tf32
            WHEN 'False' THEN regexp_replace(node_text, 'allow_tf32\s*=\s*False',
                                             'input_precision="ieee"')
            WHEN 'True' THEN regexp_replace(node_text, 'allow_tf32\s*=\s*True',
                                            'input_precision="tf32"')
            WHEN '' THEN regexp_replace(node_text, ',?\s*\)$', ', input_precision="tf32")')
          END AS text
        FROM finding
    """
    examples = (
        Example(
            bad=("import triton.language as tl\n\n\ndef body(a, b):\n    return tl.dot(a, b)\n"),
            good=(
                """import triton.language as tl


def body(a, b):
    return tl.dot(a, b, input_precision="tf32")
"""
            ),
        ),
    )


class HardcodedHelionConfig(Rule, id="PY-CUDA0018", applicability="review"):
    """A `@helion.kernel` whose config is one `helion.Config` written out: autotuned tile
    configs do not carry across GPU generations."""

    query = """
        SELECT d.path, d.start_line, d.start_column, d.end_line, d.end_column, d.name
        FROM cuda_device_code() d
        JOIN cuda_nodes() f ON f.fact_id = d.fact_id AND f.node_id = d.function_id
        JOIN cuda_nodes() w ON w.fact_id = f.fact_id AND w.node_id = f.parent_id
          AND w.type = 'decorated_definition'
        JOIN cuda_nodes() dc ON dc.fact_id = w.fact_id AND dc.parent_id = w.node_id
          AND dc.type = 'decorator'
        JOIN cuda_nodes() kw ON kw.fact_id = dc.fact_id AND kw.type = 'keyword_argument'
          AND kw.node_id BETWEEN dc.node_id + 1 AND dc.node_id + dc.descendant_count
        JOIN cuda_nodes() kn ON kn.fact_id = kw.fact_id AND kn.parent_id = kw.node_id
          AND kn.sibling_index = 0 AND kn.name = 'config'
        JOIN cuda_nodes() kv ON kv.fact_id = kw.fact_id AND kv.parent_id = kw.node_id
          AND kv.sibling_index = 2
        WHERE d.framework = 'helion' AND (
          (kv.type = 'call' AND EXISTS (SELECT 1 FROM cuda_calls() c
                                        WHERE c.fact_id = kv.fact_id AND c.node_id = kv.node_id
                                          AND c.qualified_name = 'helion.Config'))
          OR (kv.type = 'identifier' AND EXISTS (SELECT 1 FROM cuda_calls() c
                                                 WHERE c.fact_id = kv.fact_id
                                                   AND c.qualified_name = 'helion.Config'
                                                   AND c.assigned_target = kv.name)))
    """
    message = (
        "Helion kernel `{name}` hardcodes its `helion.Config`: tile configs tuned on one GPU "
        "generation run slower, or not at all, on another. Fix: draw the config from a per-arch "
        "registry"
    )
    examples = (
        Example(
            bad=(
                """import helion


@helion.kernel(config=helion.Config(block_sizes=[16]))
def scale(x):
    return x
"""
            ),
            good=("import helion\n\n\n@helion.kernel\ndef scale(x):\n    return x\n"),
        ),
    )
