"""What the PY-CUDA facts know of a value's type: whether it is device memory (`device`), a torch
tensor of unknown device (`tensor`) or an instance of a class of the project (`obj:<class key>`).

The types start from seeds and flow along edges, to a fixpoint.

Seeds: an annotation (a parameter, a return, a class field, a local), `self`, a constructor call
(`cupy.zeros`, `torch.zeros(..., device="cuda")`, `numba.cuda.to_device`, `Batch(...)`).

Edges: a name carries the type of its variable; an assignment gives a variable the type of its
value when every assignment to it agrees; an operator, a subscript and a conditional carry their
operands' array type; a call carries its callee's return type; a method or attribute of an
instance carries the member's type, found through the instance's class and the classes it
inherits from; a field takes the type of what a constructor or `self.field = value` stores in it.

Keys name the things typed: `e:<fact>:<node>` an expression, `v:<fact>:<scope>:<name>` a
variable, `f:<class key>.<name>` a field, `r:<fact>:<function node>` what a function returns.
"""

from ....catalog import Function
from .base import CudaRelation

_ARRAY_NODES = (
    "'identifier', 'attribute', 'subscript', 'call', 'binary_operator', 'unary_operator',"
    " 'parenthesized_expression', 'conditional_expression', 'comparison_operator',"
    " 'boolean_operator', 'named_expression'"
)
_CUPY_HOST_OR_OBJECT = (
    r"(asnumpy|get_array_module|dtype|result_type|can_cast|promote_types|issubdtype|isscalar|"
    r"ndim|shape|size|finfo|iinfo|fuse|vectorize|show_config|is_available|seterr|errstate|"
    r"set_printoptions|get_printoptions|get_.*|set_.*|cuda\..*|testing\..*|profiler\..*|jit\..*|"
    r".*pinned.*|time\..*|optimizing\..*|shares_memory|may_share_memory|iscomplexobj|isrealobj|"
    r"isfortran|min_scalar_type|common_type|obj2sctype|typename|mintypecode|sctype2char|"
    r"random\.(RandomState|Generator|default_rng|seed|get_random_state|BitGenerator)|"
    r"ElementwiseKernel|RawKernel|RawModule|ReductionKernel|u?int(8|16|32|64|p)?|"
    r"float(16|32|64|128)?|complex(64|128)?|bool_?|number|generic|inexact|integer|floating|"
    r"newaxis|pi|e|inf|nan|ndarray_base|Stream|Event|array_equal|allclose|array_equiv|"
    r"isclose_scalar)"
)
_TORCH_CREATION = (
    "zeros|ones|empty|full|rand|randn|randint|arange|linspace|logspace|eye|tensor|as_tensor|"
    "asarray|zeros_like|ones_like|empty_like|full_like|rand_like|randn_like|randperm|stack|cat|"
    "concat|from_numpy|from_dlpack|tril|triu|where|empty_strided|scalar_tensor|complex|polar|"
    "diag|diagflat|meshgrid|cumsum|clone"
)


class CudaCreatedArray(Function):
    """What calling the dotted name `qualified` creates: `device` for CuPy, numba-cuda and torch
    constructors given a CUDA device, `tensor` for torch ones left on the default, null for any
    other call. `call_text` is the call as written (it holds `device="cuda"` or not)."""

    qualified: str
    call_text: str

    expression = f"""
        CASE WHEN (regexp_matches(qualified, '^cupyx?\\.')
                   AND NOT regexp_matches(qualified, '^cupyx?\\.{_CUPY_HOST_OR_OBJECT}$'))
               OR regexp_matches(qualified, '^numba\\.cuda\\.(device_array|device_array_like|to_device|as_cuda_array|managed_array)$')
               OR regexp_matches(qualified, '^torch\\.cuda\\.[A-Za-z]+Tensor$')
               OR (regexp_matches(qualified, '^torch\\.({_TORCH_CREATION})$')
                   AND regexp_matches(call_text, 'device\\s*=\\s*[^,)]*cuda'))
             THEN 'device'
             WHEN regexp_matches(qualified, '^torch\\.({_TORCH_CREATION})$') THEN 'tensor' END
    """


class CudaTypeSeeds(CudaRelation):
    """Where a type is known without flowing from anywhere: annotations, `self`, constructors."""

    stored = True
    query = r"""
        WITH written AS (
          SELECT a.*, false AS element FROM cuda_annotations() a
          UNION ALL
          SELECT a.* REPLACE (cuda_annotation_element(a.annotation) AS annotation), true
          FROM cuda_annotations() a WHERE cuda_annotation_element(a.annotation) <> ''
        ), annotated AS (
          SELECT a.*, cuda_annotation_head(a.annotation) AS chain,
                 cuda_annotation_mixed(a.annotation) AS mixed
          FROM written a
        ), classified AS (
          SELECT a.*,
                 CASE WHEN a.mixed THEN NULL
                      WHEN dt.framework IS NOT NULL OR dc.key IS NOT NULL THEN 'device'
                      WHEN c.qualified = 'torch.Tensor' THEN 'tensor'
                      WHEN ac.key IS NOT NULL THEN 'array'
                      WHEN k.key IS NOT NULL THEN 'obj:' || k.key END AS ty,
                 coalesce(dt.framework, CASE WHEN dc.key IS NOT NULL THEN 'protocol'
                                             WHEN c.qualified = 'torch.Tensor' THEN 'torch' END,
                          '') AS fw
          FROM annotated a
          LEFT JOIN cuda_chains() c ON c.fact_id = a.fact_id AND c.chain = a.chain
          LEFT JOIN cuda_device_types() dt ON dt.qualified = c.qualified
          LEFT JOIN cuda_device_classes() dc ON dc.key = c.definition
          LEFT JOIN cuda_array_classes() ac ON ac.key = c.definition
          LEFT JOIN cuda_definitions() k ON k.key = c.definition AND k.kind = 'class'
        )
        SELECT CASE WHEN a.site = 'return' THEN 'r:' || a.fact_id || ':' || a.owner_id
                    WHEN a.site = 'field' OR a.is_attribute
                    THEN 'f:' || o.key || '.' || a.name
                    ELSE 'v:' || a.fact_id || ':' || a.owner_id || ':' || a.name END AS key,
               CASE WHEN a.element THEN 'seq:' || a.ty ELSE a.ty END AS ty, a.fw,
               'annotation ' || a.annotation AS why
        FROM classified a
        LEFT JOIN cuda_definitions() o ON o.fact_id = a.fact_id AND o.node_id = a.class_id
          AND o.kind = 'class'
        WHERE a.ty IS NOT NULL AND (a.site NOT IN ('field') AND NOT a.is_attribute
                                    OR o.key IS NOT NULL)
        UNION ALL
        SELECT 'v:' || m.fact_id || ':' || m.node_id || ':self', 'obj:' || m.class_key, '',
               'self'
        FROM cuda_methods() m
        WHERE EXISTS (SELECT 1 FROM cuda_stores() s
                      WHERE s.fact_id = m.fact_id AND s.scope_id = m.node_id AND s.name = 'self')
        UNION ALL
        SELECT key, ty, fw, why FROM (
          SELECT 'e:' || c.fact_id || ':' || c.node_id AS key,
                 cuda_created_array(c.qualified_name, c.node_text) AS ty,
                 CASE WHEN c.qualified_name LIKE 'cupy%' THEN 'cupy'
                      WHEN c.qualified_name LIKE 'numba%' THEN 'numba' ELSE 'torch' END AS fw,
                 c.qualified_name AS why
          FROM cuda_calls() c
          WHERE regexp_matches(c.qualified_name, '^(cupyx?|numba\.cuda|torch)\.')
        ) WHERE ty IS NOT NULL
        UNION ALL
        SELECT 'e:' || r.fact_id || ':' || r.node_id, 'module:' || n.qualified,
               CASE WHEN n.qualified = 'cupy' THEN 'cupy' WHEN n.qualified = 'torch' THEN 'torch'
                    ELSE 'numba' END, 'module ' || n.qualified
        FROM cuda_identifier_reads() r
        JOIN cuda_names() n ON n.fact_id = r.fact_id AND n.bound = r.name
        WHERE n.qualified IN ('cupy', 'torch', 'numba.cuda')
          AND NOT EXISTS (SELECT 1 FROM cuda_name_uses() u
                          WHERE u.fact_id = r.fact_id AND u.node_id = r.node_id)
        UNION ALL
        SELECT 'e:' || t.fact_id || ':' || t.node_id, 'obj:' || t.key, '', 'constructor'
        FROM cuda_call_targets() t WHERE t.kind = 'class'
    """


class CudaParameters(CudaRelation):
    """The parameters of every function in order, with the key of the variable each is."""

    stored = True
    query = """
        SELECT fact_id, function_id, position, p.name AS name, p.type AS annotation
        FROM (SELECT fact_id, node_id AS function_id, generate_subscripts(parameters, 1) AS position,
                     unnest(parameters) AS p
              FROM cuda_nodes() WHERE type = 'function_definition')
    """


class CudaAttributeReads(CudaRelation):
    """Each `object.name` read that is not a call's callee: the object's expression key, this
    expression's key and the member name."""

    stored = True
    query = r"""
        SELECT 'e:' || o.fact_id || ':' || o.node_id AS object_key,
               'e:' || a.fact_id || ':' || a.node_id AS attribute_key,
               m.name AS name
        FROM cuda_nodes() a
        JOIN cuda_nodes() o ON o.fact_id = a.fact_id AND o.parent_id = a.node_id AND o.sibling_index = 0
        JOIN cuda_nodes() m ON m.fact_id = a.fact_id AND m.parent_id = a.node_id AND m.sibling_index = 1
          AND m.type = 'identifier'
        JOIN cuda_nodes() p ON p.fact_id = a.fact_id AND p.node_id = a.parent_id
        WHERE a.type = 'attribute' AND NOT (p.type = 'call' AND a.sibling_index = 0)
    """


class CudaMethodCalls(CudaRelation):
    """Each `receiver.method(...)` call: the receiver's expression key, the call's, the name."""

    stored = True
    query = """
        SELECT 'e:' || fact_id || ':' || receiver_id AS receiver_key,
               'e:' || fact_id || ':' || node_id AS call_key, method AS name, node_text
        FROM cuda_calls() WHERE receiver_id IS NOT NULL
    """


class CudaTypeEdges(CudaRelation):
    """Where a type moves unchanged from one key to another. `only` restricts what moves:
    `arrays` lets a `device`, `tensor` or `array` through, nothing else, and a null `only` lets any type."""

    stored = True
    query = f"""
        SELECT 'e:' || c.fact_id || ':' || c.node_id AS src,
               'e:' || p.fact_id || ':' || p.node_id AS dst, CAST(NULL AS VARCHAR) AS only
        FROM cuda_nodes() p JOIN cuda_nodes() c ON c.fact_id = p.fact_id AND c.parent_id = p.node_id
        WHERE c.type IN ({_ARRAY_NODES}) AND (
              p.type IN ('parenthesized_expression', 'boolean_operator')
           OR (p.type = 'conditional_expression' AND c.sibling_index IN (0, 4))
           OR (p.type = 'named_expression' AND c.sibling_index = 2))
        UNION ALL
        SELECT 'e:' || c.fact_id || ':' || c.node_id, 'e:' || p.fact_id || ':' || p.node_id, 'arrays'
        FROM cuda_nodes() p JOIN cuda_nodes() c ON c.fact_id = p.fact_id AND c.parent_id = p.node_id
        WHERE c.type IN ({_ARRAY_NODES}) AND (
              (p.type = 'subscript' AND c.sibling_index = 0)
           OR p.type IN ('binary_operator', 'unary_operator')
           OR (p.type = 'comparison_operator' AND NOT EXISTS (
                 SELECT 1 FROM cuda_nodes() o WHERE o.fact_id = p.fact_id AND o.parent_id = p.node_id
                   AND (o.type IN ('is', 'is not', 'in', 'not in')
                        OR o.type = 'none' AND EXISTS (
                          SELECT 1 FROM cuda_nodes() q WHERE q.fact_id = p.fact_id
                            AND q.parent_id = p.node_id AND q.type IN ('==', '!='))))))
        UNION ALL
        SELECT variable, 'e:' || fact_id || ':' || node_id, NULL FROM cuda_name_uses()
        UNION ALL
        SELECT 'e:' || s.fact_id || ':' || s.source_id,
               'v:' || s.fact_id || ':' || s.scope_id || ':' || s.name, 'arrays'
        FROM cuda_stores() s WHERE s.kind = 'loop' AND s.source_id IS NOT NULL
        UNION ALL
        SELECT 'e:' || c.fact_id || ':' || c.node_id, 'r:' || r.fact_id || ':' || r.scope."function", NULL
        FROM cuda_nodes() r JOIN cuda_nodes() c ON c.fact_id = r.fact_id AND c.parent_id = r.node_id
          AND c.sibling_index = 1
        WHERE r.type = 'return_statement' AND r.scope."function" IS NOT NULL
          AND c.type IN ({_ARRAY_NODES})
        UNION ALL
        SELECT DISTINCT 'r:' || t.definition_fact || ':' || t.definition_id,
               'e:' || t.fact_id || ':' || t.node_id, NULL
        FROM cuda_call_targets() t WHERE t.kind = 'function'
        UNION ALL
        SELECT 'e:' || f.fact_id || ':' || f.source_id, m.field, NULL
        FROM cuda_field_stores() f
        JOIN cuda_definitions() c ON c.fact_id = f.fact_id AND c.node_id = f.class_id
        JOIN cuda_fields() m ON m.class_key = c.key AND m.name = f.name
        WHERE f.source_id IS NOT NULL
        UNION ALL
        SELECT DISTINCT 'e:' || a.fact_id || ':' || a.value_id, m.member, NULL
        FROM cuda_call_targets() c
        JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
          AND a.keyword IS NOT NULL
        JOIN cuda_members() m ON m.receiver = c.key AND m.name = a.keyword AND m.kind = 'field'
        WHERE c.kind = 'class'
        UNION ALL
        SELECT DISTINCT 'e:' || a.fact_id || ':' || a.value_id,
               'v:' || d.fact_id || ':' || d.node_id || ':' || p.name, NULL
        FROM cuda_call_targets() c
        JOIN cuda_definitions() d ON d.kind = 'function'
          AND d.key IN (c.key, c.key || '.__init__')
        JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
        JOIN cuda_parameters() p ON p.fact_id = d.fact_id AND p.function_id = d.node_id
          AND p.annotation = ''
          AND (p.name = a.keyword
               OR p.position = a.position + 1
                  + CASE WHEN d.class_id IS NOT NULL AND d.outer_id IS NULL
                          AND NOT EXISTS (SELECT 1 FROM cuda_decorators() x
                                          WHERE x.fact_id = d.fact_id
                                            AND x.definition_id = d.node_id
                                            AND x.written = 'staticmethod')
                         THEN 1 ELSE 0 END)
    """


class CudaUnpacks(CudaRelation):
    """The names a tuple assignment binds: the variable, the value assigned and the name's place."""

    stored = True
    query = """
        SELECT 'v:' || fact_id || ':' || scope_id || ':' || name AS variable,
               'e:' || fact_id || ':' || source_id AS source, position
        FROM cuda_stores() WHERE kind = 'unpack' AND source_id IS NOT NULL
    """


class CudaElements(CudaRelation):
    """Where one element of a container is taken out: the `source` expression and the `element`
    key it yields, a loop variable (`for stage in stages`) or an item (`stages[0]`, not a slice)."""

    stored = True
    query = """
        SELECT 'e:' || fact_id || ':' || source_id AS source,
               'v:' || fact_id || ':' || scope_id || ':' || name AS element
        FROM cuda_stores() WHERE kind = 'loop' AND source_id IS NOT NULL
        UNION ALL
        SELECT 'e:' || o.fact_id || ':' || o.node_id, 'e:' || p.fact_id || ':' || p.node_id
        FROM cuda_nodes() p
        JOIN cuda_nodes() o ON o.fact_id = p.fact_id AND o.parent_id = p.node_id
          AND o.sibling_index = 0
        JOIN cuda_nodes() i ON i.fact_id = p.fact_id AND i.parent_id = p.node_id
          AND i.sibling_index = 2
        WHERE p.type = 'subscript' AND i.type <> 'slice'
    """


class CudaVariableSources(CudaRelation):
    """The value each plain `name = value` stores in a variable. `literal` marks a `None`, which
    says nothing of the variable's type (`out = None` before `out = cupy.empty(n)`)."""

    stored = True
    query = """
        SELECT 'v:' || s.fact_id || ':' || s.scope_id || ':' || s.name AS variable,
               'e:' || s.fact_id || ':' || s.source_id AS source,
               v.type = 'none' AS literal
        FROM cuda_stores() s
        JOIN cuda_nodes() v ON v.fact_id = s.fact_id AND v.node_id = s.source_id
        WHERE s.kind = 'assignment' AND s.source_id IS NOT NULL
    """


_HOST_METHODS = (
    "'get', 'item', 'tolist', 'numpy', 'tobytes', 'tostring', 'dump', 'dumps', 'data_ptr',"
    " 'element_size', 'dim', 'size', 'numel', 'nelement', 'stride', 'is_contiguous', 'is_cuda',"
    " 'get_device', 'copy_to_host', 'storage', 'untyped_storage', 'is_floating_point',"
    " '__len__', '__int__', '__float__', '__bool__', '__index__', 'cuda_array_interface'"
)
_ARRAY_ATTRIBUTES = "'T', 'mT', 'H', 'mH', 'real', 'imag', 'flat', 'grad'"


def _fresh(key: str, *, ty: str) -> str:
    """The test that a branch of the fixpoint derives something new for `key`.

    It does when nothing is typed yet, or only an `array` (a placement not known) that a type `ty`
    other than `array` settles.
    """
    return (
        f"NOT EXISTS (SELECT 1 FROM recurring.t x WHERE x.key = {key} "
        f"AND (x.ty <> 'array' OR {ty} = 'array'))"
    )


class CudaTypes(CudaRelation):
    """The type of every key the seeds and edges reach: `ty` is `device`, `tensor`, `array`
    (placement not known), or `obj:<class key>`, `fw` the framework the memory comes from, `why`
    what put it there."""

    stored = True
    query = f"""
        WITH RECURSIVE t(key, ty, fw, why) USING KEY (key) AS (
          SELECT key, ty, fw, why FROM cuda_type_seeds()
          UNION
          (
            SELECT e.dst, d.ty, d.fw, 'flow'
            FROM t d JOIN cuda_type_edges() e ON e.src = d.key
            WHERE (e.only IS NULL OR d.ty IN ('device', 'tensor', 'array'))
              AND {_fresh("e.dst", ty="d.ty")}
            UNION ALL
            SELECT x.element, substr(d.ty, 5), d.fw, 'element'
            FROM t d JOIN cuda_elements() x ON x.source = d.key
            WHERE d.ty LIKE 'seq:%' AND {_fresh("x.element", ty="substr(d.ty, 5)")}
            UNION ALL
            SELECT a.variable, d.ty, d.fw, 'assignment'
            FROM t d JOIN cuda_variable_sources() a ON a.source = d.key
            WHERE {_fresh("a.variable", ty="d.ty")}
              AND NOT EXISTS (
                SELECT 1 FROM cuda_variable_sources() o
                WHERE o.variable = a.variable AND NOT o.literal
                  AND NOT EXISTS (SELECT 1 FROM recurring.t y
                                  WHERE y.key = o.source AND y.ty = d.ty))
            UNION ALL
            SELECT u.attribute_key, f.ty, f.fw, 'attribute ' || u.name
            FROM t d
            JOIN cuda_attribute_reads() u ON u.object_key = d.key
            JOIN cuda_members() m ON m.receiver = substr(d.ty, 5) AND m.name = u.name
              AND m.kind IN ('field', 'property')
            JOIN recurring.t f ON f.key = m.member
            WHERE d.ty LIKE 'obj:%' AND {_fresh("u.attribute_key", ty="f.ty")}
            UNION ALL
            SELECT u.attribute_key, d.ty, d.fw, 'attribute ' || u.name
            FROM t d
            JOIN cuda_members() m ON m.member = d.key AND m.kind IN ('field', 'property')
            JOIN cuda_attribute_reads() u ON u.name = m.name
            JOIN recurring.t o ON o.key = u.object_key AND o.ty = 'obj:' || m.receiver
            WHERE {_fresh("u.attribute_key", ty="d.ty")}
            UNION ALL
            SELECT c.call_key, r.ty, r.fw, 'method ' || c.name
            FROM t d
            JOIN cuda_method_calls() c ON c.receiver_key = d.key
            JOIN cuda_members() m ON m.receiver = substr(d.ty, 5) AND m.name = c.name
              AND m.kind = 'method'
            JOIN recurring.t r ON r.key = m.member
            WHERE d.ty LIKE 'obj:%' AND {_fresh("c.call_key", ty="r.ty")}
            UNION ALL
            SELECT c.call_key, d.ty, d.fw, 'method ' || c.name
            FROM t d
            JOIN cuda_members() m ON m.member = d.key AND m.kind = 'method'
            JOIN cuda_method_calls() c ON c.name = m.name
            JOIN recurring.t o ON o.key = c.receiver_key AND o.ty = 'obj:' || m.receiver
            WHERE {_fresh("c.call_key", ty="d.ty")}
            UNION ALL
            SELECT c.call_key,
                   CASE WHEN c.name = 'cuda' THEN 'device'
                        WHEN c.name = 'cpu' THEN 'tensor'
                        WHEN c.name = 'to' AND regexp_matches(c.node_text, 'cuda') THEN 'device'
                        WHEN c.name = 'to' AND regexp_matches(c.node_text, 'cpu') THEN 'tensor'
                        ELSE d.ty END,
                   CASE WHEN c.name IN ('cuda', 'cpu', 'to') THEN 'torch' ELSE d.fw END,
                   'method ' || c.name
            FROM t d JOIN cuda_method_calls() c ON c.receiver_key = d.key
            WHERE d.ty IN ('device', 'tensor', 'array') AND c.name NOT IN ({_HOST_METHODS})
              AND {_fresh("c.call_key", ty="d.ty")}
            UNION ALL
            SELECT c.call_key, cuda_created_array(substr(d.ty, 8) || '.' || c.name, c.node_text),
                   d.fw, 'method ' || c.name
            FROM t d JOIN cuda_method_calls() c ON c.receiver_key = d.key
            WHERE d.ty LIKE 'module:%'
              AND cuda_created_array(substr(d.ty, 8) || '.' || c.name, c.node_text) IS NOT NULL
              AND {_fresh("c.call_key", ty="'device'")}
            UNION ALL
            SELECT u.variable, f.ty, f.fw, 'unpack'
            FROM t d
            JOIN cuda_unpacks() u ON u.source = d.key
            JOIN cuda_field_order() o ON o.class_key = substr(d.ty, 5) AND o.position = u.position
            JOIN recurring.t f ON f.key = o.field
            WHERE d.ty LIKE 'obj:%' AND {_fresh("u.variable", ty="f.ty")}
            UNION ALL
            SELECT u.variable, d.ty, d.fw, 'unpack'
            FROM t d
            JOIN cuda_field_order() o ON o.field = d.key
            JOIN cuda_unpacks() u ON u.position = o.position
            JOIN recurring.t s ON s.key = u.source AND s.ty = 'obj:' || o.class_key
            WHERE {_fresh("u.variable", ty="d.ty")}
            UNION ALL
            SELECT u.attribute_key, d.ty, d.fw, 'attribute ' || u.name
            FROM t d JOIN cuda_attribute_reads() u ON u.object_key = d.key
            WHERE d.ty IN ('device', 'tensor', 'array') AND u.name IN ({_ARRAY_ATTRIBUTES})
              AND {_fresh("u.attribute_key", ty="d.ty")}
          )
        )
        SELECT * FROM t
    """


class CudaDeviceValues(CudaRelation):
    """Expressions holding device memory, one per node: `fw` the framework it comes from and
    `why` what makes it so (an annotation, a constructor, a field, a call)."""

    stored = True
    query = """
        SELECT n.fact_id, n.path, n.node_id, n.type, t.fw, t.why,
               n.start_line, n.start_column, n.end_line, n.end_column
        FROM cuda_types() t
        JOIN cuda_nodes() n ON t.key = 'e:' || n.fact_id || ':' || n.node_id
        WHERE t.ty = 'device'
    """


class CudaResolvedCalls(CudaRelation):
    """The definitions each call may reach: the named function or class (`cuda_call_targets`,
    `via` `named`), for `receiver.method(...)` the method of the receiver's class, found by its
    type (`receiver`), and the overrides of every class that inherits from it (`override`)."""

    stored = True
    query = """
        SELECT fact_id, node_id, key, kind, definition_id, definition_fact, 'named' AS via
        FROM cuda_call_targets()
        UNION
        SELECT c.fact_id, c.node_id, k.key, 'function', k.node_id, k.fact_id, 'receiver'
        FROM cuda_method_calls() m
        JOIN cuda_calls() c ON 'e:' || c.fact_id || ':' || c.node_id = m.call_key
        JOIN cuda_types() t ON t.key = m.receiver_key AND t.ty LIKE 'obj:%'
        JOIN cuda_members() mm ON mm.receiver = substr(t.ty, 5) AND mm.name = m.name
          AND mm.kind = 'method'
        JOIN cuda_methods() meth ON meth.ret = mm.member
        JOIN cuda_definitions() k ON k.fact_id = meth.fact_id AND k.node_id = meth.node_id
        UNION
        SELECT c.fact_id, c.node_id, k.key, 'function', k.node_id, k.fact_id, 'override'
        FROM cuda_method_calls() m
        JOIN cuda_calls() c ON 'e:' || c.fact_id || ':' || c.node_id = m.call_key
        JOIN cuda_types() t ON t.key = m.receiver_key AND t.ty LIKE 'obj:%'
        JOIN cuda_ancestors() a ON a.ancestor_key = substr(t.ty, 5) AND a.distance > 0
        JOIN cuda_methods() meth ON meth.class_key = a.class_key AND meth.name = m.name
        JOIN cuda_definitions() k ON k.fact_id = meth.fact_id AND k.node_id = meth.node_id
    """
