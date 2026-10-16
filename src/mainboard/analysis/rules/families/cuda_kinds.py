"""The scalar kinds device code declares (`i32`, `u64`, `f32` ...), for the checks on casts: which
names stand for a kind, what each variable of a device function is declared as, and which casts a
device function writes."""

from ...catalog import Function
from .cuda_base import CudaRelation


class CudaScalarKinds(CudaRelation):
    """The dotted names that stand for a scalar kind: the scalar types themselves and the module
    constants that alias one (`Index = i32`, through up to three aliases)."""

    stored = True
    query = """
        WITH RECURSIVE kinds(qualified, kind, hops) AS (
          SELECT 'patos.cuda.scalars.' || k, k, 0
          FROM (VALUES ('u8'), ('u16'), ('u32'), ('u64'), ('i8'), ('i16'), ('i32'), ('i64'),
                       ('f16'), ('f32'), ('f64')) t(k)
          UNION ALL
          SELECT d.qualified, s.kind, s.hops + 1
          FROM kinds s
          JOIN cuda_alias_targets() a ON a.target = s.qualified
          JOIN cuda_definitions() d ON d.fact_id = a.fact_id AND d.name = a.name
            AND d.kind = 'variable'
          WHERE s.hops < 3
        )
        SELECT DISTINCT qualified, kind FROM kinds
    """


class CudaAliasTargets(CudaRelation):
    """The module-level `Name = other_name` assignments, with what `other_name` resolves to."""

    stored = True
    query = """
        SELECT a.fact_id, a.name, h.qualified AS target
        FROM cuda_nodes() a
        JOIN cuda_nodes() r ON r.fact_id = a.fact_id AND r.parent_id = a.node_id
          AND r.sibling_index = 2 AND r.type IN ('identifier', 'attribute')
        JOIN cuda_chains() h ON h.fact_id = a.fact_id AND h.chain = r.name
        WHERE a.type = 'assignment' AND a.scope."function" IS NULL AND a.scope."class" IS NULL
    """


class CudaDeclaredKinds(CudaRelation):
    """The scalar kind each parameter and annotated local of a function is declared as, or for an
    array (`Vector[Index]`) the kind of its elements."""

    stored = True
    query = """
        SELECT a.fact_id, a.owner_id AS function_id, a.site, a.name, s.kind,
               e.kind AS element_kind, a.annotation
        FROM cuda_annotations() a
        LEFT JOIN cuda_chains() h ON h.fact_id = a.fact_id AND h.chain = cuda_annotation_head(a.annotation)
        LEFT JOIN cuda_scalar_kinds() s ON s.qualified = h.qualified
        LEFT JOIN cuda_chains() eh ON eh.fact_id = a.fact_id
          AND eh.chain = regexp_extract(a.annotation, '\\[\\s*([A-Za-z_][A-Za-z0-9_.]*)', 1)
        LEFT JOIN cuda_scalar_kinds() e ON e.qualified = eh.qualified
          AND regexp_matches(coalesce(h.qualified, ''), '\\.(Vector|Matrix)$')
        WHERE a.site IN ('parameter', 'return', 'local') AND NOT a.is_attribute
          AND (s.kind IS NOT NULL OR e.kind IS NOT NULL)
    """


class CudaCasts(CudaRelation):
    """Calls of a scalar type with one argument (`i32(x)`) inside device code: the kind, the
    argument and the function it is in."""

    stored = True
    query = """
        SELECT c.fact_id, c.path, c.node_id, c.scope_id, s.kind, a.value_id AS argument_id,
               c.start_line, c.start_column, c.end_line, c.end_column, c.shown
        FROM cuda_calls() c
        JOIN cuda_device_code() d ON d.fact_id = c.fact_id AND d.function_id = c.scope_id
        JOIN cuda_chains() h ON h.fact_id = c.fact_id AND h.chain = c.callee_text
        JOIN cuda_scalar_kinds() s ON s.qualified = h.qualified
        JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
          AND a.position = 0
        WHERE NOT EXISTS (SELECT 1 FROM cuda_arguments() o
                          WHERE o.fact_id = c.fact_id AND o.call_id = c.node_id
                            AND (o.position > 0 OR o.keyword IS NOT NULL))
    """


class CudaExpressionKinds(CudaRelation):
    """The scalar kind of an expression in device code where its annotations say: a variable
    declared with one (an augmented assignment's target too), an element of an array declared with
    one, a field of a record class declared with one, a cast, and a parenthesized one."""

    stored = True
    query = """
        WITH base AS (
          SELECT u.fact_id, u.node_id, k.kind
          FROM cuda_name_uses() u
          JOIN cuda_declared_kinds() k
            ON 'v:' || k.fact_id || ':' || k.function_id || ':' || k.name = u.variable
          WHERE k.kind IS NOT NULL
          UNION ALL
          SELECT s.fact_id, s.node_id, k.element_kind
          FROM cuda_nodes() s
          JOIN cuda_nodes() b ON b.fact_id = s.fact_id AND b.parent_id = s.node_id
            AND b.sibling_index = 0
          JOIN cuda_name_uses() u ON u.fact_id = b.fact_id AND u.node_id = b.node_id
          JOIN cuda_declared_kinds() k
            ON 'v:' || k.fact_id || ':' || k.function_id || ':' || k.name = u.variable
          WHERE s.type = 'subscript' AND k.element_kind IS NOT NULL
          UNION ALL
          SELECT a.fact_id, a.node_id, fk.kind
          FROM cuda_nodes() a
          JOIN cuda_nodes() b ON b.fact_id = a.fact_id AND b.parent_id = a.node_id
            AND b.sibling_index = 0
          JOIN cuda_nodes() f ON f.fact_id = a.fact_id AND f.parent_id = a.node_id
            AND f.sibling_index = 1
          JOIN cuda_name_uses() u ON u.fact_id = b.fact_id AND u.node_id = b.node_id
          JOIN cuda_annotations() v
            ON 'v:' || v.fact_id || ':' || v.owner_id || ':' || v.name = u.variable
           AND v.site IN ('parameter', 'local')
          JOIN cuda_chains() vh ON vh.fact_id = v.fact_id
            AND vh.chain = cuda_annotation_head(v.annotation)
          JOIN cuda_definitions() cls ON cls.key = vh.definition AND cls.kind = 'class'
          JOIN cuda_annotations() fa ON fa.fact_id = cls.fact_id AND fa.site = 'field'
            AND fa.owner_id = cls.node_id AND fa.name = f.name
          JOIN cuda_chains() fh ON fh.fact_id = fa.fact_id
            AND fh.chain = cuda_annotation_head(fa.annotation)
          JOIN cuda_scalar_kinds() fk ON fk.qualified = fh.qualified
          WHERE a.type = 'attribute'
          UNION ALL
          SELECT t.fact_id, t.node_id, k.kind
          FROM cuda_nodes() p
          JOIN cuda_nodes() t ON t.fact_id = p.fact_id AND t.parent_id = p.node_id
            AND t.sibling_index = 0 AND t.type = 'identifier'
          JOIN cuda_declared_kinds() k ON k.fact_id = t.fact_id
            AND k.function_id = t.scope."function" AND k.name = t.name
          WHERE p.type = 'augmented_assignment' AND k.kind IS NOT NULL
          UNION ALL
          SELECT fact_id, node_id, kind FROM cuda_casts()
        )
        SELECT * FROM base
        UNION ALL
        SELECT p.fact_id, p.node_id, k.kind
        FROM cuda_nodes() p
        JOIN cuda_nodes() i ON i.fact_id = p.fact_id AND i.parent_id = p.node_id
          AND i.type NOT IN ('(', ')')
        JOIN base k ON k.fact_id = i.fact_id AND k.node_id = i.node_id
        WHERE p.type = 'parenthesized_expression'
    """


_INTEGER = "'^[iu](8|16|32|64)$'"


def _width(kind: str) -> str:
    return f"CAST(substr({kind}, 2) AS INTEGER)"


class CudaPromoted(Function):
    """The kind an operand of C arithmetic takes first: an integer narrower than 32 bits is
    `i32`."""

    kind: str

    expression = "CASE WHEN regexp_matches(kind, '^[iu](8|16)$') THEN 'i32' ELSE kind END"


class CudaMeet(Function):
    """The kind two integer operands meet in under the C rules patos types device arithmetic by:
    two integers of different signedness meet in the unsigned one when it is at least as wide as
    the signed one, else in the signed one, each narrower than 32 bits promoted to `i32` first.
    Null for two of one signedness, which Numba types."""

    lhs: str
    rhs: str

    expression = f"""CASE WHEN regexp_matches(lhs, {_INTEGER}) AND regexp_matches(rhs, {_INTEGER})
        AND substr(cuda_promoted(lhs), 1, 1) <> substr(cuda_promoted(rhs), 1, 1)
      THEN CASE WHEN {_width("CASE WHEN starts_with(cuda_promoted(lhs), 'u') THEN cuda_promoted(lhs) ELSE cuda_promoted(rhs) END")}
                  >= {_width("CASE WHEN starts_with(cuda_promoted(lhs), 'i') THEN cuda_promoted(lhs) ELSE cuda_promoted(rhs) END")}
           THEN CASE WHEN starts_with(cuda_promoted(lhs), 'u') THEN cuda_promoted(lhs)
                     ELSE cuda_promoted(rhs) END
           ELSE CASE WHEN starts_with(cuda_promoted(lhs), 'i') THEN cuda_promoted(lhs)
                     ELSE cuda_promoted(rhs) END END END"""


class CudaCombined(Function):
    """The kind two operands of arithmetic, `min`, `max` or a comparison meet in: `cuda_meet`,
    else for two integers of one signedness the 64 bit kind of that signedness."""

    lhs: str
    rhs: str

    expression = f"""coalesce(cuda_meet(lhs, rhs),
      CASE WHEN regexp_matches(lhs, {_INTEGER}) AND regexp_matches(rhs, {_INTEGER})
                AND substr(lhs, 1, 1) = substr(rhs, 1, 1)
           THEN CASE WHEN starts_with(lhs, 'i') THEN 'i64' ELSE 'u64' END END)"""


class CudaShifted(Function):
    """The kind `lhs` shifted by `rhs` takes: their meeting when it is 64 bits wide, else the
    64 bit kind of the signedness of `lhs`."""

    lhs: str
    rhs: str

    expression = f"""CASE WHEN regexp_matches(coalesce(cuda_meet(lhs, rhs), ''), '64$')
           THEN cuda_meet(lhs, rhs)
           WHEN regexp_matches(lhs, {_INTEGER}) AND regexp_matches(rhs, {_INTEGER})
           THEN CASE WHEN starts_with(lhs, 'i') THEN 'i64' ELSE 'u64' END END"""


class CudaOperated(Function):
    """The kind `lhs op rhs` takes (`op` the operator as written), null where it is not
    told: a division, a power, a matrix product."""

    op: str
    lhs: str
    rhs: str

    expression = """CASE WHEN op IN ('/', '**', '@') THEN NULL
                         WHEN op IN ('<<', '>>') THEN cuda_shifted(lhs, rhs)
                         ELSE cuda_combined(lhs, rhs) END"""


class CudaCastRoots(CudaRelation):
    """Each cast with every expression it is a branch of: itself, and each `a if c else b` above
    it that it is the `a` or the `b` of, since a conversion of the whole converts the branch."""

    stored = True
    query = """
        WITH RECURSIVE up(fact_id, node_id, root_id) AS (
          SELECT fact_id, node_id, node_id FROM cuda_casts()
          UNION ALL
          SELECT u.fact_id, u.node_id, p.node_id
          FROM up u
          JOIN cuda_nodes() n ON n.fact_id = u.fact_id AND n.node_id = u.root_id
          JOIN cuda_nodes() p ON p.fact_id = n.fact_id AND p.node_id = n.parent_id
            AND p.type = 'conditional_expression'
          WHERE n.sibling_index IN (0, 4)
        )
        SELECT * FROM up
    """


class CudaOperations(CudaRelation):
    """The operands that meet in an operator the way the checker reads it: for each `operand` the
    `other` it meets, and `meeting`, `operated` for an arithmetic operator (`operator` as written)
    and `combined` for a comparison, `min` or `max`. A binary operator, an augmented assignment
    (the value meets the target), a comparison of two operands, and a call of `min` or `max` with
    two."""

    stored = True
    query = """
        SELECT p.fact_id, l.node_id AS operand_id, r.node_id AS other_id, o.type AS operator,
               'operated' AS meeting
        FROM cuda_nodes() p
        JOIN cuda_nodes() l ON l.fact_id = p.fact_id AND l.parent_id = p.node_id
          AND l.sibling_index = 0
        JOIN cuda_nodes() o ON o.fact_id = p.fact_id AND o.parent_id = p.node_id
          AND o.sibling_index = 1
        JOIN cuda_nodes() r ON r.fact_id = p.fact_id AND r.parent_id = p.node_id
          AND r.sibling_index = 2
        WHERE p.type = 'binary_operator'
        UNION ALL
        SELECT p.fact_id, r.node_id, l.node_id, o.type, 'operated'
        FROM cuda_nodes() p
        JOIN cuda_nodes() l ON l.fact_id = p.fact_id AND l.parent_id = p.node_id
          AND l.sibling_index = 0
        JOIN cuda_nodes() o ON o.fact_id = p.fact_id AND o.parent_id = p.node_id
          AND o.sibling_index = 1
        JOIN cuda_nodes() r ON r.fact_id = p.fact_id AND r.parent_id = p.node_id
          AND r.sibling_index = 2
        WHERE p.type = 'binary_operator'
        UNION ALL
        SELECT p.fact_id, r.node_id, l.node_id, rtrim(o.type, '='), 'operated'
        FROM cuda_nodes() p
        JOIN cuda_nodes() l ON l.fact_id = p.fact_id AND l.parent_id = p.node_id
          AND l.sibling_index = 0
        JOIN cuda_nodes() o ON o.fact_id = p.fact_id AND o.parent_id = p.node_id
          AND o.sibling_index = 1
        JOIN cuda_nodes() r ON r.fact_id = p.fact_id AND r.parent_id = p.node_id
          AND r.sibling_index = 2
        WHERE p.type = 'augmented_assignment'
        UNION ALL
        SELECT p.fact_id, x.node_id, y.node_id, '', 'combined'
        FROM cuda_nodes() p
        JOIN cuda_nodes() x ON x.fact_id = p.fact_id AND x.parent_id = p.node_id
          AND x.sibling_index IN (0, 2)
        JOIN cuda_nodes() y ON y.fact_id = p.fact_id AND y.parent_id = p.node_id
          AND y.sibling_index IN (0, 2) AND y.node_id <> x.node_id
        WHERE p.type = 'comparison_operator' AND p.children_count = 3
        UNION ALL
        SELECT c.fact_id, x.value_id, y.value_id, '', 'combined'
        FROM cuda_calls() c
        JOIN cuda_arguments() x ON x.fact_id = c.fact_id AND x.call_id = c.node_id
          AND x.keyword IS NULL AND x.position IN (0, 1)
        JOIN cuda_arguments() y ON y.fact_id = c.fact_id AND y.call_id = c.node_id
          AND y.keyword IS NULL AND y.position IN (0, 1) AND y.position <> x.position
        WHERE c.qualified_name IN ('builtins.min', 'builtins.max')
          AND (SELECT count(*) FROM cuda_arguments() n
               WHERE n.fact_id = c.fact_id AND n.call_id = c.node_id) = 2
    """


class CudaReturnKinds(CudaRelation):
    """The scalar kind each element of a function's `tuple[...]` return annotation is declared
    as, by position from 0."""

    stored = True
    query = r"""
        WITH split AS (
          SELECT fact_id, owner_id AS function_id,
                 string_split(regexp_extract(annotation,
                   '^(?:[A-Za-z_]\w*\.)*[Tt]uple\[(.*)\]$', 1), ',') AS parts
          FROM cuda_annotations()
          WHERE site = 'return' AND regexp_matches(annotation, '^(?:[A-Za-z_]\w*\.)*[Tt]uple\[')
        ), elements AS (
          SELECT fact_id, function_id, generate_subscripts(parts, 1) - 1 AS position,
                 trim(unnest(parts)) AS text
          FROM split
        )
        SELECT e.fact_id, e.function_id, e.position, s.kind
        FROM elements e
        JOIN cuda_chains() h ON h.fact_id = e.fact_id AND h.chain = e.text
        JOIN cuda_scalar_kinds() s ON s.qualified = h.qualified
    """
