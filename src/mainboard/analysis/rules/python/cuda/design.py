"""PY-CUDA rules about how GPU code is put together: cache keys, one definition per constant,
records on the device."""

from ....catalog import Example, Rule
from ..constants import cuda_paths


class CacheKeyMissesWhatMatters(Rule, id="PY-CUDA0011", applicability="review"):
    """A cache keyed by `id()` of the objects it describes (the key outlives or is reused by a
    different object), or a kernel cache whose digest hashes only the files of one folder."""

    exempt: tuple[str, ...] = cuda_paths.TESTS

    query = r"""
        WITH ids AS (
          SELECT c.fact_id, c.path, c.node_id, c.start_line, c.start_column, c.end_line,
                 c.end_column, c.shown, a.value_id AS keyed_id
          FROM cuda_calls() c
          LEFT JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
            AND a.position = 0
          WHERE c.qualified_name = 'builtins.id'
          UNION ALL
          SELECT i.fact_id, i.path, i.node_id, i.start_line, i.start_column, i.end_line,
                 i.end_column, 'map(id, ...)', a.value_id
          FROM cuda_nodes() i
          JOIN cuda_nodes() al ON al.fact_id = i.fact_id AND al.node_id = i.parent_id
            AND al.type = 'argument_list'
          JOIN cuda_calls() m ON m.fact_id = al.fact_id AND m.arguments_id = al.node_id
            AND m.qualified_name = 'builtins.map'
          LEFT JOIN cuda_arguments() a ON a.fact_id = m.fact_id AND a.call_id = m.node_id
            AND a.position = 1
          WHERE i.type = 'identifier' AND i.name = 'id'
        ), placed AS (
          SELECT i.*, b.node_id AS container_id
          FROM ids i
          JOIN cuda_nodes() s ON s.fact_id = i.fact_id AND s.type = 'subscript'
            AND i.node_id BETWEEN s.node_id + 1 AND s.node_id + s.descendant_count
          JOIN cuda_nodes() b ON b.fact_id = s.fact_id AND b.parent_id = s.node_id
            AND b.sibling_index = 0
          WHERE NOT i.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
          UNION ALL
          SELECT i.*, m.receiver_id
          FROM ids i
          JOIN cuda_calls() m ON m.fact_id = i.fact_id
            AND m.method IN ('get', 'setdefault', 'pop', '__getitem__', '__contains__', 'add',
                             'discard')
          JOIN cuda_nodes() a ON a.fact_id = m.fact_id AND a.node_id = m.arguments_id
          WHERE i.node_id BETWEEN a.node_id AND a.node_id + a.descendant_count
        ), texts AS (
          SELECT p.*, s.content,
                 syntax_slice(s.content, c.start_line, c.start_column, c.end_line, c.end_column)
                   AS container,
                 syntax_slice(s.content, k.start_line, k.start_column, k.end_line, k.end_column)
                   AS keyed
          FROM placed p
          JOIN source_files s ON s.path = p.path
          JOIN cuda_nodes() c ON c.fact_id = p.fact_id AND c.node_id = p.container_id
          LEFT JOIN cuda_nodes() k ON k.fact_id = p.fact_id AND k.node_id = p.keyed_id
        ), stores AS (
          SELECT a.fact_id,
                 syntax_slice(s.content, b.start_line, b.start_column, b.end_line, b.end_column)
                   AS container,
                 syntax_slice(s.content, v.start_line, v.start_column, v.end_line, v.end_column)
                   AS stored
          FROM cuda_nodes() a
          JOIN source_files s ON s.path = a.path
          JOIN cuda_nodes() l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
            AND l.sibling_index = 0 AND l.type = 'subscript'
          JOIN cuda_nodes() b ON b.fact_id = l.fact_id AND b.parent_id = l.node_id
            AND b.sibling_index = 0
          JOIN cuda_nodes() v ON v.fact_id = a.fact_id AND v.parent_id = a.node_id
            AND v.sibling_index = (SELECT max(w.sibling_index) FROM cuda_nodes() w
                                   WHERE w.fact_id = a.fact_id AND w.parent_id = a.node_id)
          WHERE a.type = 'assignment'
        ), marked AS (
          SELECT t.*,
                 regexp_matches(t.keyed, '^[A-Za-z_][A-Za-z0-9_.]*$') AND EXISTS (
                   SELECT 1 FROM stores x
                   WHERE x.fact_id = t.fact_id AND x.container = t.container
                     AND regexp_matches(x.stored, '(^|[^A-Za-z0-9_.])' || replace(t.keyed, '.', '\.')
                                                  || '($|[^A-Za-z0-9_])')) AS holds
          FROM texts t
        ), keyed AS (
          SELECT path, start_line, start_column, end_line, end_column, shown, 'id' AS problem
          FROM marked
          GROUP BY fact_id, node_id, path, start_line, start_column, end_line, end_column, shown
          HAVING NOT bool_or(holds)
        ), digests AS (
          SELECT g.path, g.start_line, g.start_column, g.end_line, g.end_column, g.shown,
                 'digest' AS problem
          FROM cuda_calls() g
          JOIN cuda_arguments() p ON p.fact_id = g.fact_id AND p.call_id = g.node_id
            AND p.position = 0
          JOIN cuda_nodes() pattern ON pattern.fact_id = p.fact_id AND pattern.node_id = p.value_id
          JOIN source_files s ON s.path = g.path
          WHERE g.method = 'glob' AND contains(g.callee_text, '__file__')
            AND NOT contains(syntax_slice(s.content, pattern.start_line, pattern.start_column,
                                          pattern.end_line, pattern.end_column), '**')
            AND EXISTS (SELECT 1 FROM cuda_calls() h
                        WHERE h.fact_id = g.fact_id
                          AND regexp_matches(h.callee_text,
                                '(^|\.)(digest|sha1|sha256|sha512|md5|blake2b|blake2s|hexdigest)$'))
        )
        SELECT f.*,
               CASE f.problem
                 WHEN 'id' THEN 'keys a cache by `id()`: ids are reused once an object dies, so a '
                      || 'later object can land on an earlier one''s entry, and it never matches '
                      || 'an equal object that is a different instance'
                 ELSE 'lists only the files directly in one folder for a cache digest: a source '
                      || 'in a subpackage that shapes the compiled code does not move the key, so '
                      || 'a stale kernel loads from disk' END AS what,
               CASE f.problem
                 WHEN 'id' THEN 'key by a hashable value of the thing (its type name, its '
                      || 'signature) or keep the keyed objects alive in the cache'
                 ELSE 'hash the whole package tree (`rglob("*.py")`) and the installed '
                      || 'distribution''s files' END AS remedy
        FROM (SELECT * FROM keyed UNION ALL SELECT * FROM digests) f
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(f.path, p))), false)
    """
    message = "`{shown}` {what}. Fix: {remedy}"
    examples = (
        Example(
            bad="import cupy as cp\n\nCACHE = {}\n\n\ndef compiled(kind):\n    return CACHE.get(id(kind))\n",
            good="import cupy as cp\n\nCACHE = {}\n\n\ndef compiled(kind):\n    return CACHE.get(kind)\n",
        ),
        Example(
            bad='import hashlib\nfrom pathlib import Path\n\nimport cupy as cp\n\nSOURCES = tuple(Path(__file__).parent.glob("*.py"))\n\n\ndef folder():\n    return hashlib.sha256(b"x").hexdigest()\n',
            good='import hashlib\nfrom pathlib import Path\n\nimport cupy as cp\n\nSOURCES = tuple(Path(__file__).parent.rglob("*.py"))\n\n\ndef folder():\n    return hashlib.sha256(b"x").hexdigest()\n',
        ),
    )


class OneDefinitionPerConstant(Rule, id="PY-CUDA0012", applicability="review"):
    """A bit-mixing constant written in several places (host, device function and C++ source in a
    string), or packing helpers that take their operands in opposite order."""

    exempt: tuple[str, ...] = cuda_paths.MEASURING
    minimum_digits: int = 12
    minimum_distinct: int = 5

    query = r"""
        WITH literals AS (
          SELECT n.fact_id, n.path, n.start_line, n.start_column, n.end_line, n.end_column,
                 coalesce(n.scope."function", -1) AS scope_id,
                 regexp_replace(lower(replace(n.name, '_', '')), '[ul]+$', '') AS value
          FROM cuda_nodes() n WHERE n.type = 'integer' AND regexp_matches(n.name, '^0[xX]')
          UNION ALL
          SELECT n.fact_id, n.path, n.start_line, n.start_column, n.end_line, n.end_column,
                 coalesce(n.scope."function", -1),
                 lower(unnest(regexp_extract_all(n.name, '0[xX][0-9a-fA-F]{12,}')))
          FROM cuda_nodes() n WHERE n.type = 'string_content' AND contains(lower(n.name), '0x')
        ), mixes AS (
          SELECT l.*, coalesce(d.qualname, '<module>') AS definition
          FROM literals l
          LEFT JOIN cuda_definitions() d ON d.fact_id = l.fact_id AND d.node_id = l.scope_id
            AND d.kind = 'function'
          WHERE length(l.value) - 2 >= $minimum_digits
            AND len(list_distinct(string_split(substr(l.value, 3), ''))) >= $minimum_distinct
            AND NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(l.path, p))), false)
        ), repeated AS (
          SELECT m.*, count(DISTINCT m.path || '::' || m.definition) OVER (PARTITION BY m.value) AS sites,
                 array_to_string(list_sort(list(DISTINCT m.path || ':' || m.start_line) OVER (PARTITION BY m.value)), ', ') AS where_written
          FROM mixes m
        ), shifted AS (
          SELECT b.fact_id, b.path, coalesce(b.scope."function", -1) AS scope_id, b.start_line,
                 b.start_column, op.type || k.name AS shift
          FROM cuda_nodes() b
          JOIN cuda_nodes() op ON op.fact_id = b.fact_id AND op.parent_id = b.node_id
            AND op.sibling_index = 1 AND op.type IN ('<<', '>>')
          JOIN cuda_nodes() k ON k.fact_id = b.fact_id AND k.parent_id = b.node_id
            AND k.sibling_index = 2 AND k.type = 'integer'
          WHERE b.type = 'binary_operator'
        ), python_mixes AS (
          SELECT s.fact_id, s.path, s.scope_id, min(s.start_line) AS start_line,
                 min(s.start_column) AS start_column,
                 string_agg(s.shift, ' ' ORDER BY s.start_line, s.start_column) AS signature,
                 count(*) AS shifts
          FROM shifted s
          WHERE s.scope_id <> -1 AND EXISTS (
              SELECT 1 FROM cuda_nodes() m WHERE m.fact_id = s.fact_id AND m.type = 'binary_operator'
                AND coalesce(m.scope."function", -1) = s.scope_id
                AND EXISTS (SELECT 1 FROM cuda_nodes() o WHERE o.fact_id = m.fact_id
                              AND o.parent_id = m.node_id AND o.sibling_index = 1 AND o.type = '*'))
          GROUP BY s.fact_id, s.path, s.scope_id
        ), string_mixes AS (
          SELECT n.fact_id, n.path, -1 AS scope_id, n.start_line, n.start_column,
                 list_aggregate(list_transform(regexp_extract_all(n.name, '(<<|>>)\s*\d+'),
                                lambda t: regexp_replace(t, '\s', '', 'g')), 'string_agg', ' ')
                   AS signature,
                 len(regexp_extract_all(n.name, '(<<|>>)\s*\d+')) AS shifts
          FROM cuda_nodes() n WHERE n.type = 'string_content' AND contains(n.name, '*')
            AND len(regexp_extract_all(n.name, '(<<|>>)\s*\d+')) >= 3
        ), copied AS (
          SELECT m.*, coalesce(d.qualname, 'a source string') AS definition,
                 count(DISTINCT m.path || '::' || m.scope_id || '::' || m.start_line) OVER (PARTITION BY m.signature) AS sites,
                 array_to_string(list_sort(list(DISTINCT m.path || ':' || m.start_line) OVER (PARTITION BY m.signature)), ', ') AS where_written
          FROM (SELECT * FROM python_mixes UNION ALL SELECT * FROM string_mixes) m
          LEFT JOIN cuda_definitions() d ON d.fact_id = m.fact_id AND d.node_id = m.scope_id
            AND d.kind = 'function'
          WHERE m.shifts >= 3
            AND NOT coalesce(list_bool_or(list_transform($exempt, lambda q: regexp_matches(m.path, q))), false)
        ), ordered AS (
          SELECT p.path, p.start_line, p.start_column, p.end_line, p.end_column,
                 p.qualname || '(' || CASE WHEN p.high_first THEN p.high || ', ' || p.low
                                           ELSE p.low || ', ' || p.high END || ')' AS value,
                 CAST(2 AS BIGINT) AS sites,
                 CASE WHEN q.high_first THEN q.qualname || '(' || q.high || ', ' || q.low || ')'
                      ELSE q.qualname || '(' || q.low || ', ' || q.high || ')' END AS where_written,
                 'order' AS problem
          FROM cuda_packers() p
          JOIN cuda_packers() q ON q.shift = p.shift AND q.high_first <> p.high_first
          WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda m: regexp_matches(p.path, m))), false)
        )
        SELECT u.*,
               CASE u.problem
                 WHEN 'mix' THEN 'is the same bit-mixing steps written in ' || u.sites
                      || ' places (' || u.where_written || '): a fix to one copy leaves the others '
                      || 'hashing differently, and host and device stop agreeing'
                 WHEN 'constant' THEN 'is written in ' || u.sites || ' places (' || u.where_written
                      || '), and the copies drift apart when one changes, so host and device hashes '
                      || 'stop agreeing'
                 ELSE 'packs its operands in the opposite order from `' || u.where_written
                      || '`: the same operation takes (high, low) in one and (low, high) in the '
                      || 'other, so a call that swaps them compiles and packs the wrong key' END
                 AS what,
               CASE u.problem
                 WHEN 'mix' THEN 'keep one definition (the device function) and generate the C++ '
                      || 'from it, or pin the copies together with a bit-for-bit test'
                 WHEN 'constant' THEN 'define it once and import it (C++ source takes it from the '
                      || 'Python module)'
                 ELSE 'keep one packing helper with keyword-only operands' END AS remedy
        FROM (
          SELECT r.path, r.start_line, r.start_column, r.end_line, r.end_column, r.value,
                 r.sites, r.where_written, 'constant' AS problem
          FROM repeated r WHERE r.sites > 1
          UNION ALL
          SELECT c.path, c.start_line, c.start_column, c.start_line, c.start_column,
                 c.definition || ' (' || c.signature || ')', c.sites, c.where_written, 'mix'
          FROM copied c WHERE c.sites > 1
          UNION ALL
          SELECT * FROM ordered) u
    """
    message = "`{value}` {what}. Fix: {remedy}"
    examples = (
        Example(
            bad=(
                """import cupy as cp


def left(x):
    return x * 0x9E3779B97F4A7C15


def right(y):
    return y * 0x9E3779B97F4A7C15
"""
            ),
            good=(
                """import cupy as cp

MIX = 0x9E3779B97F4A7C15


def left(x):
    return x * MIX


def right(y):
    return y * MIX
"""
            ),
        ),
        Example(
            bad="import cupy as cp\n\n\ndef pair(left: int, right: int) -> int:\n    return (left << 32) | right\n\n\ndef join(low: int, high: int) -> int:\n    return (high << 32) | low\n",
            good="import cupy as cp\n\n\ndef pair(left: int, right: int) -> int:\n    return (left << 32) | right\n\n\ndef join(high: int, low: int) -> int:\n    return (high << 32) | low\n",
        ),
    )


class RecordOnDeviceIsNamedTuple(Rule, id="PY-CUDA0010", applicability="review"):
    """A `NamedTuple` used as a value inside device code (built there, or typed on a device
    function's parameter or return): the project's device record is `Struct`, one record concept
    instead of two."""

    exempt: tuple[str, ...] = cuda_paths.TESTS
    change = "rewrote `{name}({base})` as `{name}(Struct)`"

    query = r"""
        WITH tuples AS (
          SELECT d.key AS class_key, d.fact_id, d.path, d.name, d.start_line, d.start_column,
                 d.end_line, d.end_column, b.written, ba.node_id AS base_id,
                 ba.start_line AS base_line, ba.start_column AS base_column,
                 ba.end_line AS base_end_line, ba.end_column AS base_end_column
          FROM cuda_definitions() d
          JOIN cuda_bases() b ON b.class_key = d.key AND b.qualified = 'typing.NamedTuple'
          JOIN cuda_nodes() ar ON ar.fact_id = d.fact_id AND ar.parent_id = d.node_id
            AND ar.type = 'argument_list'
          JOIN cuda_nodes() ba ON ba.fact_id = d.fact_id AND ba.parent_id = ar.node_id
            AND ba.name = b.written
        ), built AS (
          SELECT t.key AS class_key, c.path, c.start_line, c.shown, f.qualname AS function_name
          FROM cuda_resolved_calls() t
          JOIN cuda_calls() c ON c.fact_id = t.fact_id AND c.node_id = t.node_id
          JOIN cuda_device_code() f ON f.fact_id = c.fact_id AND f.function_id = c.scope_id
          WHERE t.kind = 'class'
          UNION
          SELECT h.definition, a.path, a.start_line, a.annotation, f.qualname
          FROM cuda_annotations() a
          JOIN cuda_device_code() f ON f.fact_id = a.fact_id AND f.function_id = a.owner_id
          JOIN cuda_chains() h ON h.fact_id = a.fact_id AND h.chain = cuda_annotation_head(a.annotation)
          WHERE a.site IN ('parameter', 'return') AND h.definition IS NOT NULL
        ), used AS (
          SELECT class_key, min(path || ':' || start_line) AS first_use,
                 arg_min(function_name, path || ':' || start_line) AS function_name
          FROM built GROUP BY class_key
        )
        SELECT t.path, t.start_line, t.start_column, t.end_line, t.end_column, t.name,
               t.written AS base, t.base_line, t.base_column, t.base_end_line, t.base_end_column,
               u.first_use, u.function_name
        FROM tuples t JOIN used u ON u.class_key = t.class_key
        WHERE EXISTS (SELECT 1 FROM cuda_names() n
                      WHERE regexp_matches(n.qualified, '^patos\.cuda\.typed(\.\w+)*\.Struct$'))
          AND NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(t.path, p))), false)
    """
    message = (
        "`{name}` is a `NamedTuple` used on the device (`{function_name}`, {first_use}): the device "
        "record of this project is `Struct`, so a value that crosses the kernel boundary has two "
        "record concepts, each with its own typing and calling rules. Fix: declare `{name}` as a "
        "`Struct`"
    )
    fix = """
        SELECT finding_key, path, base_line AS start_line, base_column AS start_column,
               base_end_line AS end_line, base_end_column AS end_column, 'Struct' AS text,
               'from patos.cuda.typed import Struct' AS import
        FROM finding
    """
    examples = (
        Example(
            bad=(
                """from typing import NamedTuple

from patos.cuda.typed import Struct, device


class Span(NamedTuple):
    start: int
    end: int


class Spans(Struct):
    starts: int

    @device
    def get(self, index: int) -> Span:
        return Span(index, index)
"""
            ),
            good=(
                """from typing import NamedTuple

from patos.cuda.typed import Struct, device


class Span(Struct):
    start: int
    end: int


class Spans(Struct):
    starts: int

    @device
    def get(self, index: int) -> Span:
        return Span(index, index)
"""
            ),
        ),
    )


class RedundantScalarCast(Rule, id="PY-CUDA0013", applicability="safe"):
    """A scalar cast in device code that something already does: the value has that kind (a
    variable, an array element or a record field declared with it), the parameter, the declared
    variable, the array element or the return annotation (each element of a tuple return, each
    branch of a conditional) converts to it on its own, or the operator that meets it with an
    operand of that kind does (an arithmetic operator, an augmented assignment, a comparison,
    `min` or `max`), under the C integer rules patos types device arithmetic by."""

    exempt: tuple[str, ...] = cuda_paths.TESTS
    change = "dropped the cast `{shown}`, keeping `{argument}`"

    query = r"""
        WITH callers AS (
          SELECT a.fact_id, a.value_id AS cast_id, a.call_id, a.position, a.keyword
          FROM cuda_arguments() a
        ), hits AS (
          SELECT c.*, 'kind' AS why
          FROM cuda_casts() c
          JOIN cuda_expression_kinds() k ON k.fact_id = c.fact_id AND k.node_id = c.argument_id
            AND k.kind = c.kind
          UNION
          SELECT c.*, 'parameter'
          FROM cuda_casts() c
          JOIN cuda_cast_roots() q ON q.fact_id = c.fact_id AND q.node_id = c.node_id
          JOIN callers a ON a.fact_id = q.fact_id AND a.cast_id = q.root_id
          JOIN cuda_resolved_calls() r ON r.fact_id = a.fact_id AND r.node_id = a.call_id
            AND r.kind = 'function' AND r.via <> 'override'
          JOIN cuda_definitions() f ON f.fact_id = r.definition_fact AND f.node_id = r.definition_id
          JOIN cuda_parameters() p ON p.fact_id = f.fact_id AND p.function_id = f.node_id
            AND (p.name = a.keyword
                 OR p.position = a.position + 1
                    + CASE WHEN f.class_id IS NOT NULL AND f.outer_id IS NULL
                                AND NOT EXISTS (SELECT 1 FROM cuda_decorators() x
                                                WHERE x.fact_id = f.fact_id
                                                  AND x.definition_id = f.node_id
                                                  AND x.written = 'staticmethod')
                           THEN 1 ELSE 0 END)
          JOIN cuda_declared_kinds() k ON k.fact_id = f.fact_id AND k.function_id = f.node_id
            AND k.site = 'parameter' AND k.name = p.name AND k.kind = c.kind
          UNION
          SELECT c.*, 'declaration'
          FROM cuda_casts() c
          JOIN cuda_cast_roots() q ON q.fact_id = c.fact_id AND q.node_id = c.node_id
          JOIN cuda_stores() s ON s.fact_id = q.fact_id AND s.source_id = q.root_id
            AND s.kind = 'assignment'
          JOIN cuda_declared_kinds() k ON k.fact_id = s.fact_id AND k.function_id = s.scope_id
            AND k.name = s.name AND k.kind = c.kind AND k.site IN ('parameter', 'local')
          UNION
          SELECT c.*, 'return'
          FROM cuda_casts() c
          JOIN cuda_cast_roots() q ON q.fact_id = c.fact_id AND q.node_id = c.node_id
          JOIN cuda_nodes() r ON r.fact_id = c.fact_id AND r.type = 'return_statement'
            AND r.scope."function" = c.scope_id
          JOIN cuda_nodes() v ON v.fact_id = r.fact_id AND v.parent_id = r.node_id
            AND v.node_id = q.root_id
          JOIN cuda_declared_kinds() k ON k.fact_id = c.fact_id AND k.function_id = c.scope_id
            AND k.site = 'return' AND k.kind = c.kind
          UNION
          SELECT c.*, 'tuple'
          FROM cuda_casts() c
          JOIN cuda_cast_roots() q ON q.fact_id = c.fact_id AND q.node_id = c.node_id
          JOIN cuda_nodes() x ON x.fact_id = q.fact_id AND x.node_id = q.root_id
          JOIN cuda_nodes() t ON t.fact_id = x.fact_id AND t.node_id = x.parent_id
            AND t.type IN ('expression_list', 'tuple')
          JOIN cuda_nodes() r ON r.fact_id = t.fact_id AND r.node_id = t.parent_id
            AND r.type = 'return_statement' AND r.scope."function" = c.scope_id
          JOIN cuda_return_kinds() e ON e.fact_id = c.fact_id AND e.function_id = c.scope_id
            AND e.kind = c.kind
            AND e.position = (SELECT count(*) FROM cuda_nodes() b
                              WHERE b.fact_id = t.fact_id AND b.parent_id = t.node_id
                                AND b.sibling_index < x.sibling_index
                                AND b.type NOT IN ('(', ',', ')'))
          UNION
          SELECT c.*, 'element'
          FROM cuda_casts() c
          JOIN cuda_cast_roots() q ON q.fact_id = c.fact_id AND q.node_id = c.node_id
          JOIN cuda_nodes() a ON a.fact_id = c.fact_id AND a.type = 'assignment'
          JOIN cuda_nodes() l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
            AND l.sibling_index = 0 AND l.type = 'subscript'
          JOIN cuda_nodes() b ON b.fact_id = l.fact_id AND b.parent_id = l.node_id AND b.sibling_index = 0
          JOIN cuda_name_uses() u ON u.fact_id = b.fact_id AND u.node_id = b.node_id
          JOIN cuda_declared_kinds() k
            ON 'v:' || k.fact_id || ':' || k.function_id || ':' || k.name = u.variable
           AND k.element_kind = c.kind
          WHERE q.root_id = (SELECT max(r.node_id) FROM cuda_nodes() r
                             WHERE r.fact_id = a.fact_id AND r.parent_id = a.node_id)
            AND a.node_id < q.root_id AND q.root_id <= a.node_id + a.descendant_count
          UNION
          SELECT c.*, 'operator'
          FROM cuda_casts() c
          JOIN cuda_expression_kinds() inner_kind ON inner_kind.fact_id = c.fact_id
            AND inner_kind.node_id = c.argument_id
          JOIN cuda_operations() o ON o.fact_id = c.fact_id AND o.operand_id = c.node_id
          JOIN cuda_expression_kinds() other_kind ON other_kind.fact_id = o.fact_id
            AND other_kind.node_id = o.other_id AND other_kind.kind = c.kind
          WHERE CASE o.meeting
                  WHEN 'combined' THEN cuda_combined(c.kind, inner_kind.kind)
                  ELSE cuda_operated(o.operator, c.kind, inner_kind.kind) END = c.kind
        )
        SELECT h.path, h.start_line, h.start_column, h.end_line, h.end_column, h.shown, h.kind,
               CASE h.why
                 WHEN 'kind' THEN 'the value is already `' || h.kind || '`'
                 WHEN 'parameter' THEN 'the call converts the argument to the parameter''s `'
                      || h.kind || '`'
                 WHEN 'declaration' THEN 'the assignment converts to the declared `' || h.kind || '`'
                 WHEN 'return' THEN 'the return annotation converts to `' || h.kind || '`'
                 WHEN 'tuple' THEN 'the return annotation converts this element to `' || h.kind || '`'
                 WHEN 'operator' THEN 'the operator converts to the `' || h.kind
                      || '` of the other operand'
                 ELSE 'the store into the array converts to its `' || h.kind || '` element' END
                 AS reason, arg.start_line AS argument_line, arg.start_column AS argument_column,
               arg.end_line AS argument_end_line, arg.end_column AS argument_end_column,
               syntax_slice(s.content, arg.start_line, arg.start_column, arg.end_line,
                            arg.end_column) AS argument,
               arg.type IN ('binary_operator', 'conditional_expression', 'comparison_operator',
                            'boolean_operator', 'lambda', 'not_operator', 'unary_operator')
               AND enclosing.type IN ('binary_operator', 'unary_operator', 'comparison_operator',
                                   'boolean_operator', 'not_operator', 'attribute', 'subscript',
                                   'conditional_expression', 'await') AS needs_parentheses
        FROM hits h
        JOIN cuda_nodes() arg ON arg.fact_id = h.fact_id AND arg.node_id = h.argument_id
        JOIN cuda_nodes() site ON site.fact_id = h.fact_id AND site.node_id = h.node_id
        JOIN cuda_nodes() enclosing ON enclosing.fact_id = site.fact_id AND enclosing.node_id = site.parent_id
        JOIN source_files s ON s.path = h.path
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(h.path, p))), false)
    """
    message = "`{shown}` is a redundant cast: {reason}, so dropping it changes nothing. Fix: write `{argument}`"
    fix = """
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               CASE WHEN needs_parentheses THEN '(' || argument || ')' ELSE argument END AS text
        FROM finding
    """
    examples = (
        Example(
            bad=(
                """from patos.cuda.scalars import i32
from patos.cuda.typed import device


@device
def twice(value: i32) -> i32:
    return i32(value) * 2
"""
            ),
            good=(
                """from patos.cuda.scalars import i32
from patos.cuda.typed import device


@device
def twice(value: i32) -> i32:
    return value * 2
"""
            ),
        ),
        Example(
            bad="from patos.cuda.scalars import i32\nfrom patos.cuda.typed import device\n\n\n@device\ndef bump(value: i32) -> i32:\n    total: i32 = i32(value + 1)\n    return total\n",
            good="from patos.cuda.scalars import i32\nfrom patos.cuda.typed import device\n\n\n@device\ndef bump(value: i32) -> i32:\n    total: i32 = value + 1\n    return total\n",
        ),
        Example(
            bad=(
                """from patos.cuda.scalars import i16, i64, u8, u32, u64
from patos.cuda.typed import device


@device
def widen(wide: u64, narrow: u8, step: u32, small: i16, big: i64) -> None:
    total = wide + u64(narrow)
    least = min(big, i64(small))
    step += u32(narrow)
"""
            ),
            good=(
                """from patos.cuda.scalars import i16, i64, u8, u32, u64
from patos.cuda.typed import device


@device
def widen(wide: u64, narrow: u8, step: u32, small: i16, big: i64) -> None:
    total = wide + narrow
    least = min(big, small)
    step += narrow
"""
            ),
        ),
        Example(
            bad=(
                """from patos.cuda.scalars import u32
from patos.cuda.typed import Struct, device


class Header(Struct):
    mask: u32


@device
def low(header: Header, wide: bool) -> tuple[u32, u32]:
    return u32(header.mask) if wide else header.mask, u32(header.mask)
"""
            ),
            good=(
                """from patos.cuda.scalars import u32
from patos.cuda.typed import Struct, device


class Header(Struct):
    mask: u32


@device
def low(header: Header, wide: bool) -> tuple[u32, u32]:
    return header.mask if wide else header.mask, header.mask
"""
            ),
        ),
        Example(
            bad=(
                """from ..typed import device, u32


@device
def keep(value: u32) -> u32:
    return u32(value)
"""
            ),
            good=(
                """from ..typed import device, u32


@device
def keep(value: u32) -> u32:
    return value
"""
            ),
            path="patos/cuda/primitives/keep.py",
            context={
                "pyproject.toml": '[project]\nname = "patos"\nversion = "0"\n',
                "patos/__init__.py": "",
                "patos/cuda/__init__.py": "",
                "patos/cuda/primitives/__init__.py": "",
                "patos/cuda/dec.py": "def device(function):\n    return function\n",
                "patos/cuda/scalars.py": (
                    "from numba import cuda\nimport numpy as np\n\nu32 = np.uint32\n"
                ),
                "patos/cuda/typed.py": (
                    'from .dec import device\nfrom .scalars import u32\n\n__all__ = ["device", "u32"]\n'
                ),
            },
        ),
    )
