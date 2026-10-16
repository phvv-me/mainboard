"""What the annotations of a project say a value is: a device array, an instance of a project
class, or nothing the PY-CUDA facts can use (a host array, a container, a union that mixes kinds).

An annotation is read as text. Its head is the first dotted name once the wrappers that add no
type (`Optional[`, `Runtime[`, `Annotated[`, `Final[` ...) and a `None` alternative are peeled, and
the head resolves through the file's imports to a dotted name (`cp.ndarray` -> `cupy.ndarray`) or
to a class of the project.
"""

from ...catalog import Function
from .cuda_base import CudaRelation

_WRAPPERS = r"^(?:[A-Za-z_]\w*\.)*(?:Optional|Annotated|Runtime|Final|ClassVar|Required|NotRequired|InitVar|Unpack)\["
_INNER = r"\[[^\[\]]*\]"
_NONE = r"\s*\|\s*None\b|\bNone\s*\|\s*"
_QUOTES = "' \"'''"


def _peeled(text: str) -> str:
    """`text` without its quotes, its wrapper names (three levels) and a leading `None |`."""
    unwrapped = f"trim({text}, {_QUOTES})"
    for _ in range(3):
        unwrapped = f"regexp_replace({unwrapped}, '{_WRAPPERS}', '')"
    return f"regexp_replace({unwrapped}, '^None\\s*\\|\\s*', '')"


class CudaAnnotationHead(Function):
    """The first dotted name of an annotation, past wrappers: `Runtime[SpanArray]` ->
    `SpanArray`, `cp.ndarray | None` -> `cp.ndarray`, `"Spans"` -> `Spans`."""

    text: str

    expression = f"regexp_extract({_peeled('text')}, '^([A-Za-z_][A-Za-z0-9_.]*)', 1)"


_CONTAINER = (
    r"^(?:[A-Za-z_]\w*\.)*(?:Sequence|MutableSequence|Collection|Iterable|Iterator|list|List|tuple|Tuple"
    r"|set|Set|frozenset|FrozenSet|deque|Deque)\[\s*([^\[\],]+(?:\[[^\[\]]*\])?)\s*(?:,\s*\.\.\.\s*)?\]"
    r"(?:\s*\|\s*None)?$"
)


class CudaAnnotationElement(Function):
    """What a container annotation holds, as an annotation of its own: `Sequence[RewriteStage]`
    -> `RewriteStage`, `tuple[cp.ndarray, ...]` -> `cp.ndarray`, empty for any other annotation
    (a `dict`, a `tuple` of different types)."""

    text: str

    expression = f"regexp_extract({_peeled('text')}, '{_CONTAINER}', 1)"


def _without_brackets(text: str) -> str:
    """`text` with the contents of up to three levels of brackets removed."""
    for _ in range(3):
        text = f"regexp_replace({text}, '{_INNER}', '', 'g')"
    return text


class CudaAnnotationMixed(Function):
    """Whether an annotation unites two kinds of value (`cp.ndarray | np.ndarray`), once a `None`
    alternative and everything inside brackets are set aside."""

    text: str

    expression = (
        f"contains(regexp_replace({_without_brackets('text')}, '{_NONE}', '', 'g'), '|') "
        f"OR starts_with(trim(text, {_QUOTES}), 'Union[')"
    )


class CudaDeviceTypes(CudaRelation):
    """The dotted names of the types that are device memory, by framework."""

    query = """
        SELECT * FROM (VALUES
          ('cupy.ndarray', 'cupy'), ('cupy.typing.NDArray', 'cupy'),
          ('cupy._core.core.ndarray', 'cupy'), ('cupy.core.core.ndarray', 'cupy'),
          ('numba.cuda.cudadrv.devicearray.DeviceNDArray', 'numba'),
          ('numba.cuda.devicearray.DeviceNDArray', 'numba'),
          ('numba.cuda.DeviceNDArray', 'numba'),
          ('numba.cuda.cudadrv.devicearray.DeviceRecord', 'numba'),
          ('cuda.core.Buffer', 'cuda.core'), ('cuda.core.experimental.Buffer', 'cuda.core'),
          ('patos.cuda.scalars.Vector', 'patos'), ('patos.cuda.scalars.Matrix', 'patos')
        ) t(qualified, framework)
    """


class CudaAnnotations(CudaRelation):
    """Every annotated parameter, return, class field and local, with its text: `site` is
    `parameter`, `return`, `field` (in a class body) or `local` (a variable or `self.x`),
    `owner_id` the function or class that holds it and `name` what it annotates."""

    stored = True
    query = r"""
        SELECT f.fact_id, f.path, 'parameter' AS site, f.node_id AS owner_id,
               f.scope."class" AS class_id, p.name, p.type AS annotation,
               false AS is_attribute, f.start_line, f.start_column
        FROM (SELECT *, unnest(parameters) AS p FROM cuda_nodes()
              WHERE type = 'function_definition') f
        WHERE p.type <> ''
        UNION ALL
        SELECT f.fact_id, f.path, 'return', f.node_id, f.scope."class", '', t.name,
               false, t.start_line, t.start_column
        FROM cuda_nodes() f JOIN cuda_nodes() t ON t.fact_id = f.fact_id AND t.parent_id = f.node_id
          AND t.type = 'type'
        WHERE f.type = 'function_definition'
        UNION ALL
        SELECT a.fact_id, a.path,
               CASE WHEN a.scope."function" IS NULL AND a.scope."class" IS NOT NULL
                    THEN 'field' ELSE 'local' END,
               coalesce(a.scope."function", a.scope."class"), a.scope."class",
               regexp_replace(l.name, '^self\.', ''), t.name, l.type = 'attribute',
               a.start_line, a.start_column
        FROM cuda_nodes() a
        JOIN cuda_nodes() l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
          AND l.sibling_index = 0
        JOIN cuda_nodes() t ON t.fact_id = a.fact_id AND t.parent_id = a.node_id AND t.type = 'type'
        WHERE a.type = 'assignment'
    """


class CudaDeviceClasses(CudaRelation):
    """Classes of the project that stand for device memory: one that exposes a
    `__cuda_array_interface__`, and a protocol that declares the host copy (`get`, `cpu`,
    `copy_to_host`) or the interface."""

    stored = True
    query = """
        SELECT DISTINCT c.key
        FROM cuda_definitions() c
        JOIN cuda_definitions() m ON m.fact_id = c.fact_id AND m.class_id = c.node_id
        JOIN cuda_nodes() k ON k.fact_id = c.fact_id AND k.node_id = c.node_id
        WHERE c.kind = 'class' AND m.kind = 'function'
          AND (m.name = '__cuda_array_interface__'
               OR (m.name IN ('get', 'cpu', 'copy_to_host')
                   AND list_contains(list_transform(k.parameters, lambda b: b.name), 'Protocol')))
    """


class CudaArrayClasses(CudaRelation):
    """Classes of the project that stand for an array of unknown placement: a protocol with
    `__getitem__` and an array's shape or dtype, satisfied by numpy and device arrays alike. A
    value of such a type is device memory only where a device value flows into it."""

    stored = True
    query = """
        SELECT DISTINCT c.key
        FROM cuda_definitions() c
        JOIN cuda_nodes() k ON k.fact_id = c.fact_id AND k.node_id = c.node_id
        WHERE c.kind = 'class'
          AND list_contains(list_transform(k.parameters, lambda b: b.name), 'Protocol')
          AND EXISTS (SELECT 1 FROM cuda_definitions() m WHERE m.fact_id = c.fact_id
                        AND m.class_id = c.node_id AND m.name = '__getitem__')
          AND EXISTS (SELECT 1 FROM cuda_definitions() m WHERE m.fact_id = c.fact_id
                        AND m.class_id = c.node_id
                        AND m.name IN ('shape', 'dtype', 'nbytes', 'astype'))
    """


class CudaChains(CudaRelation):
    """The dotted names annotations and class bases write, each resolved from the file that
    writes it: `qualified` is the name seen from outside (`cupy.ndarray`), `definition` the key of
    the class or function of the project it names."""

    stored = True
    query = """
        WITH written AS (
          SELECT DISTINCT fact_id, cuda_annotation_head(annotation) AS chain
          FROM cuda_annotations()
          UNION
          SELECT fact_id, regexp_extract(annotation, '\\[\\s*([A-Za-z_][A-Za-z0-9_.]*)', 1)
          FROM cuda_annotations() WHERE contains(annotation, '[')
          UNION
          SELECT fact_id, cuda_annotation_head(cuda_annotation_element(annotation))
          FROM cuda_annotations() WHERE cuda_annotation_element(annotation) <> ''
          UNION
          SELECT a.fact_id, r.name
          FROM cuda_nodes() a
          JOIN cuda_nodes() r ON r.fact_id = a.fact_id AND r.parent_id = a.node_id
            AND r.sibling_index = 2 AND r.type IN ('identifier', 'attribute')
          WHERE a.type = 'assignment' AND (
                (a.scope."function" IS NULL AND a.scope."class" IS NULL)
                OR EXISTS (SELECT 1 FROM cuda_nodes() l
                           WHERE l.fact_id = a.fact_id AND l.parent_id = a.node_id
                             AND l.sibling_index = 0 AND l.type = 'attribute'))
            AND regexp_matches(r.name, '^[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)*$')
          UNION
          SELECT fact_id, trim(element)
          FROM (SELECT fact_id, unnest(string_split(regexp_extract(annotation,
                  '^(?:[A-Za-z_]\\w*\\.)*[Tt]uple\\[(.*)\\]$', 1), ',')) AS element
                FROM cuda_annotations() WHERE site = 'return')
          WHERE regexp_matches(trim(element), '^[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)*$')
          UNION
          SELECT fact_id, written FROM cuda_decorators()
          UNION
          SELECT fact_id, callee_text FROM cuda_calls()
          WHERE regexp_matches(callee_text, '^[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)*$')
          UNION
          SELECT c.fact_id, o.name
          FROM cuda_calls() c
          JOIN cuda_nodes() o ON o.fact_id = c.fact_id AND o.parent_id = c.callee_id
            AND o.sibling_index = 0
          WHERE c.callee_type = 'subscript' AND o.type IN ('identifier', 'attribute')
          UNION
          SELECT a.fact_id, regexp_replace(l.name, '\\.[^.]+$', '')
          FROM cuda_nodes() a
          JOIN cuda_nodes() l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
            AND l.sibling_index = 0 AND l.type = 'attribute'
          WHERE a.type = 'assignment' AND regexp_matches(l.name, '^[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)+$')
            AND NOT regexp_matches(l.name, '^(self|cls)\\.')
          UNION
          SELECT fact_id, regexp_replace(name, '\\.[^.]+$', '')
          FROM cuda_nodes()
          WHERE type = 'attribute'
            AND regexp_matches(name, '^[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)*\\.(compiled|signatures|nopython_signatures|is_compiled)$')
          UNION
          SELECT fact_id, b.name
          FROM (SELECT fact_id, unnest(parameters) AS b FROM cuda_nodes()
                WHERE type = 'class_definition')
          WHERE b.type = 'extends'
        ), parts AS (
          SELECT fact_id, chain, split_part(chain, '.', 1) AS first,
                 CASE WHEN contains(chain, '.') THEN substr(chain, length(split_part(chain, '.', 1)) + 2) END
                   AS rest
          FROM written WHERE chain <> ''
        ), landed AS (
          SELECT p.fact_id, p.chain, p.rest,
                 coalesce(n.qualified, CASE WHEN d.key IS NOT NULL THEN d.qualified END,
                          'builtins.' || p.first) AS qualified,
                 coalesce(n.definition, d.key) AS definition, n.seen
          FROM parts p
          LEFT JOIN cuda_names() n ON n.fact_id = p.fact_id AND n.bound = p.first
          LEFT JOIN cuda_definitions() d ON d.fact_id = p.fact_id AND d.qualname = p.first
            AND d.kind IN ('class', 'function')
        )
        SELECT l.fact_id, l.chain,
               CASE WHEN l.rest IS NULL THEN l.qualified ELSE l.qualified || '.' || l.rest END
                 AS qualified,
               CASE WHEN l.rest IS NULL THEN l.definition
                    ELSE (SELECT min(d.key) FROM cuda_definitions() d
                          WHERE d.qualified = l.qualified || '.' || l.rest) END AS definition,
               CASE WHEN l.rest IS NULL THEN l.seen
                    ELSE list_transform(l.seen, lambda s: s || '.' || l.rest) END AS seen
        FROM landed l
    """


class CudaCallTargets(CudaRelation):
    """The definition a call reaches, as the key of a function or class of the project: by the
    name the call writes (through the file's imports and the re-exports behind them) and by the
    receiver's type (`call_calls` resolves `c.go()` when `c` is annotated)."""

    stored = True
    query = """
        SELECT DISTINCT c.fact_id, c.node_id, k.key, k.kind, k.node_id AS definition_id,
               k.fact_id AS definition_fact
        FROM cuda_calls() c
        JOIN cuda_chains() h ON h.fact_id = c.fact_id AND h.chain = c.callee_text
        JOIN cuda_definitions() k ON k.key = h.definition AND k.kind IN ('function', 'class')
        UNION
        SELECT c.fact_id, c.node_id, k.key, k.kind, k.node_id, k.fact_id
        FROM cuda_calls() c
        JOIN cuda_definitions() k ON k.qualified = c.qualified_name
         AND k.kind IN ('function', 'class')
    """
