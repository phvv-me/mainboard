"""Rules over how data models are declared and built (mcmr's PY-MODE family)."""

from ....catalog import Example, Rule

_MODELS_PACKAGE = {"pkg/__init__.py": "", "pkg/models/__init__.py": ""}
_PAIR = """from pydantic import BaseModel


class Left(BaseModel):
    size: int


class Right(BaseModel):
    size: int
"""
_POINT = """from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int
    z: int
    w: int
"""
_DUMP = """from shapes import Point


def dump(p: Point):
    return {"x": p.x, "y": p.y, "z": p.z, "w": p.w}
"""
_DUMPED = 'p.model_dump(include={"x", "y", "z", "w"})'
_UNTYPED = (
    _DUMP.replace("from shapes import Point\n\n\n", "")
    .replace("p: Point", "r")
    .replace("p.", "r.")
)
_DATACLASS = """from dataclasses import dataclass


@dataclass
class Row:
    x: int
    y: int
    z: int
    w: int


def dump(p: Row):
    return {"x": p.x, "y": p.y, "z": p.z, "w": p.w}
"""
_SPLAT = _DUMP.replace("}\n", ", **EXTRA}\n")
_REASSIGNED = _DUMP.replace("    return", "    p = p.copy()\n    return")
_NESTED = """from pydantic import BaseModel


class Inner(BaseModel):
    size: int


class Outer(BaseModel):
    x: int
    y: int
    z: int
    inner: Inner

    def as_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z, "inner": self.inner}
"""
_SERIALIZED = (
    _POINT.replace("BaseModel\n", "BaseModel, field_serializer\n", 1)
    + """
    @field_serializer("w")
    def scaled(self, value):
        return value * 2

    def as_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z, "w": self.w}
"""
)
_INHERITED = """from pydantic import BaseModel


class PointModel(BaseModel):
    x: int
    y: int


class Point(PointModel):
    z: int
    w: int

    def as_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z, "w": self.w}
"""
_PAIR_WITH_THREE = (
    _PAIR
    + """

class Top(BaseModel):
    size: int
"""
)
_PAIR_WITH_HELPER = """from pydantic import BaseModel


class Left(BaseModel):
    size: int


class Gauge:
    width = 3
"""
_PAIR_READING = """from pydantic import BaseModel

LIMIT = 4


class Left(BaseModel):
    size: int


class Right(BaseModel):
    size: int = LIMIT
"""
_RIGHT_USER = (
    "from .models.pair import Right\n\n\ndef make() -> Right:\n    return Right(size=1)\n"
)
_BOTH_USER = """from .models.pair import Right, Top


def both() -> tuple:
    return Right, Top
"""
_GAUGE_USER = "from .models.pair import Gauge\n\n\ndef read() -> int:\n    return Gauge.width\n"


class SharedModelFileShape(Rule, id="PY-MODE0001", applicability="review"):
    """Shared `models` package files that do not hold exactly one model-shaped class."""

    # A file with a model states each of its other top-level classes as a finding of its own, so
    # that moving one is progress the fixer sees; a file with none is one finding.
    query = """
        WITH tops AS (
          SELECT f.path, f.top_level_class_count, s.name, c.start_line, c.start_column,
                 NOT s.is_model_configuration_base AND (s.foundation_base OR s.is_dataclass)
                   AS is_model
          FROM class_model_files f
          JOIN class_staged s ON s.path = f.path AND NOT s.nested
          JOIN class_defs c ON c.class_id = s.class_id
          WHERE NOT f.is_package_initializer
        ), first AS (
          SELECT path, min(start_line) FILTER (WHERE is_model) AS model_line FROM tops GROUP BY path
        )
        SELECT t.path, t.start_line, t.start_column, t.name, t.is_model,
               '`' || t.name || '` shares a models file that holds exactly one model class'
                 AS message
        FROM tops t JOIN first f USING (path)
        WHERE t.top_level_class_count > 1 AND t.start_line <> f.model_line
        UNION ALL
        SELECT path, 1, 0, NULL, NULL, '`' || path || '` is a models file with no model class'
        FROM class_model_files WHERE NOT is_package_initializer AND model_class_count = 0
    """
    # Each class beyond the first model moves into a module of its own (`move-definitions`):
    # another model beside the file, a class that is no model up to the package holding the
    # `models` folder, since the file's own folder would flag it again. A destination that exists
    # stays untouched. Left alone, with the reason, what the operation refuses (a class reading a
    # name of its module that stays, an importer binding the module as `import a as b`).
    fix = r"""
        WITH moved AS (
          SELECT finding_key, path, name,
                 module_path(if(is_model, regexp_replace(path, '/?[^/]*$', ''),
                                regexp_replace(path, '/?[^/]*/[^/]*$', '')), name) AS destination
          FROM finding WHERE name IS NOT NULL
        )
        SELECT m.finding_key, 'move-definitions' AS operation, r.*
        FROM moved m, move_row(m.path, [m.name], m.destination) r
        WHERE NOT EXISTS (SELECT 1 FROM core_file c WHERE c.path = m.destination)
    """
    examples = (
        Example(
            bad=_RIGHT_USER,
            good=_RIGHT_USER.replace("models.pair", "models.right"),
            path="pkg/use.py",
            context={**_MODELS_PACKAGE, "pkg/models/pair.py": _PAIR},
        ),
        Example(
            bad=_BOTH_USER,
            good=_BOTH_USER.replace(
                "from .models.pair import Right, Top",
                "from .models.right import Right\nfrom .models.top import Top",
            ),
            path="pkg/use.py",
            context={**_MODELS_PACKAGE, "pkg/models/pair.py": _PAIR_WITH_THREE},
        ),
        Example(
            bad=_GAUGE_USER,
            good=_GAUGE_USER.replace("models.pair", "gauge"),
            path="pkg/use.py",
            context={**_MODELS_PACKAGE, "pkg/models/pair.py": _PAIR_WITH_HELPER},
        ),
        Example(
            bad=_RIGHT_USER,
            good=_RIGHT_USER,
            path="pkg/use.py",
            context={**_MODELS_PACKAGE, "pkg/models/pair.py": _PAIR_READING},
        ),
    )


class ApprovedModelFoundation(Rule, id="PY-MODE0003", applicability="review"):
    """Direct Pydantic models that bypass an established approved house base."""

    approved_base: str = "patos.FrozenModel"

    query = """
        SELECT c.* FROM class_classes c JOIN class_facts f USING (fact_id)
        WHERE f.has_approved_model_foundation_policy AND c.directly_inherits_pydantic_base_model
          AND NOT c.inherits_approved_model_foundation
    """
    message = "`{name}` inherits Pydantic directly instead of the approved model foundation"
    examples = (
        Example(
            bad="""from patos import FrozenModel
from pydantic import BaseModel


class Config(FrozenModel):
    size: int


class User(BaseModel):
    name: str
""",
            good="""from patos import FrozenModel


class Config(FrozenModel):
    size: int


class User(FrozenModel):
    name: str
""",
        ),
    )
    # The `BaseModel` in the class header becomes the approved base, imported unless the file
    # binds that name already. The base is immutable (`approved_base`): a class that assigns to
    # its fields needs a person to pick the mutable one. A header that is not on one line, or
    # names the base twice, is left.
    fix = """
        WITH pydantic AS (
          SELECT path, name AS spelled FROM import_binding_facts
          WHERE module = 'pydantic' AND imported_name = 'BaseModel'
          UNION ALL
          SELECT path, name || '.BaseModel' FROM import_binding_facts
          WHERE module = 'pydantic' AND imported_name = 'pydantic'
        ), located AS (
          SELECT f.finding_key, f.path, f.start_line, p.spelled, h.text AS header,
                 strpos(h.text, '(') AS opened
          FROM finding f
          JOIN class_direct_bases b ON b.class_id = f.class_id
          JOIN pydantic p ON p.path = f.path AND p.spelled = b.value
          JOIN imports_code_lines() h ON h.path = f.path AND h.line = f.start_line
          WHERE regexp_matches(h.text, '^\\s*class\\s')
        ), placed AS (
          SELECT *, strlen(left(header,
                      opened + strpos(substring(header, opened + 1), spelled) - 1)) AS base_column
          FROM located
          WHERE opened > 0 AND len(string_split(substring(header, opened + 1), spelled)) = 2
        )
        SELECT finding_key, path, start_line, base_column AS start_column,
               start_line AS end_line, base_column + strlen(spelled) AS end_column,
               regexp_extract($approved_base, '[^.]+$') AS text,
               CASE WHEN NOT EXISTS (SELECT 1 FROM import_binding_facts i
                                     WHERE i.path = placed.path
                                       AND i.name = regexp_extract($approved_base, '[^.]+$'))
                    THEN 'from ' || regexp_replace($approved_base, '\\.[^.]+$', '')
                         || ' import ' || regexp_extract($approved_base, '[^.]+$') END
                 AS "import"
        FROM placed
    """


class ManualModelAttributeProjectionCount(Rule, id="PY-MODE0004", applicability="review"):
    """Structures that manually repeat fields from one model instance."""

    minimum_attributes: int = 4

    query = """
        WITH attributes AS (SELECT DISTINCT projection_id, value FROM class_projection_attributes),
        outputs AS (
          SELECT DISTINCT projection_id, replace(value, '-', '_') AS value
          FROM class_projection_output_keys),
        sizes AS (
          SELECT projection_id, count(*) AS attribute_count, count(o.value) AS shared_count
          FROM attributes a LEFT JOIN outputs o USING (projection_id, value)
          GROUP BY projection_id),
        written AS (
          SELECT projection_id, count(*) AS output_count FROM outputs GROUP BY projection_id),
        names AS (
          SELECT projection_id, string_agg(value, '`, `' ORDER BY ordinal) AS names
          FROM class_projection_attributes GROUP BY projection_id)
        SELECT p.*, n.names FROM class_projections p
        LEFT JOIN sizes s USING (projection_id)
        LEFT JOIN written w USING (projection_id)
        LEFT JOIN names n USING (projection_id)
        WHERE coalesce(s.attribute_count, 0) >= $minimum_attributes
          AND coalesce(s.attribute_count, 0) = coalesce(w.output_count, 0)
          AND coalesce(s.attribute_count, 0) = coalesce(s.shared_count, 0)
    """
    message = "projection from `{root}` repeats model attributes `{names}`"
    change = "dumped `{root}` with `model_dump(include=...)` in place of `{names}`"
    # A dict display whose every entry is `"field": root.field` becomes `root.model_dump(include=
    # {...})`, when the facts prove `root` a model: `self` in a method of a Pydantic model, or a
    # parameter annotated with the one model of that name, whose own fields (annotated scalars,
    # lists and dicts of scalars) include every key. The dump orders keys as the model declares
    # them, a change to review. Left alone: a call or list of pairs, a splat or a comment among
    # the entries, a key that differs from its attribute, a field that holds a model (the dump
    # turns it into a dict), an inherited field, a model with serializers, computed fields or
    # `exclude`, and a parameter reassigned before the dict.
    fix = r"""
        WITH boxes AS (
          SELECT f.finding_key, f.path, p.file_id, f.root, c.node_id AS box,
                 (SELECT count(*) FROM core_node k
                  WHERE k.file_id = p.file_id AND k.parent_id = c.node_id
                    AND k.type NOT IN ('{', '}')) AS entries
          FROM finding f
          JOIN core_file p ON p.path = f.path
          JOIN core_node c ON c.file_id = p.file_id AND c.type = 'dictionary' AND c.start_line = f.start_line AND c.start_column = f.start_column
        ), exact AS (
          SELECT b.finding_key, b.box, list(kc.name ORDER BY pr.sibling_index) AS keys,
                 list(pr.node_id ORDER BY pr.sibling_index) AS pairs
          FROM boxes b
          JOIN core_node pr ON pr.file_id = b.file_id AND pr.parent_id = b.box
            AND pr.type = 'pair' AND pr.children_count = 2
          JOIN core_node k ON k.file_id = b.file_id AND k.parent_id = pr.node_id
            AND k.sibling_index = 0 AND k.type = 'string' AND k.children_count = 1
            AND NOT regexp_matches(k.name, '^[a-zA-Z]*[bBfF]')
          JOIN core_node kc ON kc.file_id = b.file_id AND kc.parent_id = k.node_id
            AND kc.type = 'string_content'
          JOIN core_node v ON v.file_id = b.file_id AND v.parent_id = pr.node_id
            AND v.sibling_index = 1 AND v.type = 'attribute' AND v.name = b.root || '.' || kc.name
          GROUP BY b.finding_key, b.box, b.entries HAVING count(*) = b.entries
        ), homes AS (
          SELECT b.*, h.node_id AS fn, h.parent_id AS holder
          FROM boxes b
          JOIN core_node h ON h.file_id = b.file_id AND h.type = 'function_definition'
            AND b.box BETWEEN h.node_id AND h.node_id + h.descendant_count
          QUALIFY row_number() OVER (PARTITION BY b.finding_key ORDER BY h.node_id DESC) = 1
        ), typed AS (
          SELECT h.finding_key, h.path, h.file_id, h.box, h.root, c.record_id AS model,
                 c.path AS model_path, c.class_node
          FROM homes h
          JOIN core_node w ON w.file_id = h.file_id AND w.node_id = h.holder
          JOIN core_node bk ON bk.file_id = h.file_id AND bk.type = 'block'
            AND bk.node_id = if(w.type = 'decorated_definition', w.parent_id, w.node_id)
          JOIN pydantic_model_candidates c ON c.path = h.path AND c.is_pydantic_model
            AND c.class_node = bk.parent_id
          WHERE h.root = 'self'
          UNION ALL
          SELECT h.finding_key, h.path, h.file_id, h.box, h.root, c.record_id, c.path, c.class_node
          FROM homes h
          JOIN core_node ps ON ps.file_id = h.file_id AND ps.parent_id = h.fn
            AND ps.type = 'parameters'
          JOIN core_node tp ON tp.file_id = h.file_id AND tp.parent_id = ps.node_id
            AND tp.type = 'typed_parameter'
          JOIN core_node ro ON ro.file_id = h.file_id AND ro.parent_id = tp.node_id
            AND ro.sibling_index = 0 AND ro.name = h.root
          JOIN core_node an ON an.file_id = h.file_id AND an.parent_id = tp.node_id
            AND an.type = 'type' AND regexp_matches(an.name, '^[A-Za-z_][A-Za-z0-9_]*$')
          JOIN pydantic_model_candidates c ON c.name = an.name AND c.is_pydantic_model
          WHERE h.root <> 'self'
            AND (SELECT count(*) FROM pydantic_model_candidates d
                 WHERE d.name = an.name AND d.is_pydantic_model) = 1
            AND NOT EXISTS (SELECT 1 FROM core_node a
                            WHERE a.file_id = h.file_id AND a.name = h.root
                              AND a.node_id BETWEEN h.fn AND h.box AND a.type = 'identifier'
                              AND a.parent_id IN (SELECT n.node_id FROM core_node n
                                                  WHERE n.file_id = h.file_id
                                                    AND n.type IN ('assignment', 'augmented_assignment')
                                                    AND n.node_id BETWEEN h.fn AND h.box)
                              AND a.sibling_index = 0)
        ), served AS (
          SELECT t.finding_key, t.path, t.box, t.root, e.keys, e.pairs
          FROM typed t JOIN exact e USING (finding_key, box)
          WHERE (SELECT count(*) FROM pydantic_model_models_fields f
                 WHERE f.parent_id = t.model AND list_contains(e.keys, f.name)
                   AND regexp_replace(f.annotation, '\b(int|str|float|bool|bytes|None|list|dict|set|tuple|frozenset|Optional|Path|pathlib|datetime|date|time|UUID|uuid|Decimal|decimal)\b|[\s\[\],|.]', '', 'g') = '')
                = len(e.keys)
            AND NOT EXISTS (
              SELECT 1 FROM core_file q
              JOIN core_node cn ON cn.file_id = q.file_id AND cn.node_id = t.class_node
              JOIN core_node n ON n.file_id = q.file_id AND n.type = 'identifier'
                AND n.node_id BETWEEN cn.node_id AND cn.node_id + cn.descendant_count
                AND regexp_matches(n.name, 'serializer|computed_field|^exclude$')
              WHERE q.path = t.model_path)
        )
        SELECT s.finding_key, 'rewrite-node' AS operation, s.path, s.box AS node,
               MAP {'PAIRS': s.pairs} AS captures,
               s.root || '.model_dump(include={' || array_to_string(
                 list_transform(s.keys, lambda k: '''' || k || ''''), ', ') || '})' AS template
        FROM served s
    """
    examples = (
        Example(
            bad=(
                """from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int
    z: int
    w: int

    def as_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z, "w": self.w}
"""
            ),
            good=(
                """from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int
    z: int
    w: int

    def as_dict(self):
        return self.model_dump(include={"x", "y", "z", "w"})
"""
            ),
        ),
        Example(
            bad=_DUMP,
            good=_DUMP.replace('{"x": p.x, "y": p.y, "z": p.z, "w": p.w}', _DUMPED),
            context={"shapes.py": _POINT},
        ),
        Example(bad=_UNTYPED, good=_UNTYPED),
        Example(bad=_DATACLASS, good=_DATACLASS),
        Example(bad=_SPLAT, good=_SPLAT, context={"shapes.py": _POINT}),
        Example(bad=_REASSIGNED, good=_REASSIGNED, context={"shapes.py": _POINT}),
        Example(bad=_NESTED, good=_NESTED),
        Example(bad=_SERIALIZED, good=_SERIALIZED),
        Example(bad=_INHERITED, good=_INHERITED),
    )


class StandardDataclassModel(Rule, id="PY-MODE0005"):
    """State-bearing standard dataclasses that should use the approved model foundation."""

    query = """
        SELECT * FROM class_classes
        WHERE is_dataclass AND field_count > 0 AND NOT has_ordinary_behavior
    """
    message = (
        "`{name}` is a standard dataclass with {field_count} fields instead of an approved "
        "Pydantic model"
    )


class EmptyDeclarativeModel(Rule, id="PY-MODE0006"):
    """Declarative models that own no field. A class body that assigns a call to a lowercase name
    declares its fields in a column DSL (`name = sql.PK(str)`, and so do the classes derived from
    it), a model naming a `metaclass` is the base that DSL's models derive from, and a registry
    implementation (`name = "leech"`) is keyed, not a record; so is a class an `__init_subclass__`
    hook of its own or of an ancestor registers on definition (a rule, a relation), keyed by
    the class itself."""

    query = r"""
        SELECT * FROM class_classes c
        WHERE is_declarative_model AND field_count = 0 AND NOT has_inherited_fields
          AND NOT is_protocol
          AND NOT has_explicit_registry_name
          AND NOT EXISTS (SELECT 1 FROM class_methods m
                          WHERE m.name IN ('__init_subclass__', '__pydantic_init_subclass__')
                            AND (m.class_id = c.class_id
                                 OR m.class_id IN (SELECT ancestor FROM class_lineage l
                                                   WHERE l.class_id = c.class_id)))
          AND NOT EXISTS (SELECT 1 FROM class_defs d
                          JOIN class_body_assigns a ON a.fact_id = d.fact_id
                            AND a.class_node = d.class_node
                          WHERE (d.class_id = c.class_id
                                 OR d.class_id IN (SELECT ancestor FROM class_lineage l
                                                   WHERE l.class_id = c.class_id))
                            AND a.annotation IS NULL
                            AND a.value_type = 'call' AND regexp_matches(a.target_name, '\p{Ll}'))
          AND NOT EXISTS (SELECT 1 FROM class_class_keywords k
                          WHERE k.class_id = c.class_id AND starts_with(k.value, 'metaclass='))
          AND NOT EXISTS (SELECT 1 FROM class_direct_bases b WHERE b.class_id = c.class_id
                          AND regexp_replace(b.value, '^.*\.', '', 'g')
                              IN ('ABC', 'Protocol', 'ABCMeta'))
          AND NOT EXISTS (SELECT 1 FROM class_methods m
                          JOIN class_method_decorators d USING (method_id)
                          WHERE m.class_id = c.class_id
                            AND regexp_replace(d.value, '^.*\.', '', 'g')
                                IN ('abstractmethod', 'abstractproperty'))
    """
    message = "`{name}` is a model with no fields"
