"""Rules over how a type's members are organized (mcmr's ALL-CLAS organization family)."""

from ...catalog import Example, Function, Relation, Rule


class ClassesOrganizationIsComment(Function):
    """Whether a source line is a whole-line comment (`#` or `//`)."""

    text: str

    expression = "regexp_matches(text, '^[ ' || chr(9) || ']*(#|//)')"


class ClassesOrganizationRanked(Relation):
    """Every method of every class with its place in the source (`declared_position`) and its
    place in the expected order (`expected_position`) within its region, and how many of its
    region's methods sit elsewhere than expected.

    A method's key is its decorators as written, each with its `@`, then its name:
    `@classmethod from_x`, `@property size`, `@field_validator("a") @classmethod check`, `read`.
    Keys compare byte by byte, so decorated methods come first, then the dunders (`__init__`
    included), then the `_private` names, then the public ones. A property's `@x.setter` and
    `@x.deleter` take the key of the property they extend and an `@overload` stub takes the key of
    the implementation it overloads, so each stays beside its partner in source order."""

    query = r"""
    WITH decorated AS (
      SELECT method_id, list(value ORDER BY ordinal) AS items,
             '@' || string_agg(regexp_replace(value, '\s+', ' ', 'g'), ' @' ORDER BY ordinal)
               AS decorators
      FROM class_method_decorators GROUP BY method_id
    ), own AS (
      SELECT m.class_id, m.region, m.ordinal, m.name, c.name AS class_name,
             c.path, c.start_line, c.start_column,
             m.start_line AS method_start_line, m.end_line AS method_end_line,
             coalesce(d.decorators || ' ', '') || m.name AS own_key,
             coalesce(regexp_extract(d.items[-1], '^(\w+)\.(setter|deleter|getter)$', 1), '')
               AS property_name,
             coalesce(len(list_filter(d.items, lambda x: regexp_matches(x, '(^|\.)overload$'))) > 0,
                      false) AS is_overload
      FROM class_methods m JOIN class_classes c USING (class_id)
      LEFT JOIN decorated d USING (method_id)
    ), keyed AS (
      SELECT o.*, coalesce(
        CASE WHEN o.property_name <> '' THEN
          (SELECT arg_max(a.own_key, a.ordinal) FROM own a
           WHERE a.class_id = o.class_id AND a.region = o.region AND a.name = o.property_name
             AND a.ordinal < o.ordinal AND a.property_name = '') END,
        CASE WHEN o.is_overload THEN
          (SELECT arg_min(a.own_key, a.ordinal) FROM own a
           WHERE a.class_id = o.class_id AND a.region = o.region AND a.name = o.name
             AND a.ordinal > o.ordinal AND NOT a.is_overload) END,
        o.own_key) AS sort_key
      FROM own o
    ), positioned AS (
      SELECT *, row_number() OVER (PARTITION BY class_id, region ORDER BY ordinal)
                  AS declared_position,
             row_number() OVER (PARTITION BY class_id, region ORDER BY sort_key, ordinal)
                  AS expected_position
      FROM keyed
    )
    SELECT *, count(*) OVER region AS declared_count,
           count(*) FILTER (WHERE declared_position <> expected_position) OVER region
             AS moved_count,
           min(declared_position) FILTER (WHERE declared_position <> expected_position)
             OVER region AS first_moved
    FROM positioned WINDOW region AS (PARTITION BY class_id, region)
"""


class ClassMethodOrder(Rule, id="ALL-CLAS0001", applicability="review"):
    """Classes whose methods leave their expected source order within a region: the order of
    their keys, a method's decorators as written and then its name, compared byte by byte. In
    ASCII that is the decorated methods first (`@classmethod`, `@property`, `@staticmethod`, a
    validator ...), then the dunders (`__init__` among them), then the `_private` names, then the
    public ones. A property's `@x.setter` stays below the property, and the stubs of an
    `@overload` stay above the implementation they overload. There is no setting: the order is
    the same in every project."""

    query = """
        SELECT d.path, d.start_line, d.start_column, d.class_id, d.region, d.class_name, d.name,
               d.moved_count, d.declared_count, e.name AS expected_name
        FROM classes_organization_ranked() d
        JOIN classes_organization_ranked() e
          ON e.class_id = d.class_id AND e.region = d.region
         AND e.expected_position = d.first_moved
        WHERE d.declared_position = d.first_moved
        QUALIFY d.region = min(d.region) OVER (PARTITION BY d.class_id)
    """
    message = (
        "`{class_name}` declares {moved_count} of its {declared_count} members out of order, "
        "and `{expected_name}` belongs where `{name}` sits"
    )

    # Python only. The region's methods are rewritten in their expected order in one edit, each
    # with its decorators and the comment lines directly above it, a blank line between them: a
    # class is left alone when anything else sits between its methods (fields, nested classes,
    # a comment set apart by a blank line), or when a decorator names another method of it
    # other than a property's `setter`, `deleter` and `getter` (`@dispatch.register` must stay
    # below `dispatch`).
    fix = r"""
        WITH flagged AS (
          SELECT DISTINCT finding_key, class_id, region FROM finding
          WHERE regexp_matches(path, '\.pyi?$')
        ), members AS (
          SELECT f.finding_key, r.path, r.name, r.expected_position,
                 r.method_start_line AS start_line, r.method_end_line AS end_line
          FROM flagged f
          JOIN classes_organization_ranked() r
            ON r.class_id = f.class_id AND r.region = f.region
        ), units AS (
          SELECT *, least(stop + 1, start_line) AS first_line FROM (
            SELECT m.*, coalesce(max(b.line), 0) AS stop
            FROM members m LEFT JOIN source_lines() b ON b.path = m.path
             AND b.line < m.start_line AND NOT classes_organization_is_comment(b.text)
            GROUP BY ALL)
        ), pieces AS (
          SELECT u.finding_key, u.path, u.name, u.expected_position, u.first_line, u.end_line,
                 string_agg(rtrim(l.text, chr(13)), chr(10) ORDER BY l.line) AS piece,
                 count(*) FILTER (WHERE trimmed(l.text) <> '') AS code_lines
          FROM units u JOIN source_lines() l
            ON l.path = u.path AND l.line BETWEEN u.first_line AND u.end_line
          GROUP BY ALL
        ), regions AS (
          SELECT finding_key, path, min(first_line) AS first_line, max(end_line) AS last_line,
                 string_agg(piece, chr(10) || chr(10) ORDER BY expected_position) AS block,
                 sum(code_lines) AS code_lines, list(name) AS names
          FROM pieces GROUP BY finding_key, path
        )
        SELECT r.finding_key, r.path, r.first_line AS start_line, 0 AS start_column,
               r.last_line + 1 AS end_line, 0 AS end_column, r.block || chr(10) AS text
        FROM regions r
        WHERE r.code_lines = (
                SELECT count(*) FROM source_lines() l
                WHERE l.path = r.path AND l.line BETWEEN r.first_line AND r.last_line
                  AND trimmed(l.text) <> '')
          AND NOT list_has_any(
            list_transform(
              list_filter(regexp_extract_all(r.block, '(?m)^[ \t]*@(\w+\.\w+)', 1),
                          lambda x: NOT regexp_matches(x, '\.(setter|deleter|getter)$')),
              lambda x: split_part(x, '.', 1)),
            r.names)
    """
    examples = (
        Example(
            bad=(
                "class Gauge:\n    def read(self):\n        return 1\n\n"
                "    def _reset(self):\n        self.value = 0\n\n"
                "    def __init__(self):\n        self.value = 0\n\n"
                "    @property\n    def level(self):\n        return self.value\n\n"
                "    @level.setter\n    def level(self, value):\n        self.value = value\n"
            ),
            good=(
                "class Gauge:\n    @property\n    def level(self):\n        return self.value\n\n"
                "    @level.setter\n    def level(self, value):\n        self.value = value\n\n"
                "    def __init__(self):\n        self.value = 0\n\n"
                "    def _reset(self):\n        self.value = 0\n\n"
                "    def read(self):\n        return 1\n"
            ),
        ),
        Example(
            bad=(
                "from typing import overload\n\n\nclass Box:\n"
                "    def put(self, item):\n        return item\n\n"
                "    @staticmethod\n    def empty():\n        return Box()\n\n"
                "    def take(self, key):\n        return key\n\n"
                "    @overload\n    @classmethod\n    def of(cls, item: int) -> int: ...\n\n"
                "    @overload\n    @classmethod\n    def of(cls, item: str) -> str: ...\n\n"
                "    @classmethod\n    def of(cls, item):\n        return item\n"
            ),
            good=(
                "from typing import overload\n\n\nclass Box:\n"
                "    @overload\n    @classmethod\n    def of(cls, item: int) -> int: ...\n\n"
                "    @overload\n    @classmethod\n    def of(cls, item: str) -> str: ...\n\n"
                "    @classmethod\n    def of(cls, item):\n        return item\n\n"
                "    @staticmethod\n    def empty():\n        return Box()\n\n"
                "    def put(self, item):\n        return item\n\n"
                "    def take(self, key):\n        return key\n"
            ),
        ),
    )
