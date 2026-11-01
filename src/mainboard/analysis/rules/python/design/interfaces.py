"""Rules over what an interface promises (mcmr's PY-INTE family)."""

from ....catalog import Example, Rule


class ConcreteIsinstanceCapability(Rule, id="PY-INTE0001", applicability="review"):
    """Concrete runtime checks that stand in for a capability.

    A check is an `isinstance(subject, type)` call of two arguments, but for those in an `elif`
    condition. A check in the condition of an `if` is guarded by what the branch's block does with
    the subject: reading an attribute or calling a method of it, indexing it, testing membership
    in it, doing arithmetic on it, looping over it, or passing it to `len`, `iter`, `sorted` and
    the other reading builtins. A subject that is no plain name guards nothing. A tuple of
    types is read member by member; `str` and `bool` stay concrete, and so do JSON, TOML and wire
    boundaries, which a project names in the rule's `exclude`."""

    reading: tuple[str, ...] = (
        "len",
        "iter",
        "reversed",
        "sorted",
        "enumerate",
        "sum",
        "min",
        "max",
        "any",
        "all",
        "next",
    )

    query = """
        WITH sites AS (
          SELECT c.file_id, c.node_id AS call_node, c.start_line, c.start_column,
                 CASE WHEN s.type = 'identifier' THEN s.name ELSE '' END AS subject,
                 t.node_id AS type_node, t.type AS type_kind, t.name AS type_name, g.block_node
          FROM core_node c
          JOIN core_node a ON a.file_id = c.file_id AND a.parent_id = c.node_id
            AND a.type = 'argument_list' AND a.children_count = 4
          JOIN core_node s ON s.file_id = c.file_id AND s.parent_id = a.node_id
            AND s.sibling_index = 1
          JOIN core_node t ON t.file_id = c.file_id AND t.parent_id = a.node_id
            AND t.sibling_index = 2
          LEFT JOIN LATERAL (
            SELECT b.node_id AS block_node
            FROM core_node i
            JOIN core_node k ON k.file_id = i.file_id AND k.parent_id = i.node_id
              AND k.sibling_index = 1
            JOIN core_node b ON b.file_id = i.file_id AND b.parent_id = i.node_id
              AND b.type = 'block'
            WHERE i.file_id = c.file_id AND i.type = 'if_statement'
              AND c.node_id BETWEEN k.node_id AND k.node_id + k.descendant_count
            ORDER BY i.node_id DESC LIMIT 1) g ON true
          WHERE c.type = 'call' AND c.name = 'isinstance'
            AND NOT EXISTS (
              SELECT 1 FROM core_node e
              JOIN core_node k ON k.file_id = e.file_id AND k.parent_id = e.node_id
                AND k.sibling_index = 1
              WHERE e.file_id = c.file_id AND e.type = 'elif_clause'
                AND c.node_id BETWEEN k.node_id AND k.node_id + k.descendant_count)
        ), named AS (
          SELECT * EXCLUDE (type_node, type_kind, type_name), type_name AS concrete_type
          FROM sites WHERE type_kind IN ('identifier', 'attribute')
          UNION ALL
          SELECT s.* EXCLUDE (type_node, type_kind, type_name), e.name AS concrete_type
          FROM sites s
          JOIN core_node e ON e.file_id = s.file_id AND e.parent_id = s.type_node
            AND e.type IN ('identifier', 'attribute')
          WHERE s.type_kind = 'tuple'
        ), guarded AS (
          SELECT DISTINCT s.file_id, s.call_node
          FROM sites s
          JOIN core_node b ON b.file_id = s.file_id AND b.node_id = s.block_node,
          LATERAL (
            SELECT n.node_id FROM core_node n
            WHERE n.file_id = s.file_id AND n.type = 'attribute'
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
              AND starts_with(n.name, s.subject || '.')
              AND regexp_matches(substr(n.name, length(s.subject) + 2), '^\\w+$')
            UNION ALL
            SELECT n.node_id FROM core_node n
            JOIN core_node v ON v.file_id = n.file_id AND v.parent_id = n.node_id
              AND v.sibling_index = 0
            WHERE n.file_id = s.file_id AND n.type = 'subscript'
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
              AND v.type IN ('identifier', 'attribute') AND v.name = s.subject
            UNION ALL
            SELECT n.node_id FROM core_node n
            JOIN core_node o ON o.file_id = n.file_id AND o.parent_id = n.node_id
              AND o.sibling_index = 1 AND o.type IN ('in', 'not in')
            JOIN core_node r ON r.file_id = n.file_id AND r.parent_id = n.node_id
              AND r.sibling_index = 2
            WHERE n.file_id = s.file_id AND n.type = 'comparison_operator'
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
              AND r.type IN ('identifier', 'attribute') AND r.name = s.subject
            UNION ALL
            SELECT n.node_id FROM core_node n
            JOIN core_node o ON o.file_id = n.file_id AND o.parent_id = n.node_id
              AND o.sibling_index IN (0, 2)
            WHERE n.file_id = s.file_id AND n.type = 'binary_operator'
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
              AND o.type IN ('identifier', 'attribute') AND o.name = s.subject
            UNION ALL
            SELECT n.node_id FROM core_node n
            JOIN core_node l ON l.file_id = n.file_id AND l.parent_id = n.node_id
              AND l.type = 'argument_list'
            JOIN core_node x ON x.file_id = n.file_id AND x.parent_id = l.node_id
              AND x.type IN ('identifier', 'attribute') AND x.name = s.subject
            WHERE n.file_id = s.file_id AND n.type = 'call'
              AND list_contains($reading::VARCHAR[], n.name)
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
            UNION ALL
            SELECT n.node_id FROM core_node n
            JOIN core_node r ON r.file_id = n.file_id AND r.parent_id = n.node_id
              AND r.sibling_index = 3
            WHERE n.file_id = s.file_id AND n.type IN ('for_statement', 'for_in_clause')
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
              AND r.type IN ('identifier', 'attribute') AND r.name = s.subject
          ) op
          WHERE s.subject <> ''
        )
        SELECT f.path, n.start_line, n.start_column, n.subject,
               string_agg(DISTINCT n.concrete_type, ', ' ORDER BY n.concrete_type) AS types,
               string_agg(DISTINCT a.capability, ' or ' ORDER BY a.capability) AS capability
        FROM named n JOIN core_file f USING (file_id)
        JOIN (VALUES ('complex', 'numbers.Complex'), ('dict', 'collections.abc.Mapping'),
                     ('float', 'numbers.Real'), ('int', 'numbers.Integral'),
                     ('list', 'collections.abc.Sequence'), ('set', 'collections.abc.Set'),
                     ('tuple', 'collections.abc.Sequence')) a(concrete_type, capability)
          USING (concrete_type)
        WHERE f.language = 'python'
          AND EXISTS (SELECT 1 FROM guarded g
                      WHERE g.file_id = n.file_id AND g.call_node = n.call_node)
        GROUP BY f.path, n.call_node, n.start_line, n.start_column, n.subject
    """
    message = (
        "`{subject}` is checked against concrete `{types}` where its block needs only "
        "`{capability}`"
    )
    change = "checked `{subject}` against `{capability}` in place of `{types}`"
    # The concrete types of the check become their abstract capability (`list` and `tuple` the
    # `Sequence` of `collections.abc`, `int` the `Integral` of `numbers`), each imported by name;
    # a tuple keeps the types it has no capability for (`str`) and names a capability once. The
    # check then admits every type that registers as the capability (a `str` is a `Sequence`
    # too), the change to review. Left alone: a type argument that is no plain name or tuple of
    # names, one holding a comment, and a file where the capability's name is already something
    # else.
    fix = r"""
        WITH checks AS (
          SELECT f.finding_key, f.path, p.file_id, t.node_id AS argument, t.type AS kind
          FROM finding f
          JOIN core_file p ON p.path = f.path
          JOIN core_node c ON c.file_id = p.file_id AND c.type = 'call' AND c.name = 'isinstance'
            AND c.start_line = f.start_line AND c.start_column = f.start_column
          JOIN core_node a ON a.file_id = c.file_id AND a.parent_id = c.node_id
            AND a.type = 'argument_list'
          JOIN core_node t ON t.file_id = c.file_id AND t.parent_id = a.node_id
            AND t.sibling_index = 2
        ), shown AS (
          SELECT c.*, e.node_id AS member, e.sibling_index AS place, e.type AS member_kind,
                 k.abstract, k.home, coalesce(k.abstract, e.name) AS text
          FROM checks c
          JOIN core_node e ON e.file_id = c.file_id AND e.type NOT IN ('(', ')', ',')
            AND (CASE WHEN c.kind = 'tuple' THEN e.parent_id ELSE e.node_id END) = c.argument
          LEFT JOIN (VALUES ('complex', 'Complex', 'numbers'),
                            ('dict', 'Mapping', 'collections.abc'),
                            ('float', 'Real', 'numbers'), ('int', 'Integral', 'numbers'),
                            ('list', 'Sequence', 'collections.abc'),
                            ('set', 'Set', 'collections.abc'),
                            ('tuple', 'Sequence', 'collections.abc')) k(concrete, abstract, home)
            ON k.concrete = e.name AND e.type = 'identifier'
        ), settled AS (
          SELECT finding_key FROM shown s
          GROUP BY finding_key
          HAVING bool_and(member_kind IN ('identifier', 'attribute'))
             AND NOT bool_or(abstract IS NOT NULL AND (
               EXISTS (SELECT 1 FROM import_binding_facts i
                       WHERE i.path = s.path AND i.name = s.abstract
                         AND (i.module <> s.home OR i.imported_name <> s.abstract))
               OR (NOT EXISTS (SELECT 1 FROM import_binding_facts i
                               WHERE i.path = s.path AND i.name = s.abstract
                                 AND i.module = s.home AND i.imported_name = s.abstract)
                   AND EXISTS (SELECT 1 FROM core_node n
                               WHERE n.file_id = s.file_id AND n.type = 'identifier'
                                 AND n.name = s.abstract))))
        ), spelled AS (
          SELECT finding_key,
                 if(count(*) = 1, any_value(text), '(' || string_agg(text, ', ' ORDER BY first)
                                                   || ')') AS template
          FROM (SELECT finding_key, text, min(place) AS first FROM shown
                GROUP BY finding_key, text)
          GROUP BY finding_key
        ), members AS (
          SELECT finding_key, list(member ORDER BY place) AS members,
                 coalesce(list(DISTINCT 'from ' || home || ' import ' || abstract)
                          FILTER (abstract IS NOT NULL), []) AS imports
          FROM shown GROUP BY finding_key
        )
        SELECT c.finding_key, 'rewrite-node' AS operation, c.path, c.argument AS node,
               MAP {'MEMBERS': m.members} AS captures, s.template, m.imports
        FROM checks c JOIN settled USING (finding_key)
        JOIN spelled s USING (finding_key) JOIN members m USING (finding_key)
    """
    examples = (
        Example(
            bad=(
                """def first(items):
    if isinstance(items, (list, tuple)):
        return items[0]
    return items
"""
            ),
            good=(
                """from collections.abc import Sequence


def first(items):
    if isinstance(items, Sequence):
        return items[0]
    return items
"""
            ),
        ),
        Example(
            bad=(
                """def size(data):
    if isinstance(data, (dict, list, str)):
        return len(data)
    return 0
"""
            ),
            good=(
                """from collections.abc import Mapping, Sequence


def size(data):
    if isinstance(data, (Mapping, Sequence, str)):
        return len(data)
    return 0
"""
            ),
        ),
        Example(
            bad=(
                """def bump(count):
    if isinstance(count, int):
        return count + 1
    return count
"""
            ),
            good=(
                """from numbers import Integral


def bump(count):
    if isinstance(count, Integral):
        return count + 1
    return count
"""
            ),
        ),
        Example(
            bad=(
                """from typing import Sequence


def first(items: Sequence):
    if isinstance(items, (list, tuple)):
        return items[0]
    return items
"""
            ),
            good=(
                """from typing import Sequence


def first(items: Sequence):
    if isinstance(items, (list, tuple)):
        return items[0]
    return items
"""
            ),
        ),
        Example(
            bad=(
                """def first(items):
    if isinstance(items, (list, tuple, Row)):
        return items[0]
    return items
"""
            ),
            good=(
                """from collections.abc import Sequence


def first(items):
    if isinstance(items, (Sequence, Row)):
        return items[0]
    return items
"""
            ),
        ),
        Example(
            bad=(
                """def first(items):
    if isinstance(items, (list, make())):
        return items[0]
    return items
"""
            ),
            good=(
                """def first(items):
    if isinstance(items, (list, make())):
        return items[0]
    return items
"""
            ),
        ),
        Example(
            bad=(
                """def first(items):
    if isinstance(items, (list,  # rows
                          tuple)):
        return items[0]
    return items
"""
            ),
            good=(
                """def first(items):
    if isinstance(items, (list,  # rows
                          tuple)):
        return items[0]
    return items
"""
            ),
        ),
    )


class SingleImplementationAbstractBase(Rule, id="PY-INTE0002"):
    """Abstract bases that support only one repository implementation."""

    query = r"""
        SELECT c.*, s.value AS subclass FROM class_classes c
        LEFT JOIN (SELECT class_id, value FROM class_direct_subclasses
                   QUALIFY ordinal = min(ordinal) OVER (PARTITION BY class_id)) s USING (class_id)
        WHERE c.scope = 'module' AND NOT c.is_protocol AND c.descendant_count = 1 AND (
          c.class_id IN (SELECT class_id FROM class_direct_bases
                         WHERE regexp_replace(value, '^.*\.', '', 'g') IN ('ABC', 'ABCMeta'))
          OR c.class_id IN (
            SELECT m.class_id FROM class_method_decorators d JOIN class_methods m USING (method_id)
            WHERE regexp_replace(d.value, '^.*\.', '', 'g') IN (
              'abstractclassmethod', 'abstractmethod', 'abstractproperty',
              'abstractstaticmethod')))
    """
    message = "`{name}` is an abstract base with only `{subclass}` below it"


class ProtocolClass(Rule, id="PY-INTE0003"):
    """Classes deriving from `typing.Protocol`. A contract the project's own classes fulfil is an
    abstract base they subclass, so a missing method fails at construction and every type checker
    reads the same nominal type; a shape of data is a model. An object of a library the project
    cannot subclass is typed by the library's own types, or by a stub (`.pyi`) where the library is
    absent, which this rule leaves alone."""

    query = "SELECT * FROM class_classes WHERE is_protocol AND NOT ends_with(path, '.pyi')"
    message = (
        "`{name}` is a Protocol: an abstract base the implementations subclass, a model for "
        "data, or a stub for a library's objects says the same nominally"
    )
    examples = (
        Example(
            bad=(
                """from typing import Protocol


class Reader(Protocol):
    def read(self) -> str: ...
"""
            ),
            good=(
                """from abc import ABC, abstractmethod


class Reader(ABC):
    @abstractmethod
    def read(self) -> str: ...
"""
            ),
        ),
    )
