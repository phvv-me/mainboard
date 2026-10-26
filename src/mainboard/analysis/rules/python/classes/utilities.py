"""Rules over classes that only namespace functions (mcmr's PY-CLAS utilities family)."""

from ....catalog import Example, Rule
from ..constants.functions import PARAMETER_LIST_ATOM


class UtilityNamespaceClassCount(Rule, id="PY-CLAS0003"):
    """Stateless classes that only namespace module-shaped functions. A class under
    `if TYPE_CHECKING:` is left out, as a type stub is. A dunder method (`__class_getitem__`,
    `__getattr__`) is Python's hook on the type, which no module function can stand in for, so a
    class holding one is not a namespace."""

    query = """
        WITH methods AS (
          SELECT m.class_id, m.name, m.ordinal,
                 ((coalesce(d.decorated, 0) > 0 AND coalesce(d.unbound, 0) = 0)
                  OR (coalesce(d.decorated, 0) = 0 AND m.kind = 'method'))
                 AND NOT m.reads_receiver_state AND NOT m.is_protocol_name AS is_function
          FROM class_methods m
          LEFT JOIN (
            SELECT method_id, count(*) AS decorated,
                   count(*) FILTER (WHERE value NOT IN ('staticmethod', 'classmethod')) AS unbound
            FROM class_method_decorators GROUP BY method_id) d USING (method_id)
        ), namespaced AS (
          SELECT class_id, string_agg(name, '`, `' ORDER BY ordinal) AS method_names
          FROM methods GROUP BY class_id HAVING coalesce(bool_and(is_function), true)
        )
        SELECT c.*, n.method_names FROM class_classes c JOIN namespaced n USING (class_id)
        WHERE NOT c.is_test AND NOT ends_with(c.path, '.pyi') AND NOT c.has_instance_fields
          AND NOT EXISTS (SELECT 1 FROM class_direct_bases b
                          WHERE b.class_id = c.class_id AND b.value <> 'object')
          AND NOT EXISTS (SELECT 1 FROM class_class_keywords k
                          WHERE k.class_id = c.class_id AND k.value <> 'metaclass=type')
          AND NOT EXISTS (SELECT 1 FROM type_checking_blocks() t
                          WHERE t.path = c.path AND c.start_line BETWEEN t.start_line AND t.end_line)
        """
    message = "`{name}` only namespaces `{method_names}`"


class StaticmethodCallingClassmethodCount(Rule, id="PY-CLAS0004", applicability="review"):
    """Static methods that hard-code their owner to call a sibling class method."""

    query = """
            SELECT m.*, c.name AS class_name FROM class_methods m
            JOIN class_classes c USING (class_id)
            WHERE m.kind = 'static_method'
              AND m.method_id IN (
                SELECT method_id FROM class_method_decorators
                GROUP BY method_id HAVING count(*) = 1 AND min(value) = 'staticmethod')
              AND EXISTS (
                SELECT 1 FROM class_owner_qualified_calls q
                JOIN class_methods s ON s.class_id = m.class_id AND s.kind = 'class_method'
                WHERE q.method_id = m.method_id AND q.value = c.name || '.' || s.name)
        """
    message = (
        "static method `{class_name}.{name}` hard-codes its owner to call a sibling class method"
    )
    # `@staticmethod` becomes `@classmethod`, the method gains a first parameter `cls`, and each
    # `Owner.sibling` naming a class method of the owner becomes `cls.sibling`. Left alone: a
    # method that already uses the name `cls`, one reaching its owner in the signature (a default
    # is evaluated before `cls` exists), and one whose parameter list the text reading below
    # cannot follow (a string or deep brackets in it). Review: a subclass now reaches its own
    # class methods through the call, which is the point and a change of behavior.
    fix = (
        r"""
        WITH methods AS (
          SELECT f.finding_key, f.path, f.class_id, f.class_name, f.name AS method_name,
                 s.fact_id, d.ordinal AS decorator_ordinal, d.start_line AS decorator_line,
                 d.start_column AS decorator_column, d.end_line AS decorator_end_line,
                 d.end_column AS decorator_end_column,
                 node_text(s.source, n0.byte_start, n0.byte_length) AS definition_text,
                 n0.start_line AS definition_line, n0.start_column AS definition_column
          FROM finding f
          JOIN syntax_facts s ON s.path = f.path AND s.start_line = f.start_line
            AND s.start_column = f.start_column AND s.kind = 'callable'
          JOIN syntax_nodes n0 ON n0.fact_id = s.fact_id AND n0.ordinal = 0
          JOIN syntax_nodes d ON d.fact_id = s.fact_id AND d.depth = 2 AND d.kind = 'name'
            AND d.name = 'staticmethod'
        ), headers AS (
          SELECT *, regexp_extract(definition_text,
                                   '(?s)^.*?\bdef[ \t]+' || method_name || '[ \t]*\(') AS opening
          FROM methods
        ), parsed AS (
          SELECT *, substr(definition_text, length(opening) + 1) AS rest,
                 regexp_extract(substr(definition_text, length(opening) + 1),
                                '^({atom}*?)\)', 1) AS parameters,
                 regexp_matches(substr(definition_text, length(opening) + 1), '^{atom}*?\)')
                   AS followed
          FROM headers WHERE opening <> ''
        ), placed AS (
          SELECT *, definition_line + len(string_split(opening || parameters || ')', chr(10))) - 1
                      AS header_end_line,
                 string_split(opening, chr(10)) AS opening_lines,
                 regexp_extract(rest, '^\s*') AS blanks
          FROM parsed WHERE followed
        ), owner_references AS (
          SELECT p.finding_key, r.start_line, r.start_column, r.end_line, r.end_column,
                 p.header_end_line
          FROM placed p
          JOIN class_methods sibling ON sibling.class_id = p.class_id
            AND sibling.kind = 'class_method'
          JOIN syntax_nodes r ON r.fact_id = p.fact_id AND r.kind = 'name'
            AND r.name = p.class_name
          JOIN syntax_nodes m ON m.node_id = r.parent_id AND m.kind = 'member'
            AND m.name = sibling.name
        ), editable AS (
          SELECT p.* FROM placed p
          WHERE NOT EXISTS (SELECT 1 FROM syntax_nodes c
                            WHERE c.fact_id = p.fact_id AND c.name = 'cls')
            AND NOT EXISTS (SELECT 1 FROM owner_references o
                            WHERE o.finding_key = p.finding_key
                              AND o.start_line <= p.header_end_line)
        )
        SELECT finding_key, path, decorator_line AS start_line, decorator_column AS start_column,
               decorator_end_line AS end_line, decorator_end_column AS end_column,
               'classmethod' AS text
        FROM editable
        UNION ALL BY NAME
        SELECT finding_key, path,
               definition_line + len(opening_lines) - 1 AS start_line,
               if(len(opening_lines) = 1, definition_column, 0) + strlen(opening_lines[-1])
                 AS start_column,
               definition_line + len(opening_lines) - 1 AS end_line,
               if(len(opening_lines) = 1, definition_column, 0) + strlen(opening_lines[-1])
                 AS end_column,
               CASE WHEN starts_with(substr(rest, length(blanks) + 1), ')') THEN 'cls'
                    WHEN contains(blanks, chr(10)) THEN blanks || 'cls,'
                    ELSE 'cls, ' END AS text
        FROM editable
        UNION ALL BY NAME
        SELECT o.finding_key, e.path, o.start_line, o.start_column, o.end_line, o.end_column,
               'cls' AS text
        FROM owner_references o JOIN editable e USING (finding_key)
    """
    ).replace("{atom}", PARAMETER_LIST_ATOM.replace("'", "''"))
    examples = (
        Example(
            bad=(
                """class Owner:
    @classmethod
    def make(cls, size):
        return cls()

    @staticmethod
    def build(size):
        return Owner.make(size)
"""
            ),
            good=(
                """class Owner:
    @classmethod
    def make(cls, size):
        return cls()

    @classmethod
    def build(cls, size):
        return cls.make(size)
"""
            ),
        ),
    )
