"""Rules over how classes are named and grouped (mcmr's PY-CLAS organization family)."""

from ....catalog import Example, Function, Rule


class ClassesOrganizationRegistryKey(Function):
    """The key patos' `Registry` derives from a class name: `HTTPServerBackend` becomes
    `http-server-backend`, cut where a lowercase letter meets a capital and before a capitalized
    word, then lowercased."""

    class_name: str

    expression = (
        r"lower(regexp_replace(regexp_replace(class_name, '([a-z])([A-Z])', '\1-\2', 'g'), "
        r"'([a-zA-Z0-9])([A-Z][a-z])', '\1-\2', 'g'))"
    )


class ExplicitRegistryName(Rule, id="PY-CLAS0001", applicability="review"):
    """Registry classes that override their derivable class key."""

    registry_bases: tuple[str, ...] = ("Registry", "Strategy", "Backend", "Provider", "Component")

    query = r"""
            SELECT c.*, b.bases FROM class_classes c
            JOIN (SELECT class_id, string_agg(value, '`, `' ORDER BY ordinal) AS bases,
                         bool_or(list_contains($registry_bases::VARCHAR[],
                           regexp_replace(value, '(?s)^.*\.', '', 'g'))) AS registry
                  FROM class_direct_bases GROUP BY class_id) b USING (class_id)
            WHERE c.has_explicit_registry_name AND b.registry
        """
    message = "`{name}` explicitly overrides the registry name derived by `{bases}`"
    # The class-level `name = "..."` goes when it spells what patos' `Registry` derives from the
    # class name, so removing it keeps the key; a different spelling is a wire name and stays.
    # The statement's line goes with the blank lines after it when the body goes on (else the
    # ones before it, so the class keeps its separation from what follows), and an assignment
    # that was the whole body becomes `pass`. Review: the derivation is patos', which a
    # project's own `Backend` or `Provider` base may not share.
    fix = r"""
        WITH assignments AS (
          SELECT f.finding_key, f.name AS class_name, n.fact_id, n.ordinal, n.path,
                 n.start_line, n.start_column, n.end_line, n.end_column,
                 node_text(s.source, n.byte_start, n.byte_length) AS statement
          FROM finding f
          JOIN syntax_facts s ON s.path = f.path AND s.start_line = f.start_line
            AND s.start_column = f.start_column AND s.kind = 'type'
          JOIN syntax_nodes n ON n.fact_id = s.fact_id AND n.depth = 2 AND n.kind = 'binding'
            AND n.name = 'name'
        ), spelled AS (
          SELECT *, coalesce(
                   nullif(regexp_extract(statement, '^name\s*=\s*"([^"\\]*)"$', 1), ''),
                   nullif(regexp_extract(statement, '^name\s*=\s*''([^''\\]*)''$', 1), '')
                 ) AS explicit,
                 EXISTS (SELECT 1 FROM syntax_nodes o
                         WHERE o.fact_id = a.fact_id AND o.depth = 2 AND o.ordinal <> a.ordinal
                           AND o.kind NOT IN ('name', 'literal', 'call', 'member', 'index')
                        ) AS has_other_statements,
                 EXISTS (SELECT 1 FROM source_lines() l
                         WHERE l.path = a.path AND l.line = a.start_line
                           AND trimmed(l.text) = a.statement) AS owns_its_line,
                 (SELECT min(l.line) FROM source_lines() l WHERE l.path = a.path
                    AND l.line > a.end_line AND trimmed(l.text) <> '') AS next_code,
                 (SELECT max(l.line) FROM source_lines() l WHERE l.path = a.path
                    AND l.line < a.start_line AND trimmed(l.text) <> '') AS previous_code
          FROM assignments a
        ), placed AS (
          SELECT *, coalesce((SELECT strlen(l.text) - strlen(ltrim(l.text, ' ' || chr(9)))
                              FROM source_lines() l
                              WHERE l.path = s.path AND l.line = s.next_code), 0)
                      >= s.start_column AS body_goes_on
          FROM spelled s
        )
        SELECT finding_key, path,
               CASE WHEN NOT has_other_statements OR body_goes_on THEN start_line
                    ELSE previous_code + 1 END AS start_line,
               if(has_other_statements, 0, start_column) AS start_column,
               CASE WHEN NOT has_other_statements THEN end_line
                    WHEN body_goes_on THEN next_code ELSE end_line + 1 END AS end_line,
               if(has_other_statements, 0, end_column) AS end_column,
               if(has_other_statements, '', 'pass') AS text
        FROM placed
        WHERE explicit = classes_organization_registry_key(class_name) AND owns_its_line
          AND start_line = end_line
    """
    examples = (
        Example(
            bad=(
                """from patos import Registry


class Backend(Registry):
    pass


class JsonBackend(Backend):
    name = "json-backend"

    def run(self):
        return 1
"""
            ),
            good=(
                """from patos import Registry


class Backend(Registry):
    pass


class JsonBackend(Backend):
    def run(self):
        return 1
"""
            ),
        ),
    )


class CoupledNestedTypeCandidate(Rule, id="PY-CLAS0002"):
    """Short tightly named classes that may form one nested namespace."""

    suffixes: tuple[str, ...] = ("Content", "Kind")
    minimum_types: int = 2
    minimum_coimports: int = 2
    maximum_type_lines: int = 30
    minimum_prefix_length: int = 3

    query = """
            SELECT g.*, coalesce(r.role_suffixes, '') AS role_suffixes FROM class_coupled_groups g
            LEFT JOIN (SELECT group_id,
                              string_agg(suffix, '`, `' ORDER BY ordinal) AS role_suffixes
                       FROM class_coupled_group_suffixes GROUP BY group_id) r USING (group_id)
            WHERE length(g.prefix) >= $minimum_prefix_length AND g.type_count >= $minimum_types
              AND g.maximum_type_lines <= $maximum_type_lines
              AND g.coimporting_module_count >= $minimum_coimports
              AND NOT EXISTS (SELECT 1 FROM class_coupled_group_suffixes s
                              WHERE s.group_id = g.group_id
                                AND NOT list_contains($suffixes::VARCHAR[], s.suffix))
        """
    message = (
        "`{prefix}` groups {type_count} short types ending in `{role_suffixes}` and they are "
        "co-imported by {coimporting_module_count} modules"
    )
