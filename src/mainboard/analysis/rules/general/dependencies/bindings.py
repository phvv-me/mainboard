"""Rules over artifacts one language declares and another reaches (mcmr's ALL-BIND family)."""

from ....catalog import Part, Relation


class BindingsCrossings(Relation):
    """The `bindings_crossings()` relation."""

    query = """
    SELECT r.fact_id, r.ordinal, r."language", r.path, r.line, f.mechanism, f.name
    FROM interop_references r JOIN interop_facts f USING (fact_id)
    WHERE r."language" <> f.declared_language AND r."language" <> 'manifest'
"""


class UnreachedCrossLanguageArtifact(Part):
    """Artifacts one language declares that no other language reaches."""

    query = """
        SELECT * FROM interop_facts ANTI JOIN bindings_crossings() USING (fact_id)
        WHERE mechanism <> 'console-script'
    """
    message = (
        "{mechanism} `{name}` is declared in {declared_language} but no other language reaches it"
    )
    # A CUDA kernel whose name no file spells but its definition goes with the definition. Any
    # other artifact is a contract a person narrows. Review: a consumer outside the repository may
    # launch the kernel, which the facts cannot see.
    fix = """
        SELECT f.finding_key, 'delete-node' AS operation, f.path, d.node_id AS node
        FROM finding f
        JOIN ast_nodes d ON d.path = f.path AND d.type = 'function_definition' AND d.name = f.name
        WHERE f.mechanism = 'kernel'
          AND NOT EXISTS (SELECT 1 FROM source_files s
                          WHERE contains(s.content, f.name)
                            AND (s.path <> f.path OR len(string_split(s.content, f.name)) > 2))
    """
    change = "deleted the kernel `{name}`, which no other language reaches"


class CrossLanguageBoundaryWidth(Part):
    """Artifacts reached from more than one other language."""

    maximum: int = 1

    query = """
        SELECT *, count(*) OVER (PARTITION BY fact_id) AS width FROM (
          SELECT fact_id, "language", mechanism, name, arg_min(path, ordinal) AS path,
                 arg_min(line, ordinal) AS start_line, 0 AS start_column,
                 string_agg(path || ':' || line, '`, `' ORDER BY ordinal) AS locations
          FROM bindings_crossings() GROUP BY ALL)
        QUALIFY width > $maximum
    """
    message = "{mechanism} `{name}` is reached from {language} at `{locations}`"
