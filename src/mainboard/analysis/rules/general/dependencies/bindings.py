"""Rules over artifacts one language declares and another reaches (mcmr's ALL-BIND family)."""

from ....catalog import Relation, Rule


class BindingsCrossings(Relation):
    """The `bindings_crossings()` relation."""

    query = """
    SELECT r.fact_id, r.ordinal, r."language", r.path, r.line, f.mechanism, f.name
    FROM interop_references r JOIN interop_facts f USING (fact_id)
    WHERE r."language" <> f.declared_language AND r."language" <> 'manifest'
"""


class UnreachedCrossLanguageArtifact(Rule, id="ALL-BIND0001"):
    """Artifacts one language declares that no other language reaches."""

    query = """
        SELECT * FROM interop_facts ANTI JOIN bindings_crossings() USING (fact_id)
        WHERE mechanism <> 'console-script'
    """
    message = (
        "{mechanism} `{name}` is declared in {declared_language} but no other language reaches it"
    )


class CrossLanguageBoundaryWidth(Rule, id="ALL-BIND0002"):
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
