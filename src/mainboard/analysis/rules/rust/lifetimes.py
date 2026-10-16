"""Rules over the lifetimes Rust modules name (mcmr's RS-LIFE family)."""

from ...catalog import Example, Relation, Rule


class LifetimesAnnotations(Relation):
    """The `lifetimes_annotations()` relation."""

    query = """
    SELECT a.*, coalesce(n.stated_names, '') AS stated_names FROM rust_surface_annotations a
    LEFT JOIN (
      SELECT parent_id, string_agg('`''' || string_value || '`', ', ' ORDER BY ordinal)
               AS stated_names
      FROM rust_surface_values WHERE relation = 'annotations.names' GROUP BY parent_id) n
      ON n.parent_id = a.record_id
"""


class ElidableLifetimeAnnotation(Rule, id="RS-LIFE0001", applicability="safe"):
    """Lifetime annotations that elision would have written on its own."""

    query = """
        WITH stated AS (
          SELECT * FROM lifetimes_annotations() WHERE kind IN ('function', 'method')
            AND names_length > 0
        ), named AS (
          SELECT parent_id, string_value AS name,
                 count(*) FILTER (relation = 'annotations.parameters') AS parameter_count,
                 count(*) FILTER (relation = 'annotations.returned') AS matching_count,
                 bool_or(relation IN ('annotations.beyond', 'annotations.required_by_syntax'))
                   AS blocked,
                 bool_or(relation = 'annotations.names') AS stated
          FROM rust_surface_values
          WHERE relation IN ('annotations.names', 'annotations.parameters', 'annotations.returned',
                             'annotations.beyond', 'annotations.required_by_syntax')
          GROUP BY parent_id, string_value HAVING stated
        ), returned AS (
          SELECT parent_id, count(*) AS returned_count FROM rust_surface_values
          WHERE relation = 'annotations.returned' GROUP BY parent_id
        ), counted AS (
          SELECT n.*, s.receiver, coalesce(r.returned_count, 0) AS returned_count,
                 n.parameter_count + (s.receiver = n.name)::INT AS inputs
          FROM named n JOIN stated s ON s.record_id = n.parent_id
          LEFT JOIN returned r ON r.parent_id = n.parent_id
        ), elidable AS (
          SELECT parent_id FROM counted GROUP BY parent_id
          HAVING bool_and(name <> 'static' AND NOT blocked AND
            CASE WHEN matching_count = 0 THEN inputs = 1
                 ELSE receiver = name AND inputs = 1 AND matching_count = returned_count END)
        )
        SELECT f.path, s.line AS start_line, 0 AS start_column, s.owner, s.stated_names
        FROM stated s SEMI JOIN elidable e ON e.parent_id = s.record_id
        JOIN rust_surface_facts f USING (fact_order, fact_id)
    """
    message = "`{owner}` names {stated_names}, which elision would have produced on its own"
    # The signature (from the `fn` line to the body's `{`) is rewritten whole: the lifetime list
    # leaves the generics, `&'a T` becomes `&T`, and any other mention (`Foo<'a>`, `dyn T + 'a`)
    # becomes `'_`, which is what elision builds. A lifetime with a bound (`'b: 'a`) or a signature
    # holding a comment is left.
    fix = r"""
        SELECT finding_key, path, start_line, 0 AS start_column, end_line + 1 AS end_line,
               0 AS end_column,
               list_reduce(names, lambda acc, lifetime: regexp_replace(
                   regexp_replace(acc, '&\s*\x27' || lifetime || '\s+', '&', 'g'),
                   '\x27' || lifetime || '\b', chr(39) || '_', 'g'), declared_free)
                 || rest || chr(10)
                 AS text
        FROM (
          SELECT *, regexp_replace(regexp_replace(head,
                      '(?s)(\bfn\s+' || ident
                        || '\s*)<\s*(?:\x27\w+\s*,\s*)*\x27\w+\s*,?\s*>', '\1'),
                    '(?s)(\bfn\s+' || ident || '\s*<\s*)(?:\x27\w+\s*,\s*)+', '\1')
                    AS declared_free
          FROM (
            SELECT *, regexp_extract(signature, '(?s)^([^{]*)', 1) AS head,
                   regexp_extract(signature, '(?s)^[^{]*(.*)$', 1) AS rest
            FROM (
              SELECT *, (SELECT string_agg(l.text, chr(10) ORDER BY l.line)
                         FROM source_lines() l WHERE l.path = f.path
                           AND l.line BETWEEN f.start_line AND f.end_line) AS signature,
                     regexp_extract_all(f.stated_names, '\x27(\w+)', 1) AS names,
                     regexp_extract(f.owner, '(\w+)$', 1) AS ident
              FROM (SELECT *, (SELECT min(l.line) FROM source_lines() l
                               WHERE l.path = finding.path AND l.line >= finding.start_line
                                 AND contains(l.text, '{')) AS end_line
                    FROM finding) f))
          WHERE NOT regexp_matches(signature, '//|/\*')
            AND NOT regexp_matches(head, '\x27\w+\s*:')
            AND regexp_matches(head, '\bfn\s+' || ident))
    """
    examples = (
        Example(
            bad="fn first<'a>(&'a self) -> &'a str {\n    &self.value\n}\n",
            good="fn first(&self) -> &str {\n    &self.value\n}\n",
        ),
    )


class DemandedStaticLifetime(Rule, id="RS-LIFE0002"):
    """Declarations that demand a `'static` reference from every caller."""

    query = """
        SELECT f.path, p.line AS start_line, 0 AS start_column,
               CASE WHEN p.owner = '' THEN f.path ELSE p.owner END AS owner
        FROM rust_surface_pins p JOIN rust_surface_facts f USING (fact_order, fact_id)
        WHERE p.position = 'demand'
    """
    message = (
        "`{owner}` demands a `'static` reference, so no caller can hand it anything read at run "
        "time"
    )


class LifetimeAnnotationCount(Rule, id="RS-LIFE0003"):
    """Modules that state more lifetime annotations than a reader can hold."""

    maximum: int = 2

    query = """
        SELECT f.path, a.line AS start_line, 0 AS start_column, a.owner, a.kind, a.stated_names
        FROM lifetimes_annotations() a JOIN rust_surface_facts f USING (fact_order, fact_id)
        WHERE f.annotations_length > $maximum
    """
    message = "`{owner}` is a {kind} naming {stated_names}"
