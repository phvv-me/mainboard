"""Rules over annotated collections (mcmr's PY-COLL annotations family)."""

from ...catalog import Example, Rule


class ConcreteCollectionParameter(Rule, id="PY-COLL0001", applicability="review"):
    """Concrete collection parameters that require only an abstract capability. A fixed-size
    `tuple[A, B]` is a pair, not a collection, so no abstract type says it."""

    query = """
            WITH found AS (
              SELECT coalesce(p.span_path, f.path) AS path,
                     coalesce(p.span_start_line, f.start_line) AS start_line,
                     coalesce(p.span_start_column, f.start_column) AS start_column,
                     p.owner, p.name, p.annotation, p.record_id
              FROM parameter_parameters p JOIN parameter_facts f USING (fact_order, fact_id)
              WHERE p.annotation IN ('list', 'tuple', 'dict', 'set')
                AND p.all_uses_known AND NOT p.is_return_value
                AND NOT EXISTS (
                  SELECT 1 FROM parameter_values v
                  WHERE v.relation = 'parameters.operations' AND v.parent_id = p.record_id
                    AND v.string_value IN ('add', 'append', 'clear', 'delitem', 'extend', 'insert',
                                           'pop', 'remove', 'setitem', 'update'))
            ), written AS (
              SELECT f.record_id,
                     syntax_slice(s.content, a.start_line, a.start_column, a.end_line,
                                  a.end_column) AS text
              FROM found f
              JOIN core_annotation a ON a.path = f.path AND a.role = 'parameter'
                AND a.start_line = f.start_line AND a.start_column > f.start_column
                AND a.end_line = f.start_line
              JOIN source_files s ON s.path = a.path
              WHERE f.annotation = 'tuple'
              QUALIFY row_number() OVER (PARTITION BY f.record_id ORDER BY a.start_column) = 1
            )
            SELECT f.* FROM found f LEFT JOIN written w USING (record_id)
            WHERE NOT (regexp_matches(coalesce(w.text, ''), '^tuple\\[') AND NOT contains(w.text, '...'))
        """
    message = (
        "`{owner}` declares `{name}` as a `{annotation}` and never does anything only a "
        "`{annotation}` can do"
    )
    # The annotation's head becomes the narrowest `collections.abc` type the body's operations
    # fit: `Iterable` (only iteration and reductions), `Collection` (with `len` or `in`),
    # `Sequence` (with indexing), `Mapping` for a dict, `AbstractSet` for a set, importing it
    # unless the file does. Only when every operation is one of those the type offers: the
    # finding also stands for `setdefault` or `sort`, which an abstract type lacks, and for a
    # parameter with no operation at all, whose capability nothing shows. A fixed-size
    # `tuple[A, B]` has no abstract spelling and stays. Review: callers may pass a type the
    # abstract one does not cover; a type checker settles that.
    fix = r"""
        WITH used AS (
          SELECT parent_id, list(DISTINCT string_value) AS operations
          FROM parameter_values WHERE relation = 'parameters.operations' GROUP BY parent_id
        ), annotated AS (
          SELECT f.finding_key, f.path, f.annotation,
                 syntax_slice(s.content, a.start_line, a.start_column, a.end_line, a.end_column)
                   AS node_text,
                 u.operations, a.start_line, a.start_column, a.end_line, a.end_column,
                 ['iter', 'sorted', 'min', 'max', 'sum', 'enumerate', 'any', 'all'] AS reductions
          FROM finding f
          JOIN used u ON u.parent_id = f.record_id
          JOIN core_annotation a ON a.path = f.path AND a.role = 'parameter'
            AND a.start_line = f.start_line AND a.start_column > f.start_column
            AND a.end_line = f.start_line
          JOIN source_files s ON s.path = a.path
          QUALIFY row_number() OVER (PARTITION BY f.finding_key ORDER BY a.start_column) = 1
        ), chosen AS (
          SELECT *, CASE
            WHEN annotation = 'dict' AND len(list_filter(operations, lambda o: NOT list_contains(
                   list_concat(reductions, ['getitem', 'get', 'items', 'keys', 'values',
                                            'contains', 'len']), o))) = 0 THEN 'Mapping'
            WHEN annotation = 'set' AND len(list_filter(operations, lambda o: NOT list_contains(
                   list_concat(reductions, ['contains', 'len']), o))) = 0 THEN 'AbstractSet'
            WHEN annotation IN ('list', 'tuple') AND len(list_filter(operations,
                   lambda o: NOT list_contains(list_concat(reductions, [
                     'getitem', 'len', 'contains', 'reversed', 'index', 'count']), o))) = 0
              THEN CASE WHEN list_has_any(operations, ['getitem', 'reversed', 'index', 'count'])
                        THEN 'Sequence'
                        WHEN list_has_any(operations, ['len', 'contains']) THEN 'Collection'
                        ELSE 'Iterable' END
          END AS capability
          FROM annotated
        )
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               regexp_replace(regexp_replace(node_text, '^(list|tuple|dict|set)\b', capability),
                              ',\s*\.\.\.\s*\]', ']') AS text,
               if(EXISTS (SELECT 1 FROM import_binding_facts i
                          WHERE i.path = chosen.path AND i.name = chosen.capability),
                  NULL,
                  if(capability = 'AbstractSet', 'from collections.abc import Set as AbstractSet',
                     'from collections.abc import ' || capability)) AS "import"
        FROM chosen
        WHERE capability IS NOT NULL AND regexp_matches(node_text, '^(list|tuple|dict|set)\b')
          AND (annotation <> 'tuple' OR NOT starts_with(node_text, 'tuple[')
               OR len(regexp_extract_all(node_text, '\.\.\.')) = 1)
    """
    examples = (
        Example(
            bad="import os\n\n\ndef first(values: list[int]) -> int:\n    return values[0]\n",
            good=(
                "import os\nfrom collections.abc import Sequence\n\n\n"
                "def first(values: Sequence[int]) -> int:\n    return values[0]\n"
            ),
        ),
    )
