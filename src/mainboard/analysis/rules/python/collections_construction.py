"""Rules over how collections are built (mcmr's PY-COLL construction family)."""

from ...catalog import Example, Rule


class LiteralPairSequenceMappingCandidate(Rule, id="PY-COLL0002"):
    """Fixed pair sequences whose complete use has unique-key mapping semantics."""

    query = """
            SELECT f.path, f.start_line, f.start_column, f.fact_id, count(*) AS value
            FROM collection_pair_sequences p JOIN collection_facts f USING (fact_id)
            WHERE p.pair_count >= 2 AND p.keys_are_unique_literals AND p.has_single_assignment
              AND p.all_reads_are_lookup_loops
            GROUP BY ALL
        """
    message = "literal pair sequence mapping candidate is {value} for `{fact_id}`"


class LocalCollectionRepresentationCandidate(Rule, id="PY-COLL0003", applicability="review"):
    """Local collections whose use proves a list, tuple, or set interchangeable."""

    sequence_preference: str = "list"
    prefer_membership_set: bool = True

    query = """
            SELECT f.path, f.start_line, f.start_column, f.fact_id, count(*) AS value
            FROM collection_local_collections c JOIN collection_facts f USING (fact_id)
            WHERE c.value_count >= 2 AND c.has_homogeneous_literals
              AND ((c.all_reads_are_iteration AND c.kind <> $sequence_preference)
                   OR ($prefer_membership_set AND c.all_reads_are_membership
                       AND c.values_are_unique))
            GROUP BY ALL
        """
    message = "local collection representation candidate is {value} for `{fact_id}`"
    # The literal `name = (a, b)` or `[a, b]` swaps its brackets for those of the representation
    # its reads prove: the preferred sequence for a collection only iterated, `{...}` for one only
    # tested with `in`. Found as the one bracketed literal a function binds to that name; a name
    # bound twice in the file, a literal without brackets, or a collection with no read (both
    # proofs hold vacuously) is left alone. Review: the new type may be relied on elsewhere.
    fix = r"""
        WITH candidates AS (
          SELECT f.fact_id, f.path, c.record_id, c.name,
                 CASE WHEN c.all_reads_are_iteration AND c.all_reads_are_membership THEN NULL
                      WHEN c.all_reads_are_membership THEN 'set'
                      ELSE $sequence_preference END AS target
          FROM finding f JOIN collection_local_collections c ON c.fact_id = f.fact_id
          WHERE c.value_count >= 2 AND c.has_homogeneous_literals
            AND ((c.all_reads_are_iteration AND c.kind <> $sequence_preference)
                 OR ($prefer_membership_set AND c.all_reads_are_membership
                     AND c.values_are_unique))
        ), located AS (
          SELECT k.*, col.start_line, col.start_column, col.end_line, col.end_column,
                 node_text(s.source, col.byte_start, col.byte_length) AS literal,
                 count(*) OVER (PARTITION BY k.fact_id, k.name) AS bindings
          FROM candidates k
          JOIN syntax_facts s ON s.path = k.path AND s.kind = 'callable'
          JOIN syntax_nodes b ON b.fact_id = s.fact_id AND b.kind = 'binding' AND b.name = k.name
          JOIN syntax_children h ON h.fact_id = b.fact_id AND h.parent_ordinal = b.ordinal
            AND h.child_order = 0
          JOIN syntax_nodes col ON col.fact_id = h.fact_id AND col.ordinal = h.child_ordinal
            AND col.kind = 'collection'
        ), rewritten AS (
          SELECT c.fact_id, c.record_id, l.path, l.start_line, l.start_column, l.end_line,
                 l.end_column,
                 CASE c.target WHEN 'set' THEN '{' WHEN 'list' THEN '[' ELSE '(' END
                   || substr(l.literal, 2, length(l.literal) - 2)
                   || CASE c.target WHEN 'set' THEN '}' WHEN 'list' THEN ']' ELSE ')' END AS text,
                 l.bindings = 1 AND regexp_matches(l.literal, '(?s)^(\(.*\)|\[.*\])$')
                   AND c.target IS NOT NULL AS usable
          FROM candidates c LEFT JOIN located l ON l.record_id = c.record_id
        )
        SELECT hash(record_id) AS finding_key, path, start_line, start_column, end_line,
               end_column, text
        FROM rewritten WHERE usable
    """
    examples = (
        Example(
            bad=(
                "def total(values):\n    steps = (1, 2, 3)\n"
                "    return [step * values for step in steps]\n"
            ),
            good=(
                "def total(values):\n    steps = [1, 2, 3]\n"
                "    return [step * values for step in steps]\n"
            ),
        ),
    )


class ImmutableCollectionConstructor(Rule, id="PY-COLL0004", applicability="review"):
    """Calls of the builtin `tuple` or `frozenset`, the place a concrete immutable container enters
    project code: lists and sets are the representation unless immutability is a proven contract
    (a dictionary key, a stable public boundary, another API's demand)."""

    query = """
        SELECT c.node_path AS path, c.node_start_line AS start_line,
               c.node_start_column AS start_column, c.node_end_line AS end_line,
               c.node_end_column AS end_column, c.call_id,
               replace(c.qualified_name, 'builtins.', '') AS constructor
        FROM call_calls c
        WHERE c.qualified_name IN ('builtins.tuple', 'builtins.frozenset') AND NOT c.is_shadowed
          AND NOT EXISTS (SELECT 1 FROM call_keywords k WHERE k.call_id = c.call_id)
    """
    message = "`{constructor}(...)` builds an immutable collection where a list or set would do"
    # An empty call or one over a bracketed literal becomes the literal itself: `tuple()` `[]`,
    # `tuple([a, b])` and `tuple((a, b))` `[a, b]`, `frozenset()` `set()`, `frozenset({a})`
    # `{a}`. A call over anything else (an iterator, a name) changes what it accepts and stays
    # report-only. Review: hashability and the declared return type may rely on the old one.
    fix = r"""
        WITH arguments AS (
          SELECT f.finding_key, f.path, f.start_line, f.start_column, f.end_line, f.end_column,
                 f.constructor, count(e.call_id) AS given, min(e.text) AS argument
          FROM finding f
          LEFT JOIN call_expressions e ON e.call_id = f.call_id AND e.root_relation = 'argument'
           AND e.depth = 0
          GROUP BY ALL
        ), rewritten AS (
          SELECT *, CASE
            WHEN given = 0 THEN CASE constructor WHEN 'tuple' THEN '[]' ELSE 'set()' END
            WHEN given > 1 THEN NULL
            WHEN constructor = 'tuple' AND regexp_matches(argument, '(?s)^\[.*\]$') THEN argument
            WHEN constructor = 'tuple' AND regexp_matches(argument, '(?s)^\(.*,.*\)$')
              THEN '[' || substr(argument, 2, length(argument) - 2) || ']'
            WHEN constructor = 'frozenset' AND regexp_matches(argument, '(?s)^\{.+\}$') THEN argument
          END AS text
          FROM arguments
        )
        SELECT finding_key, path, start_line, start_column, end_line, end_column, text
        FROM rewritten WHERE text IS NOT NULL
    """
    examples = (
        Example(
            bad='def names():\n    return tuple(["a", "b"])\n',
            good='def names():\n    return ["a", "b"]\n',
        ),
    )
