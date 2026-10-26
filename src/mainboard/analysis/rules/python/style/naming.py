"""Rules over what names promise (mcmr's PY-NAMI family)."""

from ....catalog import Example, Rule


class BooleanPredicateName(Rule, id="PY-NAMI0001", applicability="review"):
    """Boolean symbols whose names do not read as predicates.

    A name reads as one when it opens with an auxiliary (`prefixes`) or its first word is a verb
    in the third person (`fits`, `reaches`), a participle (`failed`, `locked`) or an adjective of
    capability (`recoverable`): a word ending in one of `endings`, but for the nouns that end
    the same way (`status`, `access`, `basis`).
    """

    prefixes: tuple[str, ...] = (
        "is_",
        "are_",
        "was_",
        "were_",
        "has_",
        "have_",
        "had_",
        "can_",
        "could_",
        "should_",
        "must_",
        "will_",
        "would_",
        "does_",
        "did_",
        "needs_",
        "supports_",
    )
    endings: tuple[str, ...] = ("s", "ed", "able", "ible")
    nouns: tuple[str, ...] = ("ss", "us", "is")

    query = """
        WITH flagged AS (
          SELECT s.* REPLACE (
            CASE WHEN s.reference_id IS NULL THEN f.path
                 ELSE s.reference_declaration_span_path END AS path,
            CASE WHEN s.reference_id IS NULL THEN f.start_line
                 ELSE s.reference_declaration_span_start_line END AS start_line,
            CASE WHEN s.reference_id IS NULL THEN f.start_column
                 ELSE s.reference_declaration_span_start_column END AS start_column,
            CASE WHEN s.reference_id IS NULL THEN f.end_line
                 ELSE s.reference_declaration_span_end_line END AS end_line,
            CASE WHEN s.reference_id IS NULL THEN f.end_column
                 ELSE s.reference_declaration_span_end_column END AS end_column)
          FROM symbol_symbols s JOIN symbol_facts f USING (fact_order, fact_id)
          WHERE s.returns_boolean AND NOT coalesce(list_bool_or(list_transform(
            $prefixes::VARCHAR[], lambda p: starts_with(ltrim(s.name, '_'), p))), false)
            AND NOT (list_bool_or(list_transform($endings::VARCHAR[], lambda e: ends_with(
                       regexp_extract(ltrim(s.name, '_'), '^[a-z0-9]+'), e)))
                     AND NOT list_bool_or(list_transform($nouns::VARCHAR[], lambda n: ends_with(
                       regexp_extract(ltrim(s.name, '_'), '^[a-z0-9]+'), n))))
        )
        SELECT s.*, r."references"
        FROM flagged s LEFT JOIN (
          SELECT parent_id AS record_id,
                 list({'path': span_path, 'start_line': span_start_line,
                       'start_column': span_start_column, 'end_line': span_end_line,
                       'end_column': span_end_column} ORDER BY ordinal) AS "references"
          FROM symbol_symbols_reference_references GROUP BY parent_id) r USING (record_id)
    """
    message = (
        "`{name}` answers with a Boolean and its name does not say so: it opens with no "
        "auxiliary such as is_ or has_, no verb such as fits and no participle such as failed"
    )
    examples = (
        Example(
            bad="""def ready(count: int) -> bool:
    return count > 0


started = ready(3)
""",
            good="""def is_ready(count: int) -> bool:
    return count > 0


started = is_ready(3)
""",
        ),
    )
    # The first prefix, at the declaration and every reference bound to it, when the kernel
    # saw them all. A name that carries a prefix further in (`_annotation_is_nullable`) already
    # reads as a predicate, and is left as written. The references the kernel binds stop at the
    # module, so a name another module imports or reaches as an attribute is left for a person.
    fix = """
        WITH renamed AS (
          SELECT *, regexp_extract(name, '^_*') || ($prefixes::VARCHAR[])[1] || ltrim(name, '_')
                   AS renamed_to
          FROM finding f
          WHERE reference_id IS NOT NULL AND reference_are_references_complete
            AND len($prefixes::VARCHAR[]) > 0
            AND NOT coalesce(list_bool_or(list_transform(
              $prefixes::VARCHAR[], lambda p: contains(name, '_' || p))), false)
            AND NOT EXISTS (SELECT 1 FROM import_binding_facts i
                            WHERE i.imported_name = f.name OR i.name = f.name)
            AND NOT EXISTS (SELECT 1 FROM core_ref a
                            WHERE a.parent_type = 'attribute' AND a.place = 1
                              AND a.name = f.name AND a.path <> f.path)
        )
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               renamed_to AS text
        FROM renamed
        UNION ALL BY NAME
        SELECT finding_key, renamed_to AS text, unnest("references", recursive := true)
        FROM renamed
    """


class AttributeVisibility(Rule, id="PY-NAMI0002", applicability="review"):
    """Public methods proven to be private implementation details."""

    query = """
        SELECT d.*, d.span_path AS path, d.span_start_line AS start_line,
               d.span_start_column AS start_column, d.span_end_line AS end_line,
               d.span_end_column AS end_column
        FROM symbol_reach_declarations d JOIN symbol_reach_facts f USING (fact_order, fact_id)
        WHERE NOT f.is_test_module AND d.kind = 'method' AND NOT d.is_decorated
          AND d.visibility = 'public' AND d.owner_visibility <> 'public'
          AND NOT d.owner_has_inheritance AND d.owner_references > 0
          AND d.non_owner_references = 0 AND d.unresolved_name_references = 0
    """
    message = "`{qualname}` is public although only its non-public owner reaches it"
    examples = (
        Example(
            bad="""class _Helper:
    def run(self):
        return self.step()

    def step(self):
        return 1
""",
            good="""class _Helper:
    def run(self):
        return self._step()

    def _step(self):
        return 1
""",
        ),
    )
    # The method gets a leading underscore at its `def` and at every access inside its owner
    # class, when those are exactly the owner references the facts counted and the private name
    # is free.
    fix = """
        WITH owners AS (
          SELECT f.finding_key, f.path, f.qualname, f.owner_references,
                 regexp_extract(f.qualname, '[^.]+$') AS method, f.span_start_line AS def_line,
                 c.start_line AS class_start, c.end_line AS class_end
          FROM finding f
          JOIN class_classes c ON c.path = f.path AND c.start_line <= f.span_start_line
            AND f.span_start_line <= c.end_line
            AND c.name = string_split(f.qualname, '.')[-2]
        ), uses AS (
          SELECT o.finding_key, o.path, a.end_line AS line,
                 a.end_column::BIGINT - strlen(o.method) AS column_at
          FROM owners o
          JOIN core_file f ON f.path = o.path
          JOIN source_files s ON s.path = o.path
          JOIN core_node a ON a.file_id = f.file_id AND a.type = 'attribute'
            AND a.start_line BETWEEN o.class_start AND o.class_end
          JOIN core_node m ON m.file_id = a.file_id AND m.parent_id = a.node_id
            AND m.sibling_index = 1 AND m.name = o.method
          JOIN core_node r ON r.file_id = a.file_id AND r.parent_id = a.node_id
            AND r.sibling_index = 0
          JOIN core_def k ON k.file_id = a.file_id AND k.def_id = a.class_id
          WHERE (r.type = 'identifier' AND (r.name IN ('self', 'cls') OR r.name = k.name)
                 OR r.type = 'call' AND syntax_slice(s.content, r.start_line, r.start_column,
                                                     r.end_line, r.end_column) = 'super()')
            AND ends_with(syntax_slice(s.content, a.start_line, a.start_column, a.end_line,
                                       a.end_column), '.' || o.method)
        ), definitions AS (
          SELECT o.finding_key, o.path, o.def_line AS line,
                 strlen(left(l.text, strpos(l.text, 'def ' || o.method || '(') + 3))
                   AS column_at
          FROM owners o JOIN imports_code_lines() l ON l.path = o.path AND l.line = o.def_line
          WHERE strpos(l.text, 'def ' || o.method || '(') > 0
        ), sites AS (
          SELECT * FROM uses
          UNION ALL BY NAME
          SELECT * FROM definitions
        ), complete AS (
          SELECT o.finding_key FROM owners o
          WHERE (SELECT count(*) FROM uses u WHERE u.finding_key = o.finding_key)
                = o.owner_references
            AND EXISTS (SELECT 1 FROM definitions d WHERE d.finding_key = o.finding_key)
            AND NOT EXISTS (SELECT 1 FROM symbol_reach_declarations d
                            WHERE d.qualname = replace(o.qualname, '.' || o.method,
                                                       '._' || o.method))
        )
        SELECT s.finding_key, s.path, s.line AS start_line, s.column_at AS start_column,
               s.line AS end_line, s.column_at AS end_column, '_' AS text
        FROM sites s SEMI JOIN complete USING (finding_key)
    """
