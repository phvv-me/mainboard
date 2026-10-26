"""Rules over how far each declaration's references reach (mcmr's ALL-REAC family)."""

from ....catalog import Example, Relation, Rule
from ...families.imports.common import name_reads


class ReachConfined(Relation):
    """The `reach_confined()` relation."""

    query = """
    SELECT d.span_path AS path, d.span_start_line AS start_line,
           d.span_start_column AS start_column, d.qualname, d.kind,
           d.own_file_references AS own
    FROM symbol_reach_declarations d JOIN symbol_reach_facts f USING (fact_order, fact_id)
    WHERE NOT f.is_test_module AND d.is_module_scope AND NOT d.is_decorated
      AND d.visibility = 'public' AND d.other_file_references = 0
"""


class ReachQuotedWords(Relation):
    """The words of every string literal that is a bare name or dotted path, once per file: how a
    name no reference fact links to a use is still reached (a dispatch key, a `module:name` entry
    point)."""

    stored = True
    query = r"""
        SELECT DISTINCT path, unnest(regexp_extract_all(literal, '[A-Za-z_][A-Za-z0-9_]*')) AS word
        FROM (SELECT path,
                     unnest(regexp_extract_all(content, '["''][A-Za-z_][A-Za-z0-9_.:]*["'']'))
                       AS literal
              FROM source_files)
    """


class UnreferencedPublicDeclaration(Rule, id="ALL-REAC0001", applicability="review"):
    """Public module-level classes and functions no reference in the repository reaches. A module
    that a configuration file names (a `[project.scripts]` or plugin entry) is loaded by a
    framework that calls its functions by name, and so is an `on_<event>` function of a `hooks/`
    folder (MkDocs reads those from `mkdocs.yml`), so none is dead. Neither is a class some
    ancestor's `__init_subclass__` registers on definition (a rule, a relation, a plugin): the
    hook is its reference, and deleting it would delete the registry's entries (505 of mainboard's
    own 539 findings were rules, 2026-10-23)."""

    query = r"""
        WITH RECURSIVE hooked AS (
          SELECT class_id FROM class_methods
          WHERE name IN ('__init_subclass__', '__pydantic_init_subclass__')
          UNION
          SELECT e.child FROM override_edges e JOIN hooked h ON e.parent = h.class_id
        )
        SELECT r.* FROM reach_confined() r
        WHERE kind IN ('class', 'function') AND own = 0
          -- Called by name (`carrier.call("where", ...)`, a dispatch key, an `__all__` entry),
          -- which is why the fix keeps it too.
          AND NOT EXISTS (SELECT 1 FROM reach_quoted_words() w
                          WHERE w.word = regexp_extract(r.qualname, '[^.]+$'))
          AND NOT (kind = 'class' AND EXISTS (
            SELECT 1 FROM hooked h JOIN class_classes c USING (class_id)
            WHERE c.path = r.path AND c.scope = 'module'
              AND c.name = regexp_extract(r.qualname, '[^.]+$')))
          AND NOT (starts_with(regexp_extract(qualname, '[^.]+$'), 'on_')
                   AND list_contains(string_split(r.path, '/'), 'hooks'))
          AND NOT EXISTS (
            SELECT 1 FROM source_files c
            WHERE regexp_matches(c.path, '\.(ya?ml|toml|json|cfg|ini)$')
              AND (contains(c.content, r.path)
                   OR contains(c.content, replace(regexp_replace(
                        regexp_replace(r.path, '\.pyi?$', ''), '^src/', ''), '/', '.'))))
    """
    message = "`{qualname}` is a public {kind} that no repository reference reaches"
    examples = (
        Example(
            bad="def used():\n    return 1\n\n\ndef unused():\n    return 2\n\n\nused()\n",
            good="def used():\n    return 1\n\n\nused()\n",
        ),
    )
    # The whole definition with the blank lines after it, unless some string literal of the
    # repository names it (an `__all__` entry, a dispatch key, an entry point) or it is a hook by
    # convention. What only it used becomes unreferenced in turn, and later passes delete that.
    fix = """
        SELECT finding_key, path,
               functions_delete_start(lines, first_line, last_line) AS start_line,
               0 AS start_column,
               functions_delete_end(lines, last_line) AS end_line, 0 AS end_column,
               CASE WHEN NOT starts_with(name, '__') AND name <> 'main'
                      AND NOT starts_with(name, 'pytest_')
                      AND NOT EXISTS (SELECT 1 FROM reach_quoted_words() w WHERE w.word = name)
                    THEN '' END AS text
        FROM (
          SELECT f.finding_key, f.path, f.start_line AS first_line, s.end_line AS last_line,
                 regexp_extract(f.qualname, '[^.]+$') AS name, c.content,
                 string_split(c.content, chr(10)) AS lines
          FROM finding f
          JOIN syntax_facts s ON s.path = f.path AND s.start_line = f.start_line
            AND s.start_column = 0 AND s.kind IN ('callable', 'type')
          JOIN source_files c ON c.path = f.path
          WHERE ends_with(f.path, '.py')) located
    """


class ReachNameOccurrences(Relation):
    """Where the name of each file-local Python function occurs in its own file.

    One row per occurrence, with how many there are and where the `def` puts its name. The
    occurrences are the syntax tree's identifiers used as names (attributes, keywords,
    parameters and import parts left out) plus the declaration, so a word in a string, a
    docstring or a comment is none. A name reads as renameable only when they are exactly its
    declaration plus the reads the facts counted (`occurrences = own + 1`), which leaves out a
    local of the same name.
    """

    query = (
        r"""
        WITH declared AS (
          SELECT c.path, c.start_line, c.own, regexp_extract(c.qualname, '[^.]+$') AS name,
                 g.definition_start_column
                   + strlen(regexp_extract(g.definition_text, '^(async\s+)?def\s+'))
                   AS declared_column
          FROM reach_confined() c
          JOIN function_functions g ON g.path = c.path AND g.definition_start_line = c.start_line
          WHERE c.kind = 'function' AND c.own > 0 AND g.scope = 'module'
            AND ends_with(c.path, '.py')
        ), occurring AS (
          SELECT d.path, d.start_line, d.name, d.own, d.declared_column, i.start_line AS line,
                 i.start_column AS at_column
          FROM declared d
          JOIN ast_nodes i ON i.path = d.path AND i.type = 'identifier' AND i.name = d.name
        """
        + name_reads()[0]
        + "WHERE "
        + name_reads()[1]
        + r"""
          UNION ALL
          SELECT path, start_line, name, own, declared_column, start_line AS line,
                 declared_column AS at_column
          FROM declared
        )
        SELECT *, count(*) OVER (PARTITION BY path, start_line) AS occurrences FROM occurring
    """
    )


class FileLocalPublicDeclaration(Rule, id="ALL-REAC0002", applicability="review"):
    """Public module-level functions only their own file ever uses."""

    query = """
        SELECT *, own || CASE WHEN own = 1 THEN ' time' ELSE ' times' END AS reads
        FROM reach_confined() WHERE kind = 'function' AND own > 0
    """
    message = (
        "`{qualname}` is a public {kind} read {reads} inside this file and nowhere outside it"
    )
    examples = (
        Example(
            bad="def helper():\n    return 1\n\n\ndef run():\n    return helper()\n",
            good="def _helper():\n    return 1\n\n\ndef run():\n    return _helper()\n",
        ),
    )
    # A leading underscore at the declaration and every read, when the file's text shows exactly
    # those uses, no `_name` already exists in it, and no string literal of the repository names
    # the function (a dispatch key, an entry point).
    fix = """
        SELECT f.finding_key, o.path, o.line AS start_line, o.at_column AS start_column,
               o.line AS end_line, o.at_column AS end_column, '_' AS text
        FROM finding f
        JOIN reach_name_occurrences() o ON o.path = f.path AND o.start_line = f.start_line
        JOIN source_files c ON c.path = f.path
        WHERE o.occurrences = o.own + 1
          AND NOT starts_with(o.name, 'test_') AND NOT starts_with(o.name, 'pytest_')
          AND o.name <> 'main'
          AND NOT regexp_matches(c.content, '\\b_' || o.name || '\\b')
          AND NOT EXISTS (SELECT 1 FROM reach_quoted_words() w WHERE w.word = o.name)
    """


class RepositoryWideDeclaration(Rule, id="ALL-REAC0003"):
    """Declarations whose use spreads across more packages than a contract should."""

    maximum_packages: int = 3

    query = """
        SELECT span_path AS path, span_start_line AS start_line, span_start_column AS start_column,
               qualname, referencing_packages, referencing_directories, referencing_files
        FROM symbol_reach_declarations WHERE referencing_packages > $maximum_packages
    """
    message = (
        "`{qualname}` is reached from {referencing_packages} packages, "
        "{referencing_directories} directories, and {referencing_files} files"
    )
