"""Python modules, their imports and their definitions, as relations over sitting_duck's trees.

A file's module name follows mainboard's closure walk (`jobs/closure.py`): the file is named from
the import root above its outermost regular package, climbing through identifier-named
directories, so `experiments/x/y.py` under a package `experiments` is `experiments.x.y` though
`x` has no initializer. A script outside any package is importable only from its own directory.
"""

from ...catalog import Relation


def _qualname(node: str) -> str:
    """The SQL for the name of the definition `node` inside its module (`Class.method`)."""
    return f"array_to_string(list_transform({node}.qualified_name, lambda q: q.name), '.')"


class ModuleFiles(Relation):
    """Each Python file with its module name, its package and the import root it sits under.

    packaged: the file lies in a package chain, so any file may import it by name; a script
        (False) is importable only from the files of its own directory.
    """

    stored = True
    query = r"""
        WITH sources AS (
          SELECT path, regexp_extract(path, '^(.*)/[^/]*$', 1) AS dir,
                 regexp_extract(path, '[^/]*$') AS name
          FROM python.ast_files
          WHERE path LIKE '%.py'
        ), initializers AS (
          SELECT DISTINCT dir FROM sources WHERE name = '__init__.py'
        ), levels AS (
          SELECT s.path, u.i, array_to_string(s.parts[1:u.i], '/') AS prefix,
                 regexp_full_match(s.parts[u.i], '[A-Za-z_][A-Za-z0-9_]*') AS identifier
          FROM (SELECT path, string_split(dir, '/') AS parts FROM sources WHERE dir <> '') s,
               unnest(generate_series(1, len(s.parts))) AS u(i)
        ), outermost AS (
          SELECT l.path, min(l.i) AS level
          FROM levels l
          JOIN (SELECT path, coalesce(max(i) FILTER (WHERE NOT identifier), 0) AS stop
                FROM levels GROUP BY path) b USING (path)
          JOIN initializers n ON n.dir = l.prefix
          WHERE l.i > b.stop
          GROUP BY l.path
        ), homes AS (
          SELECT s.path, s.dir, s.name, o.level IS NOT NULL AS packaged,
                 CASE WHEN o.level IS NULL THEN s.dir
                      ELSE array_to_string(string_split(s.dir, '/')[1:o.level - 1], '/')
                 END AS home
          FROM sources s LEFT JOIN outermost o USING (path)
        ), named AS (
          SELECT *, regexp_replace(
                      replace(regexp_replace(
                        CASE WHEN home = '' THEN path ELSE substr(path, length(home) + 2) END,
                        '\.py$', ''), '/', '.'),
                      '(^|\.)__init__$', '') AS module
          FROM homes
        )
        SELECT path, dir, home, module, packaged, name = '__init__.py' AS initializer,
               CASE WHEN name = '__init__.py' THEN module
                    ELSE regexp_replace(module, '\.?[^.]*$', '') END AS package
        FROM named
    """


class ModuleImports(Relation):
    """One row per name an import statement binds, with its place and absolute module.

    shape: `import` (`import a.b as c`) or `from` (`from a import b as c`).
    module: the absolute dotted module (a relative one resolved from the file's package, NULL when
        it climbs above the top package).
    name: the imported name of a `from` import (`*` for a wildcard), NULL for `import`.
    lazy: the statement sits in a function body, so it runs on a call and not at import.
    type_only: the statement sits under `if TYPE_CHECKING:` and never runs.
    """

    stored = True
    query = r"""
        WITH statements AS (
          SELECT n.path, n.node_id, n.type, n.name, n.start_line, n.start_column,
                 n.scope.function IS NOT NULL AS lazy,
                 EXISTS (
                   SELECT 1 FROM python.ast_nodes block
                   JOIN python.ast_nodes guard ON guard.path = block.path
                     AND guard.node_id = block.parent_id AND guard.type = 'if_statement'
                   JOIN python.ast_nodes test ON test.path = guard.path
                     AND test.parent_id = guard.node_id AND test.sibling_index = 1
                   WHERE block.path = n.path AND block.node_id = n.parent_id
                     AND test.name IN ('TYPE_CHECKING', 'typing.TYPE_CHECKING')
                 ) AS type_only
          FROM python.ast_nodes n
          WHERE n.type IN ('import_statement', 'import_from_statement')
        ), bound AS (
          SELECT s.*, c.type AS child_type, c.name AS child_name, c.node_id AS child_id,
                 c.sibling_index
          FROM statements s
          JOIN python.ast_nodes c ON c.path = s.path AND c.parent_id = s.node_id
          WHERE c.type IN ('dotted_name', 'aliased_import', 'wildcard_import')
        ), named AS (
          SELECT b.*, CASE WHEN b.child_type = 'aliased_import' THEN d.name ELSE b.child_name END
                        AS spec,
                 CASE WHEN b.child_type = 'aliased_import' THEN b.child_name END AS alias
          FROM bound b
          LEFT JOIN python.ast_nodes d ON d.path = b.path AND d.parent_id = b.child_id
            AND d.type = 'dotted_name' AND b.child_type = 'aliased_import'
          WHERE b.type = 'import_statement' OR b.sibling_index >= 2
        ), relative AS (
          SELECT n.*, m.package,
                 length(regexp_extract(n.name, '^\.*')) AS level,
                 ltrim(n.name, '.') AS rest
          FROM named n LEFT JOIN module_files() m ON m.path = n.path
        ), resolved AS (
          SELECT r.*,
                 CASE WHEN r.package = '' THEN []::VARCHAR[] ELSE string_split(r.package, '.') END
                   AS pieces
          FROM relative r
        )
        SELECT path, start_line AS line, start_column AS "column", node_id AS statement,
               CASE type WHEN 'import_statement' THEN 'import' ELSE 'from' END AS shape,
               CASE WHEN type = 'import_statement' THEN spec
                    WHEN level = 0 THEN name
                    WHEN level - 1 >= len(pieces) THEN NULL
                    ELSE nullif(array_to_string(list_concat(
                           pieces[1:len(pieces) - (level - 1)],
                           CASE WHEN rest = '' THEN []::VARCHAR[]
                                ELSE string_split(rest, '.') END),
                         '.'), '') END AS module,
               CASE WHEN type = 'import_from_statement' THEN
                      CASE WHEN child_type = 'wildcard_import' THEN '*' ELSE spec END END AS name,
               alias, level, lazy, type_only
        FROM resolved
    """


class ModuleEdges(Relation):
    """The modules each file imports: one row per statement and imported file.

    kind: `import` for the module the statement names (and, for `from a import b`, the module
        `a.b` when it is one); `package` for the initializers of its parent packages, which run
        first.
    """

    stored = True
    query = r"""
        WITH requests AS (
          SELECT path, line, statement, lazy, type_only, module AS request, true AS chain
          FROM module_imports() WHERE module IS NOT NULL
          UNION ALL
          SELECT path, line, statement, lazy, type_only, module || '.' || name, false
          FROM module_imports()
          WHERE shape = 'from' AND name <> '*' AND module IS NOT NULL
        ), pieces AS (
          SELECT *, string_split(request, '.') AS parts FROM requests
        ), prefixes AS (
          SELECT p.path, p.line, p.statement, p.lazy, p.type_only,
                 array_to_string(p.parts[1:u.j], '.') AS target, u.j = len(p.parts) AS leaf
          FROM pieces p,
               unnest(generate_series(CASE WHEN p.chain THEN 1 ELSE len(p.parts) END,
                                      len(p.parts))) AS u(j)
        )
        SELECT DISTINCT p.path AS importer, m.path AS imported, p.line, p.statement,
               CASE WHEN p.leaf THEN 'import' ELSE 'package' END AS kind, p.lazy, p.type_only
        FROM prefixes p
        JOIN module_files() home ON home.path = p.path
        JOIN module_files() m ON m.module = p.target AND (m.packaged OR m.dir = home.dir)
        WHERE m.path <> p.path
    """


class ModuleBindings(Relation):
    """The names each file binds by importing, resolved to a project file.

    local: the name (or dotted name, for `import a.b.c` without an alias) the file uses.
    imported: the file the name stands for; NULL when the import is external.
    member: the name inside `imported` a `from` import binds; NULL when the local name is the
        module itself.
    """

    stored = True
    query = r"""
        SELECT i.path, i.line, i.statement,
               CASE WHEN i.shape = 'import' THEN coalesce(i.alias, i.module)
                    ELSE coalesce(i.alias, i.name) END AS local,
               CASE WHEN i.shape = 'import' THEN base.path
                    ELSE coalesce(sub.path, base.path) END AS imported,
               CASE WHEN i.shape = 'from' AND sub.path IS NULL THEN i.name END AS member
        FROM module_imports() i
        JOIN module_files() home ON home.path = i.path
        LEFT JOIN module_files() base ON base.module = i.module
          AND (base.packaged OR base.dir = home.dir) AND base.path <> i.path
        LEFT JOIN module_files() sub ON i.shape = 'from' AND sub.module = i.module || '.' || i.name
          AND (sub.packaged OR sub.dir = home.dir) AND sub.path <> i.path
        WHERE i.module IS NOT NULL AND (i.shape = 'import' OR i.name <> '*')
    """


class ModuleDefinitions(Relation):
    """The functions, classes, methods and module constants each file defines.

    kind: `function`, `class`, `method` or `constant` (a module-level assignment).
    qualname: the name inside its module (`Board.verdicts` for a method).
    """

    stored = True
    query = f"""
        SELECT n.path, m.module, n.name, {_qualname("n")} AS qualname,
               CASE n.type WHEN 'class_definition' THEN 'class'
                           WHEN 'assignment' THEN 'constant'
                           WHEN 'function_definition' THEN
                             CASE WHEN n.scope.class IS NULL THEN 'function' ELSE 'method' END
               END AS kind,
               n.start_line AS line, n.end_line, n.scope.class IS NULL AS top_level
        FROM python.ast_nodes n
        JOIN module_files() m ON m.path = n.path
        WHERE n.name <> '' AND n.scope.function IS NULL
          AND (n.type IN ('function_definition', 'class_definition')
               OR (n.type = 'assignment' AND n.scope.class IS NULL
                   AND NOT starts_with(n.name, '__')
                   AND EXISTS (SELECT 1 FROM python.ast_nodes target
                               WHERE target.path = n.path AND target.parent_id = n.node_id
                                 AND target.sibling_index = 0 AND target.type = 'identifier'
                                 AND target.name = n.name)))
    """


class ReverseReach(Relation):
    """The files that import `seeds`, directly or through others, with the fewest import hops to
    one of them (the seeds are at 0). A cycle is cut at 64 hops, past any real chain.

    seeds: project-relative paths.
    """

    seeds: list[str]
    query = r"""
        WITH RECURSIVE reach(path, hops) AS (
          SELECT unnest(seeds), 0
          UNION
          SELECT e.importer, r.hops + 1
          FROM reach r JOIN module_edges() e ON e.imported = r.path
          WHERE r.hops < 64
        )
        SELECT path, min(hops) AS hops FROM reach GROUP BY path
    """


class DesignReach(Relation):
    """The files that import `seeds` through design edges only (no `if TYPE_CHECKING:` import, no
    facade or parent-package initializer), the reach mcmr's `impact` computed.

    seeds: project-relative paths.
    """

    seeds: list[str]
    query = r"""
        WITH RECURSIVE reach(path) AS (
          SELECT unnest(seeds)
          UNION
          SELECT e.importer FROM reach r JOIN design_edges() e ON e.imported = r.path
        )
        SELECT path FROM reach
    """


class DesignEdges(Relation):
    """The module dependencies of the code's design: one row per importing statement and the module
    it depends on, leaving out what a reader of the design does not count.

    `from package import name` depends on the module defining `name`, found through the
    initializers that re-export it, not on the facade; an `import` under `if TYPE_CHECKING:` is
    left out, since breaking an import cycle is what it is for.
    """

    stored = True
    query = r"""
        WITH RECURSIVE statements AS (
          SELECT i.path AS importer, i.line, i.shape, i.module, i.name, home.dir
          FROM module_imports() i JOIN module_files() home ON home.path = i.path
          WHERE i.module IS NOT NULL AND NOT i.type_only
        ), modules AS (
          SELECT s.importer, s.line, m.path AS imported
          FROM statements s
          JOIN module_files() m ON (m.packaged OR m.dir = s.dir) AND m.module =
            CASE WHEN s.shape = 'import' OR s.name = '*' THEN s.module
                 ELSE s.module || '.' || s.name END
        ), members AS (
          SELECT s.importer, s.line, base.path AS file, s.name
          FROM statements s
          JOIN module_files() base
            ON base.module = s.module AND (base.packaged OR base.dir = s.dir)
          WHERE s.shape = 'from' AND s.name <> '*'
            AND NOT EXISTS (
              SELECT 1 FROM module_files() sub
              WHERE sub.module = s.module || '.' || s.name AND (sub.packaged OR sub.dir = s.dir))
        ), chase(importer, line, asked, file, name, depth) AS (
          SELECT importer, line, name, file, name, 0 FROM members
          UNION
          SELECT c.importer, c.line, c.asked, b.imported, b.member, c.depth + 1
          FROM chase c
          JOIN module_bindings() b ON b.path = c.file AND b.local = c.name
            AND b.member IS NOT NULL AND b.imported IS NOT NULL
          WHERE c.depth < 8
        ), defined AS (
          SELECT importer, line, arg_max(file, depth) AS imported
          FROM chase GROUP BY importer, line, asked
        )
        SELECT DISTINCT importer, imported, line FROM (
          SELECT * FROM modules UNION ALL SELECT * FROM defined
        ) WHERE importer <> imported
    """


class ClassBases(Relation):
    """Each class (outside functions) with each base it spells, and the project class that base
    names when it resolves (through imports and the initializers that re-export it); NULL for a
    base from outside the project.
    """

    stored = True
    query = rf"""
        WITH RECURSIVE classes AS (
          SELECT c.path, c.node_id, c.start_line AS line, f.module, {_qualname("c")} AS qualname
          FROM python.ast_nodes c JOIN module_files() f ON f.path = c.path
          WHERE c.type = 'class_definition' AND c.scope.function IS NULL
        ), spelled AS (
          SELECT c.path, c.node_id AS class_id,
                 CASE b.type WHEN 'subscript' THEN s.name ELSE b.name END AS base
          FROM classes c
          JOIN python.ast_nodes a
            ON a.path = c.path AND a.parent_id = c.node_id AND a.type = 'argument_list'
          JOIN python.ast_nodes b
            ON b.path = a.path AND b.parent_id = a.node_id
           AND b.type IN ('identifier', 'attribute', 'subscript')
          LEFT JOIN python.ast_nodes s
            ON b.type = 'subscript' AND s.path = b.path AND s.parent_id = b.node_id
           AND s.sibling_index = 0
        ), wanted AS (
          SELECT path, class_id, base, path AS file, base AS name
          FROM spelled WHERE strpos(base, '.') = 0
          UNION ALL
          SELECT sp.path, sp.class_id, sp.base, b.imported, regexp_extract(sp.base, '[^.]+$')
          FROM spelled sp
          JOIN module_bindings() b
            ON b.path = sp.path AND b.local = regexp_replace(sp.base, '\.[^.]+$', '')
           AND b.member IS NULL AND b.imported IS NOT NULL
          WHERE strpos(sp.base, '.') > 0
        ), chase(path, class_id, base, file, name, depth) AS (
          SELECT path, class_id, base, file, name, 0 FROM wanted
          UNION
          SELECT c.path, c.class_id, c.base, b.imported, b.member, c.depth + 1
          FROM chase c
          JOIN module_bindings() b ON b.path = c.file AND b.local = c.name
            AND b.member IS NOT NULL AND b.imported IS NOT NULL
          WHERE c.depth < 8
        ), resolved AS (
          SELECT ch.path, ch.class_id, ch.base,
                 min_by(d.module, ch.depth) AS target_module,
                 min_by(d.qualname, ch.depth) AS target
          FROM chase ch JOIN classes d ON d.path = ch.file AND d.qualname = ch.name
          GROUP BY ch.path, ch.class_id, ch.base
        )
        SELECT c.path, c.line, c.module, c.qualname, s.base, r.target_module, r.target
        FROM classes c
        JOIN spelled s ON s.path = c.path AND s.class_id = c.node_id
        LEFT JOIN resolved r ON r.path = s.path AND r.class_id = s.class_id AND r.base = s.base
    """
