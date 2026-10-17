"""mcmr's AutomationTaskFact, derived from the root `mainboard.toml`'s syntax tree.

A task is a capability the manifest automates, declared under `[tasks]` or `[envs.<name>.tasks]`
as a string, an array of words, or a table whose `run` or `cmd` is one of those (a task with
neither runs its `depends`). A capability declared in several tables keeps every distinct command,
the default table's first. Documentation is the project's `.md`, `.rst` and `.txt` files that
Git's ignores keep (`project_guides`).
"""

from pathlib import Path

from ...catalog import Family, Function
from ...facts import Facts
from .project import ProjectTree


class ProjectHeads(Function):
    """The program each part of a shell `command` runs, with the parts joined by `&`, `|`, `;`
    or a newline taken apart."""

    command: str

    expression = r"""
        list_filter(list_transform(
          list_filter(regexp_split_to_array(command, '[&|;\n]'), lambda part: part <> ''),
          lambda part: regexp_extract(part, '^\s*(\S+)', 1)), lambda head: head <> '')"""


class ProjectWords(Function):
    """Every word of a shell `command`, the operators that join its parts dropped."""

    command: str

    expression = r"""
        flatten(list_transform(
          list_filter(regexp_split_to_array(command, '[&|;\n]'), lambda part: part <> ''),
          lambda part: list_filter(regexp_split_to_array(trim(part), '\s+'),
                                   lambda word: word <> '')))"""


class ProjectStaysInside(Function):
    """Whether `command` operates the checkout through its declared environment: it installs
    nothing, reaches no other host and names no absolute or home path."""

    command: str

    expression = r"""
        NOT list_has_any(project_heads(command),
            ['sudo', 'doas', 'su', 'ssh', 'scp', 'apt', 'apt-get', 'brew', 'dnf', 'yum', 'pacman',
             'apk', 'choco', 'winget', 'snap', 'curl', 'wget'])
        AND len(list_filter(project_heads(command),
                            lambda head: starts_with(head, '/') OR starts_with(head, '~'))) = 0
        AND NOT contains(command, '$HOME') AND NOT contains(command, '~/')"""


class ProjectRunsUnattended(Function):
    """Whether `command` completes with nobody at the terminal."""

    command: str

    expression = """
        NOT list_has_any(project_heads(command),
            ['vi', 'vim', 'nano', 'emacs', 'less', 'more', 'gdb', 'lldb'])
        AND NOT list_has_any(project_words(command),
            ['-it', '-ti', '--interactive', '--pdb', '--edit'])"""


class AutomationTask(Family):
    family = "AutomationTaskFact"
    verified = True
    relations = {
        "facts": r"""
            SELECT 0 AS fact_order, 'automation:mainboard' AS fact_id, 'mainboard.toml' AS path,
                   1 AS start_line, 0 AS start_column, 1 AS end_line, 0 AS end_column,
                   'python' AS language
            FROM ast_files WHERE path = 'mainboard.toml' AND language = 'toml'
        """,
        "declarations": r"""
            WITH RECURSIVE toml AS (
              SELECT n.* FROM ast_nodes n JOIN ast_files a USING (fact_id)
              WHERE a.path = 'mainboard.toml' AND a.language = 'toml'
            ), pairs AS (
              SELECT p.node_id, CASE WHEN t.type = 'table' THEN t.name || '.' || p.name
                                     ELSE p.name END AS key
              FROM toml p
              JOIN toml t ON t.node_id = p.parent_id AND t.type IN ('table', 'document')
              WHERE p.type = 'pair'
              UNION ALL
              SELECT c.node_id, f.key || '.' || c.name
              FROM pairs f
              JOIN toml i ON i.parent_id = f.node_id AND i.type = 'inline_table'
              JOIN toml c ON c.parent_id = i.node_id AND c.type = 'pair'
            ), parts AS (
              SELECT key, v.type, v.name AS text, v.node_id AS value
              FROM pairs p JOIN toml v ON v.parent_id = p.node_id AND v.sibling_index = 2
              UNION ALL
              SELECT name || '.', 'table', NULL, NULL FROM toml WHERE type = 'table'
            ), located AS (
              SELECT regexp_extract(key, '^(?:envs[.]([^.]+)[.])?tasks[.]([^.]+)(?:[.](.*))?$',
                                    ['env', 'capability', 'field']) AS part, type, text, value
              FROM parts
            )
            SELECT nullif(part.env, '') AS env, part.capability AS capability,
                   part.field AS field, type,
                   CASE WHEN type = 'string' THEN project_literal(text) END AS text,
                   CASE WHEN type = 'array' THEN
                     coalesce((SELECT list(project_literal(e.name) ORDER BY e.sibling_index)
                               FROM toml e WHERE e.parent_id = value AND e.type = 'string'),
                              []::VARCHAR[]) END AS words,
                   CASE WHEN type = 'array' THEN
                     (SELECT count(*) FROM toml e WHERE e.parent_id = value
                      AND e.type NOT IN ('[', ']', ',', 'comment', 'string')) END AS others
            FROM located WHERE part.capability <> ''
        """,
        "commands": r"""
            SELECT env, capability,
                   CASE type
                     WHEN 'string' THEN text
                     WHEN 'array' THEN CASE WHEN others = 0 THEN array_to_string(words, ' ') END
                     ELSE CASE run_type
                            WHEN 'string' THEN run_text
                            WHEN 'array' THEN CASE WHEN run_others = 0
                                                   THEN array_to_string(run_words, ' ') END
                            ELSE CASE WHEN depends_type = 'array' AND depends_others = 0
                                           AND len(depends_words) > 0
                                      THEN array_to_string(depends_words, ' && ') END
                          END
                   END AS command
            FROM (
              SELECT env, capability,
                     coalesce(any_value(type) FILTER (WHERE field = ''), 'table') AS type,
                     any_value(text) FILTER (WHERE field = '') AS text,
                     any_value(words) FILTER (WHERE field = '') AS words,
                     any_value(others) FILTER (WHERE field = '') AS others,
                     arg_min(type, field = 'cmd') FILTER (WHERE field IN ('run', 'cmd'))
                       AS run_type,
                     arg_min(text, field = 'cmd') FILTER (WHERE field IN ('run', 'cmd'))
                       AS run_text,
                     arg_min(words, field = 'cmd') FILTER (WHERE field IN ('run', 'cmd'))
                       AS run_words,
                     arg_min(others, field = 'cmd') FILTER (WHERE field IN ('run', 'cmd'))
                       AS run_others,
                     arg_min(type, field) FILTER (WHERE field LIKE 'depends%') AS depends_type,
                     arg_min(words, field) FILTER (WHERE field LIKE 'depends%') AS depends_words,
                     arg_min(others, field) FILTER (WHERE field LIKE 'depends%') AS depends_others
              FROM automation_task_declarations GROUP BY env, capability)
        """,
        "tasks": r"""
            WITH distinct_commands AS (
              SELECT DISTINCT capability, command FROM automation_task_commands
              WHERE command IS NOT NULL
            ), stated AS (
              SELECT c.capability, count(d.command) AS commands_length,
                     coalesce(bool_and(project_stays_inside(d.command)), true)
                       AS is_repository_owned,
                     coalesce(bool_and(project_runs_unattended(d.command)), true)
                       AS is_noninteractive
              FROM (SELECT DISTINCT capability FROM automation_task_commands) c
              LEFT JOIN distinct_commands d USING (capability)
              GROUP BY c.capability
            ), guides AS (
              SELECT content FROM project_guides
            )
            SELECT f.fact_id, f.fact_id AS parent_id,
                   f.fact_id || '/tasks:' || t.ordinal AS record_id, t.ordinal, t.capability,
                   t.commands_length, t.guidance_locations_length, t.is_repository_owned,
                   t.is_noninteractive
            FROM automation_task_facts f
            JOIN (SELECT s.*, row_number() OVER (ORDER BY s.capability) - 1 AS ordinal,
                         (SELECT count(*) FROM guides g
                          WHERE contains(g.content, 'mainboard run ' || s.capability)
                             OR (s.capability = 'setup'
                                 AND contains(g.content, 'mainboard install')))
                           AS guidance_locations_length
                  FROM stated s) t ON true
        """,
    }

    @classmethod
    def provide(cls, facts: Facts, root: Path) -> None:
        """Load the guides the tasks relation reads: `project_guides` (path, content)."""
        ProjectTree(root=root).register(facts)
        files = [
            str(root / entry)
            for (entry,) in facts.db.execute(
                "SELECT entry FROM project_entries "
                "WHERE NOT is_directory AND regexp_matches(entry, '[.](md|rst|txt)$')"
            ).fetchall()
        ]
        facts.db.execute("CREATE TABLE project_guides (path VARCHAR, content VARCHAR)")
        if files:
            facts.db.execute(
                "INSERT INTO project_guides SELECT substr(filename, length($root) + 2), content "
                "FROM read_text($files)",
                {"root": str(root), "files": files},
            )
