"""Rules over repeated implementation (mcmr's ALL-DUPL source rules): methods written the same in
sibling classes. Copies of arbitrary blocks (clone detection) are not looked for."""

from ....catalog import Example, Rule


class RepeatedSiblingMethod(Rule, id="ALL-DUPL0001", applicability="review"):
    """Methods written identically in sibling classes of one project base, a concrete pull-up
    boundary: decorators, name, signature and statements match node for node once the docstring
    and comments are set aside. A placeholder body (empty, `pass`, `...`,
    `raise NotImplementedError`, `return NotImplemented`, as its leaves read joined by spaces) is
    not an implementation, and classes sharing no resolved project base are never compared, since
    no owner for the extraction can be proven. When the base already defines the method its own
    way, the copies belong in an intermediate class rather than the base.

    The fix pulls the first copy up and deletes them all, only where nothing can resolve
    differently: the base lacks the method, the base and every copy are module classes of one
    file (the body reads the same names there), and each sibling keeps another statement."""

    placeholder: str = r"^(pass|\.\.\.|raise NotImplementedError( \( \))?|return NotImplemented)?$"

    query = r"""
        WITH defs AS (
          SELECT DISTINCT m.method_id, m.class_id, m.name, m.path, m.start_line, m.start_column,
                 d.file_id, coalesce(d.wrapper_id, d.def_id) AS first_id,
                 d.def_id + d.size AS last_id, d.body_id, d.docstring IS NOT NULL AS documented,
                 d.start_line AS first_line, d.end_line AS last_line,
                 c.scope = 'module' AND k.statements >= 2 AS movable
          FROM class_methods m
          JOIN class_classes c ON c.class_id = m.class_id AND NOT c.is_test
          JOIN core_file f ON f.path = m.path
          JOIN core_def d ON d.file_id = f.file_id AND d.kind = 'function' AND d.name = m.name
            AND d.end_line = m.end_line AND d.end_column = m.end_column
          JOIN core_def k ON k.file_id = d.file_id AND k.def_id = d.class_id
        ), docstrings AS (
          SELECT d.method_id, s.node_id AS first_id, s.node_id + s.descendant_count AS last_id
          FROM defs d
          JOIN core_node s ON s.file_id = d.file_id AND s.parent_id = d.body_id
            AND s.semantic_type <> 'METADATA_COMMENT'
          WHERE d.documented
          QUALIFY s.sibling_index = min(s.sibling_index) OVER (PARTITION BY d.method_id)
        ), shapes AS (
          SELECT d.method_id,
                 string_agg(n.type || '=' || n.name, ' ' ORDER BY n.node_id) AS shape,
                 coalesce(string_agg(coalesce(nullif(n.name, ''), n.type), ' ' ORDER BY n.node_id)
                            FILTER (WHERE n.node_id > d.body_id AND n.children_count = 0), '')
                   AS body
          FROM defs d
          LEFT JOIN docstrings s USING (method_id)
          JOIN core_node n ON n.file_id = d.file_id
            AND n.node_id BETWEEN d.first_id AND d.last_id
            AND n.node_id NOT BETWEEN coalesce(s.first_id, -1) AND coalesce(s.last_id, -1)
            AND n.semantic_type <> 'METADATA_COMMENT'
          GROUP BY d.method_id
        ), grouped AS (
          SELECT d.*, e.parent AS base,
                 count(DISTINCT d.class_id) OVER copies AS siblings,
                 row_number() OVER copies AS rank,
                 list({'path': d.path, 'first_line': d.first_line, 'last_line': d.last_line,
                       'movable': d.movable}) OVER (copies ROWS BETWEEN UNBOUNDED PRECEDING
                                                   AND UNBOUNDED FOLLOWING) AS members
          FROM defs d
          JOIN shapes s USING (method_id)
          JOIN override_edges e ON e.child = d.class_id
          WHERE NOT regexp_matches(s.body, $placeholder)
          WINDOW copies AS (PARTITION BY e.parent, s.shape ORDER BY d.path, d.start_line)
        )
        SELECT *, CASE WHEN defined
                       THEN 'share it through an intermediate class, since `' || base
                            || '` defines its own'
                       ELSE 'pull it up into `' || base || '`' END AS remedy
        FROM (SELECT DISTINCT ON (g.method_id, g.base)
                     g.path, g.start_line, g.start_column, g.name, g.rank, g.members,
                     g.members[1].path || ':' || g.members[1].first_line AS first,
                     b.name AS base, b.path AS base_path, b.end_line AS base_end,
                     b.scope = 'module' AS base_movable,
                     EXISTS (SELECT 1 FROM class_methods x
                             WHERE x.class_id = g.base AND x.name = g.name) AS defined
              FROM grouped g JOIN class_classes b ON b.class_id = g.base
              WHERE g.siblings >= 2 AND g.rank > 1)
    """
    message = "`{name}` repeats `{first}` in another subclass of `{base}`; {remedy}"
    # A base declaring the method only as an `@abstractmethod` placeholder takes the first copy in
    # its place (`pull-up`), keeping the placeholder's docstring when the copy has none; a base
    # with a real definition of its own is left to an intermediate class, by hand.
    fix = """
        WITH pulled AS (
          SELECT f.finding_key, f.base_path AS path, f.base_end, f.members,
                 string_split(s.content, chr(10)) AS lines
          FROM finding f JOIN source_files s ON s.path = f.base_path
          WHERE f.rank = 2 AND NOT f.defined AND f.base_movable
            AND list_bool_and(list_transform(f.members,
                  lambda m: m.path = f.base_path AND m.movable))
        )
        SELECT finding_key, path, base_end AS start_line,
               strlen(rtrim(lines[base_end], chr(13))) AS start_column, base_end AS end_line,
               strlen(rtrim(lines[base_end], chr(13))) AS end_column,
               chr(10) || chr(10) || array_to_string(
                 lines[members[1].first_line:members[1].last_line], chr(10)) AS text
        FROM pulled
        UNION ALL BY NAME
        SELECT finding_key, path, functions_member_start(lines, m.first_line) AS start_line,
               0 AS start_column, m.last_line + 1 AS end_line, 0 AS end_column, '' AS text
        FROM (SELECT finding_key, path, lines, unnest(members) AS m FROM pulled)
        UNION ALL BY NAME
        SELECT f.finding_key, 'pull-up' AS operation, f.base_path AS path, s.content, f.base,
               f.name, f.members
        FROM finding f JOIN source_files s ON s.path = f.base_path
        WHERE f.rank = 2 AND f.defined AND f.base_movable
          AND list_bool_and(list_transform(f.members,
                lambda m: m.path = f.base_path AND m.movable))
    """
    examples = (
        Example(
            bad=(
                '''class Keyed:
    name: str


class Category(Keyed):
    kind = 'category'

    @property
    def key(self):
        """The category's key."""
        return self.name.lower().replace(' ', '-')


class Criterion(Keyed):
    kind = 'criterion'

    @property
    def key(self):
        # the criterion's key
        return self.name.lower().replace(' ', '-')
'''
            ),
            good=(
                '''class Keyed:
    name: str

    @property
    def key(self):
        """The category's key."""
        return self.name.lower().replace(' ', '-')


class Category(Keyed):
    kind = 'category'


class Criterion(Keyed):
    kind = 'criterion'
'''
            ),
        ),
        Example(
            bad=(
                """from abc import ABC, abstractmethod


class Edit(ABC):
    path: str

    @abstractmethod
    def file(self):
        \"\"\"The path the edit changes.\"\"\"


class Replace(Edit):
    path = "a"

    def file(self):
        return self.path


class Remove(Edit):
    path = "b"

    def file(self):
        return self.path
"""
            ),
            good=(
                """from abc import ABC


class Edit(ABC):
    path: str

    def file(self):
        \"\"\"The path the edit changes.\"\"\"
        return self.path


class Replace(Edit):
    path = "a"


class Remove(Edit):
    path = "b"
"""
            ),
        ),
    )
