"""Rules over the order a manuscript is read in (mcmr's ALL-MANU0001 to ALL-MANU0003)."""

from collections.abc import Sequence

from .....catalog import Rule

_HELD = (
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "remark",
    "example",
    "section",
    "equation",
    "figure",
    "table",
    "algorithm",
    "proof",
)


class ForwardReferenceToUnreadMaterial(Rule, id="ALL-MANU0001"):
    """References sending a reader to something they have not read yet."""

    kinds: Sequence[str] = _HELD
    marked_commands: Sequence[str] = ()

    query = """
        WITH labels AS (
          SELECT fact_id, name AS target, arg_min(reading_order, ordinal) AS target_order,
                 arg_min(kind, ordinal) AS target_kind, arg_min(path, ordinal) AS target_path,
                 arg_min(line, ordinal) AS target_line
          FROM manuscript_labels GROUP BY fact_id, name)
        SELECT r.path, r.line AS start_line, 0 AS start_column, r.command, r.target,
               l.target_kind, l.target_path, l.target_line
        FROM manuscript_references r JOIN labels l USING (fact_id, target)
        WHERE l.target_order > r.reading_order
          AND len(list_filter($kinds::VARCHAR[], lambda k: starts_with(l.target_kind, k))) > 0
          AND NOT (r.command = 'proofref' AND l.target_kind = 'proof')
          AND NOT list_contains($marked_commands::VARCHAR[], r.command)
    """
    message = (
        "`\\{command}{{{target}}}` sends the reader to a {target_kind} first read at "
        "`{target_path}:{target_line}`"
    )


class SymbolUsedBeforeItIsIntroduced(Rule, id="ALL-MANU0002"):
    """Symbols a reader meets before the manuscript says what they are."""

    minimum_uses: int = 3
    ignored: Sequence[str] = ("i", "j", "k", "n", "x", "y", "z")

    query = """
        WITH introductions AS (
          SELECT fact_id, symbol AS name, min(reading_order) AS introduced_order
          FROM manuscript_notation_sites WHERE NOT is_local GROUP BY fact_id, symbol)
        SELECT s.path, s.line AS start_line, 0 AS start_column, s.name,
               coalesce(s.free_use_count, s.use_count) AS use_count,
               CASE WHEN i.introduced_order IS NULL THEN 'nowhere'
                    ELSE 'only after the reader has met it' END AS introduced
        FROM manuscript_notation_symbols s LEFT JOIN introductions i USING (fact_id, name)
        WHERE coalesce(s.free_use_count, s.use_count) >= $minimum_uses
          AND NOT list_contains($ignored::VARCHAR[], s.name)
          AND NOT EXISTS (SELECT 1 FROM manuscript_notation_entries e
                          WHERE e.fact_id = s.fact_id AND e.symbol = s.name)
          AND (i.introduced_order IS NULL
               OR coalesce(s.first_free_order, s.first_order) < i.introduced_order)
    """
    message = "`{name}` is used {use_count} times and is introduced {introduced}"


class TermUsedBeforeItIsMarked(Rule, id="ALL-MANU0003"):
    """Terms a reader meets in prose before the manuscript marks them."""

    marking_commands: Sequence[str] = ("term",)
    minimum_uses: int = 2

    query = """
        SELECT path, line AS start_line, 0 AS start_column, term, command, use_count
        FROM manuscript_notation_terms
        WHERE list_contains($marking_commands::VARCHAR[], command) AND use_count >= $minimum_uses
          AND first_use_order > 0 AND first_use_order < mark_order
    """
    message = "`{term}` is used before `\\{command}` marks it, {use_count} uses in all"
