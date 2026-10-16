"""Rules over a manuscript's notation index (mcmr's ALL-MANU0007 to ALL-MANU0009)."""

from ...catalog import Rule


class SymbolMissingFromTheNotationIndex(Rule, id="ALL-MANU0007"):
    """Symbols used across sections that a manuscript's notation index never lists."""

    minimum_sections: int = 2
    minimum_uses: int = 3

    query = """
        SELECT s.path, s.line AS start_line, 0 AS start_column, s.name,
               coalesce(s.free_section_count, s.section_count) AS section_count
        FROM manuscript_notation_symbols s
        WHERE coalesce(s.free_section_count, s.section_count) >= $minimum_sections
          AND coalesce(s.free_use_count, s.use_count) >= $minimum_uses
          AND EXISTS (SELECT 1 FROM manuscript_notation_entries e WHERE e.fact_id = s.fact_id)
          AND NOT EXISTS (SELECT 1 FROM manuscript_notation_entries e
                          WHERE e.fact_id = s.fact_id AND e.symbol = s.name)
    """
    message = "`{name}` crosses {section_count} sections and the notation index never lists it"


class NotationEntryAbsentFromTheBody(Rule, id="ALL-MANU0008"):
    """Notation index rows whose symbol the body never sets."""

    query = """
        SELECT path, line AS start_line, 0 AS start_column, symbol, left(meaning, 60) AS meaning
        FROM manuscript_notation_entries e
        WHERE NOT EXISTS (SELECT 1 FROM manuscript_notation_symbols s
                          WHERE s.fact_id = e.fact_id AND s.name = e.symbol)
    """
    message = "the notation index lists `{symbol}` as `{meaning}` and the body never sets it"


class SymbolIntroducedUnderTwoMeanings(Rule, id="ALL-MANU0009"):
    """Symbols introduced under two meanings the index does not separate into senses."""

    query = """
        WITH sites AS (
          SELECT fact_id, symbol, count(DISTINCT section_number) AS section_count,
                 count(DISTINCT meaning) AS meaning_count,
                 string_agg(DISTINCT meaning, ' | ' ORDER BY meaning) AS meanings,
                 arg_min(path, ordinal) AS path, arg_min(line, ordinal) AS start_line
          FROM manuscript_notation_sites WHERE NOT is_local AND meaning <> ''
          GROUP BY fact_id, symbol HAVING count(DISTINCT meaning) >= 2)
        SELECT s.path, s.start_line, 0 AS start_column, s.symbol, s.section_count, s.meanings,
               CASE WHEN s.section_count = 1 THEN 'section' ELSE 'sections' END AS noun
        FROM sites s
        WHERE coalesce((SELECT max(sense_count) FROM manuscript_notation_entries e
                        WHERE e.fact_id = s.fact_id AND e.symbol = s.symbol), 1) < 2
    """
    message = (
        "`{symbol}` has distinct declared roles in {section_count} {noun}: {meanings}; "
        "the index separates no senses"
    )
