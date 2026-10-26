"""Rules over the evidence a manuscript's numbers rest on (mcmr's ALL-MANU0013 to ALL-MANU0015)."""

from ....catalog import Rule


class ProseNumberThatNearlyMatchesItsTable(Rule, id="ALL-MANU0013"):
    """Prose numbers within a few percent of, but not equal to, a cell of a table they cite."""

    tolerance: float = 0.02
    minimum_digits: int = 4

    query = r"""
        WITH numbers AS (
          SELECT *, CASE WHEN regexp_matches(plain, '^[0-9]*\.?[0-9]*$')
                         THEN CAST(plain AS DOUBLE) END AS stated
          FROM (SELECT *, replace(literal, ',', '') AS plain FROM manuscript_numbers()
                WHERE contains(literal, '.') AND digits >= $minimum_digits)
          WHERE stated IS NOT NULL AND stated <> 0.0
        ), reachable AS (
          SELECT DISTINCT c.fact_id, c.section_number, n.stated AS held
          FROM manuscript_evidence_references c
          JOIN numbers n ON n.fact_id = c.fact_id AND n.float_label = c.target
          WHERE n.in_cells AND length(n.float_label) > 0
        ), near AS (
          SELECT n.path, n.line AS start_line, n.literal, arg_min(r.held, gap) AS held
          FROM numbers n JOIN reachable r USING (fact_id, section_number),
               LATERAL (SELECT abs(n.stated - r.held) / abs(n.stated) AS gap)
          WHERE NOT n.in_cells AND gap > 0.0 AND gap <= $tolerance
          GROUP BY n.fact_id, n.record_id, n.path, n.line, n.literal)
        SELECT path, start_line, 0 AS start_column, literal,
               round(held * scale, 0) / scale AS held
        FROM (SELECT *, 10.0 ** (5 - floor(log10(abs(held)))) AS scale FROM near)
    """
    message = "prose states `{literal}` where the nearest referenced cell reads `{held}`"


class RatioPublishedWithoutItsParts(Rule, id="ALL-MANU0014"):
    """Numbers named as a ratio or rate beside too few numbers to recompute it."""

    minimum_digits: int = 3
    minimum_parts: int = 3

    query = """
        SELECT path, line AS start_line, 0 AS start_column, literal,
               sentence_number_count - 1 AS others
        FROM manuscript_numbers()
        WHERE NOT in_cells AND NOT is_mathematical AND NOT has_ratio_basis AND names_ratio
          AND digits >= $minimum_digits AND sentence_number_count < $minimum_parts
    """
    message = "`{literal}` is named as a derived quantity beside {others} other numbers"


class MeasurementRestingOnAnUnpinnedCitation(Rule, id="ALL-MANU0015"):
    """Measurements whose supporting citation names no page or section."""

    reach: int = 4
    minimum_digits: int = 3

    query = """
        SELECT c.path, c.line AS start_line, 0 AS start_column, c.key,
               arg_min(n.literal, n.ordinal) AS literal
        FROM manuscript_evidence_citations c
        JOIN manuscript_numbers() n
          ON n.fact_id = c.fact_id AND n.sentence_order = c.sentence_order
        WHERE length(c.pin) = 0 AND NOT n.in_cells AND NOT n.is_mathematical
          AND n.digits >= $minimum_digits AND abs(n.reading_order - c.reading_order) <= $reach
        GROUP BY c.fact_id, c.record_id, c.path, c.line, c.key
    """
    message = "`{key}` supports `{literal}` and names no page or section"
