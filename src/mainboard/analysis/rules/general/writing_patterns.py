"""Rules over prose patterns (mcmr's ALL-WRIT0005)."""

from ...catalog import Rule
from ...facts import Setting


class SentenceOpenerConcentration(Rule, id="ALL-WRIT0005"):
    """Sections where one word opens more than `maximum` percent of the sentences."""

    maximum: float = 30
    minimum_sentences: int = 6
    ignored_openers: tuple[str, ...] = ("a", "i")

    query = """
        WITH commonest AS (
          SELECT fact_id, parent_id, lower(string_value) AS opener, count(*) AS opener_count,
                 sum(count(*)) OVER (PARTITION BY fact_id, parent_id) AS sentence_count
          FROM prose_segment_values
          WHERE relation = 'sections.sentence_openers'
            AND NOT list_contains($ignored::VARCHAR[], lower(string_value))
          GROUP BY fact_id, parent_id, lower(string_value)
          QUALIFY sentence_count >= $minimum_sentences
              AND row_number() OVER (PARTITION BY fact_id, parent_id
                                     ORDER BY count(*) DESC, lower(string_value)) = 1
        ), shared AS (
          SELECT *, max(opener_count / sentence_count * 100.0) OVER (PARTITION BY fact_id) AS peak
          FROM commonest)
        SELECT f.path, f.start_line, f.start_column, c.opener_count, c.sentence_count, c.opener
        FROM shared c JOIN prose_segment_facts f USING (fact_id)
        WHERE c.peak > $maximum
    """
    message = (
        "{opener_count} of the {sentence_count} sentences in one section open with `{opener}`"
    )

    def parameters(self) -> dict[str, Setting]:
        ignored = [word.casefold() for word in self.ignored_openers]
        return super().parameters() | {"ignored": ignored}
