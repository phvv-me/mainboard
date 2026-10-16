"""Rules over how uniform prose rhythm is (mcmr's ALL-WRIT0003 and ALL-WRIT0004)."""

from typing import ClassVar

from ...catalog import Rule
from ...facts import Setting


class _Uniformity(Rule):
    """Facts whose most uniform section has lengths within `maximum` percent of their mean: a
    subclass names the lengths (`relation`) and how many make a section (`minimum_entries`)."""

    relation: ClassVar[str]
    maximum: float = 90

    query = r"""
        WITH lengths AS (
          SELECT fact_id, parent_id, CAST(integer_value AS DOUBLE) AS length,
                 avg(integer_value) OVER section AS average,
                 count(*) OVER section AS entry_count
          FROM prose_segment_values
          WHERE relation = $relation AND integer_value >= $minimum_words
          WINDOW section AS (PARTITION BY fact_id, parent_id)
        ), greatest AS (
          SELECT fact_id,
                 max(greatest(0.0, 1.0 - deviation / average) * 100.0) AS value
          FROM (SELECT fact_id, parent_id, any_value(average) AS average,
                       avg(abs(length - average)) AS deviation
                FROM lengths WHERE entry_count >= $minimum_entries GROUP BY ALL)
          GROUP BY fact_id HAVING value > $maximum
        ), scaled AS (
          SELECT *, 10.0 ** (3 - floor(log10(abs(value)))) AS scale FROM greatest)
        SELECT f.path, f.start_line, f.start_column, f.fact_id,
               regexp_replace(CAST(round(value * scale, 0) / scale AS VARCHAR), '\.0$', '', 'g')
                 AS shown
        FROM scaled JOIN prose_segment_facts f USING (fact_id)
    """

    def minimum_entries(self) -> int:
        raise NotImplementedError

    def parameters(self) -> dict[str, Setting]:
        return super().parameters() | {
            "relation": self.relation,
            "minimum_entries": self.minimum_entries(),
        }


class SentenceLengthUniformity(_Uniformity, id="ALL-WRIT0003"):
    """Sections whose sentences are all about the same length."""

    relation = "sections.sentence_word_counts.root"
    minimum_sentences: int = 5
    minimum_words: int = 3

    message = "sentence length uniformity is {shown} percent for `{fact_id}`"

    def minimum_entries(self) -> int:
        return self.minimum_sentences


class ParagraphLengthUniformity(_Uniformity, id="ALL-WRIT0004"):
    """Sections whose paragraphs are all about the same length."""

    relation = "sections.paragraph_word_counts.root"
    minimum_paragraphs: int = 4
    minimum_words: int = 30

    message = "paragraph length uniformity is {shown} percent for `{fact_id}`"

    def minimum_entries(self) -> int:
        return self.minimum_paragraphs
