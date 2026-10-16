"""Rules over a manuscript's sentences, paragraphs and headings (mcmr's ALL-MANU0010 to 0012)."""

from pydantic import model_validator

from ...catalog import Rule

_OPENERS = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "who",
    "is",
    "are",
    "does",
    "do",
    "can",
    "should",
    "must",
    "will",
)


class SentenceLengthOutsideHouseBounds(Rule, id="ALL-MANU0010"):
    """Prose sentences shorter or longer than the house word-count bounds."""

    minimum_words: int = 10
    maximum_words: int = 48

    @model_validator(mode="after")
    def ordered(self) -> SentenceLengthOutsideHouseBounds:
        if self.minimum_words > self.maximum_words:
            raise ValueError("minimum_words must not exceed maximum_words")
        return self

    query = r"""
        SELECT s.path, s.line AS start_line, 0 AS start_column, s.word_count,
               CASE WHEN s.index = 0 THEN 'a paragraph-opening sentence' ELSE 'a sentence' END
                 AS sentence, left(s.text, 60) AS opening
        FROM manuscript_sentences s
        LEFT JOIN manuscript_paragraphs p
          ON p.fact_id = s.fact_id AND p.reading_order = s.reading_order
        WHERE (s.word_count < $minimum_words OR s.word_count > $maximum_words)
          AND NOT coalesce(p.in_cells OR p.in_float OR p.in_bibliography, false)
          AND regexp_matches(s.text, '\p{L}')
    """
    message = "{sentence} of {word_count} words opens `{opening}`"


class ParagraphLongerThanAReaderHolds(Rule, id="ALL-MANU0011"):
    """Prose paragraphs over the word or sentence ceiling."""

    maximum_words: int = 200
    maximum_sentences: int = 12

    query = """
        SELECT path, line AS start_line, 0 AS start_column, word_count, sentence_count
        FROM manuscript_paragraphs
        WHERE NOT in_cells AND NOT in_float
          AND (word_count > $maximum_words OR sentence_count > $maximum_sentences)
    """
    message = "a paragraph of {word_count} words over {sentence_count} sentences"


class SectionTitleThatIsNotANounPhrase(Rule, id="ALL-MANU0012"):
    """Section headings that ask a question, open with a verb or run past a short name."""

    openers: tuple[str, ...] = _OPENERS
    maximum_words: int = 6

    query = """
        WITH verdicts AS (
          SELECT *, ends_with(title, '?') AS asks,
                 list_contains(list_transform($openers::VARCHAR[], lambda w: lower(w)),
                               str_split(lower(title), ' ')[1]) AS answers
          FROM manuscript_sections)
        SELECT path, line AS start_line, 0 AS start_column, title,
               CASE WHEN asks THEN 'asks a question' WHEN answers THEN 'answers one'
                    ELSE 'reads as a sentence' END AS verdict
        FROM verdicts WHERE asks OR answers OR title_word_count > $maximum_words
    """
    message = "the heading `{title}` {verdict}"
