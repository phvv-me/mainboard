"""The SQL functions the readers of a manuscript's sites share: trimming, sentence cuts, the
subscript of a symbol and the commas inside brackets."""

from .....catalog import Function
from ...constants.manuscripts.core import SPACE_CHARS
from .constants import WORD


class ManuscriptTrimmed(Function):
    """The text without the whitespace around it, Unicode's as Rust reads it."""

    text: str

    expression = f"regexp_replace(text, '^[{SPACE_CHARS}]+|[{SPACE_CHARS}]+$', '', 'g')"


class ManuscriptLastSentence(Function):
    """The last sentence of a text, cut as `manuscript_ends_sentence` cuts: what follows the last
    sentence end when that is more than whitespace, else the sentence before."""

    text: str

    expression = (
        f"list_transform([regexp_extract_all(text, '{WORD}|[{SPACE_CHARS}]+')], "
        "lambda tokens: list_transform([list_filter(range(1, len(tokens) + 1), "
        "lambda i: manuscript_ends_sentence(tokens[i]))], lambda cuts: "
        "CASE WHEN len(cuts) = 0 THEN array_to_string(tokens, '') "
        "WHEN len(list_filter(list_slice(tokens, cuts[-1] + 1, len(tokens)), "
        f"lambda t: regexp_matches(t, '[^{SPACE_CHARS}]'))) > 0 "
        "THEN array_to_string(list_slice(tokens, cuts[-1] + 1, len(tokens)), '') "
        "ELSE array_to_string(list_slice(tokens, CASE WHEN len(cuts) > 1 THEN cuts[-2] + 1 "
        "ELSE 1 END, cuts[-1]), '') END)[1])[1]"
    )


class ManuscriptFirstSentence(Function):
    """The first sentence of a text, cut as `manuscript_ends_sentence` cuts; the whole text when
    it ends no sentence."""

    text: str

    expression = (
        f"list_transform([regexp_extract_all(text, '{WORD}|[{SPACE_CHARS}]+')], "
        "lambda tokens: list_transform([list_filter(range(1, len(tokens) + 1), "
        "lambda i: manuscript_ends_sentence(tokens[i]))[1]], lambda cut: "
        "CASE WHEN cut IS NULL THEN text "
        "ELSE array_to_string(list_slice(tokens, 1, cut), '') END)[1])[1]"
    )


class ManuscriptIndexBase(Function):
    """What stands before a one-letter subscript (`x` of `x_i` and `x_{i}`), NULL when the symbol
    has none."""

    name: str

    expression = r"nullif(regexp_extract(name, '^(.*)_(?:\{[A-Za-z]\}|[A-Za-z])$', 1), '')"


class ManuscriptIndexLetter(Function):
    """The one-letter subscript of a symbol (`i` of `x_i` and `x_{i}`), NULL when it has none."""

    name: str

    expression = (
        r"coalesce(nullif(regexp_extract(name, '^(.*)_(?:\{([A-Za-z])\}|([A-Za-z]))$', 2), ''), "
        r"nullif(regexp_extract(name, '^(.*)_(?:\{([A-Za-z])\}|([A-Za-z]))$', 3), ''))"
    )


class ManuscriptMaskCommas(Function):
    """The text with the commas inside brackets, braces and parentheses turned into `§`, so the
    commas that remain separate its top-level items."""

    text: str

    expression = (
        "array_to_string(list_transform(range(1, length(text) + 1), lambda i: "
        "CASE WHEN substr(text, i, 1) = ',' AND "
        "length(regexp_replace(substr(text, 1, i), '[^(\\[{]', '', 'g')) "
        "> length(regexp_replace(substr(text, 1, i), '[^)\\]}]', '', 'g')) "
        "THEN '§' ELSE substr(text, i, 1) END), '')"
    )
