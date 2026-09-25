# The two words a receipt carries, and only one of them is ours.
#
# `Outcome` is fixed: whether the instrument worked, the word `mainboard verdict` branches an exit
# code on. The settled word beside it is the consumer's vocabulary, whose meaning this module does
# not know; each word carries only the letter and terminal markup a progress line prints.
#
# A dead hypothesis is a result and exits zero: any settled word is `passed` as long as the
# reading was taken, so colour is the only difference between words and nobody learns to ignore a
# red line. A trial that settled nothing failed, because that is the instrument breaking.

from enum import StrEnum, auto

from patos import FrozenModel


class Outcome(StrEnum):
    """Whether a trial's instrument worked, which is never the consumer's word to choose."""

    PASSED = auto()
    FAILED = auto()


class Stance(StrEnum):
    """What one settled word does to the prediction behind it, declared per word.

    `neither` is the default, so a consumer that never thinks about stance is never recorded as
    having claimed anything, and an inconclusive separation is not rounded into a decisive word.
    """

    CONFIRMS = auto()
    REFUTES = auto()
    NEITHER = auto()


class Word(FrozenModel):
    """One settled word of a consumer's own vocabulary, and how a terminal prints it.

    name: the value a receipt's `verdict` column carries.
    letter: the progress-line character, the word's initial when empty.
    markup: the terminal markup, exactly the mapping `pytest_report_teststatus` takes.
    stance: what settling on this word does to the prediction, so a tally can group words without
        knowing the consumer's spelling.
    """

    name: str
    letter: str = ""
    markup: dict[str, bool] = {}
    stance: Stance = Stance.NEITHER

    @property
    def mark(self) -> str:
        """The progress character, the declared letter or the word's own initial."""
        return self.letter or self.name[:1].upper()


class Vocabulary(FrozenModel):
    """Every settled word a consumer declares, in the order a report prints them."""

    words: tuple[Word, ...] = ()

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __getitem__(self, name: str) -> Word:
        """One declared word, refusing an undeclared one by naming the whole table."""
        for word in self.words:
            if word.name == name:
                return word
        raise KeyError(f"{name!r} is not a declared settle word; declared: {self.names}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(word.name for word in self.words)

    @classmethod
    def of(cls, *names: str) -> Vocabulary:
        """A plain vocabulary from bare words in report order, each printing its own initial."""
        return cls(words=tuple(Word(name=name) for name in names))

    def stanced(self, stance: Stance) -> tuple[str, ...]:
        """The declared words taking one stance."""
        return tuple(word.name for word in self.words if word.stance is stance)
