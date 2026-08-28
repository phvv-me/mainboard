# THE TWO WORDS A RECEIPT CARRIES, AND ONLY ONE OF THEM IS OURS.
#
# `Outcome` is fixed. It says whether the instrument worked, `passed` or `failed`, and it is the
# word `mainboard verdict` already branches an exit code on, so no consumer may redefine it.
#
# The settled word beside it is the consumer's whole vocabulary and this module knows nothing
# about what any of it means. A lab that settles `validated`, `refuted`, `known` and `abandoned`
# declares those four here; a lab that settles `held` and `broke` declares those two. The table
# carries a letter and a terminal markup per word because a progress line has to print something,
# and that is the entire extent of the opinion this module holds.
#
# A DEAD HYPOTHESIS IS A RESULT AND EXITS ZERO. Whatever word a trial settles on, its outcome is
# `passed` as long as the reading was actually taken, so the colour is the whole of the difference
# between the words and nobody learns to ignore a red line that only ever meant a prediction died.
# A trial that settled nothing at all is the one that failed, because that is the instrument
# breaking rather than a claim losing.

from enum import StrEnum, auto

from patos import FrozenModel


class Outcome(StrEnum):
    """Whether a trial's instrument worked, which is never the consumer's word to choose."""

    PASSED = auto()
    FAILED = auto()


class Stance(StrEnum):
    """What one settled word does to the prediction behind it, declared per word.

    A vocabulary of only confirmations and refutations forces every reading into one of the two,
    and a program whose subject is numeric noise then rounds an inconclusive separation into a
    decisive word. `neither` is the honest third position and it is the DEFAULT, so a consumer
    that never thinks about stance is never recorded as having claimed anything.
    """

    CONFIRMS = auto()
    REFUTES = auto()
    NEITHER = auto()


class Word(FrozenModel):
    """One settled word of a consumer's own vocabulary, and how a terminal prints it.

    name: the word itself, the value a receipt's `verdict` column carries.
    letter: the single character a progress line prints, the word's initial when empty.
    markup: the terminal markup the word is printed under, exactly the mapping pytest's own
        `pytest_report_teststatus` takes.
    stance: what settling on this word does to the prediction behind it. A word that narrates a
        quantity, abandons a line of attack, or reports a separation too small to decide is
        `neither`, and a reader tallying claims must be able to see that without knowing the
        consumer's spelling.
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
    """Every settled word a consumer declares, in the order a report prints them.

    words: the declared table, empty for a consumer that settles nothing but `passed`.
    """

    words: tuple[Word, ...] = ()

    def __contains__(self, name: str) -> bool:
        """Whether `name` is a word this vocabulary declares."""
        return name in self.names

    def __getitem__(self, name: str) -> Word:
        """One declared word, refusing an undeclared one by naming the whole table."""
        for word in self.words:
            if word.name == name:
                return word
        raise KeyError(f"{name!r} is not a declared settle word; declared: {self.names}")

    @property
    def names(self) -> tuple[str, ...]:
        """The declared words in order, which is what a tally and a refusal both list."""
        return tuple(word.name for word in self.words)

    @classmethod
    def of(cls, *names: str) -> Vocabulary:
        """A plain vocabulary from bare words, each printing its own initial and no markup.

        names: the settled words, in report order.
        """
        return cls(words=tuple(Word(name=name) for name in names))

    def stanced(self, stance: Stance) -> tuple[str, ...]:
        """The declared words taking one stance, so a tally can group without knowing spellings.

        stance: which position to collect, `neither` being every word that decides nothing.
        """
        return tuple(word.name for word in self.words if word.stance is stance)
