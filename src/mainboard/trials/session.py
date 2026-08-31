# ONE RUN AND ONE TRIAL, WHICH IS EVERYTHING THE HOOKS IN `pytest_plugin` STAND ON.
#
# A `Session` is the run: its identity, the provenance stamped on every row it writes, the store
# each claim's receipts land in, and the baseline of every tracked flag. A `Trial` is one row of
# it, and the whole of what a lane touches.
#
# A LANE SUPPLIES MEASUREMENTS AND NOTHING ELSE. Who ran, on what card, at which commit, under
# which trial of which claim is already here, because every one of those is derivable and a fact a
# test has to retype is a fact a test will eventually retype wrong.
#
# AND THE FLAG COLUMN SAYS WHICH QUESTION IT ANSWERS. `session_<flag>` is what the flag read when
# the run opened and is a fact about the SESSION. It is not a fact about the reading, since a lane
# that moves a flag has readings on both sides of it, so `<flag>` beside it is the LIVE value read
# at the instant the trial settled. A review of the reference found 24 of 40 rows of one claim
# carrying a policy their own reading was not taken under, purely because only the session-level
# value existed. A lane measuring under two policies in one trial cannot be answered by one column
# either way and carries the policy observed beside each reading inside `measured`.
#
# A RUN IS IDENTIFIED BY 128 BITS AND ORDERED BY A COORDINATE IT WRITES DOWN. The identity used to
# be a second-resolution timestamp and eight hex characters, which is 32 bits of collision room
# under a name every reader also SORTED by, so two runs inside one second were ordered by their
# random suffix and a `newest` was whichever tail happened to sort higher. Those are two jobs and
# they are now two facts: `run` is a uuid7 under a readable timestamp and is IDENTITY, while
# `opened_at_ns` is the creation coordinate every recency question is answered from.
#
# AND `case_id` IS THE FIELD THAT USED TO LIE. It held the last component of the pytest node id
# under the name `run_id`, beside a `run` column that already held the actual run, so a reader
# joining receipts on `run_id` joined them on the test case. The value was always right and the
# name was always wrong; the column is now spelled for what it holds, and the full node id stays
# in `trial` where it always was.

from datetime import UTC, datetime
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING
from uuid import uuid7

from pydantic import JsonValue

from .coverage import PROBED, Cell, LaneStatus, Probed
from .dataset import ADMISSIBILITY, LEDGER, OPENED, PARTIAL
from .flags import moved, reading
from .provenance import Admissibility, Preflight, digested
from .stage import Stage
from .vocabulary import Outcome

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pytest

    from .declaration import Declaration
    from .ledger import TrialReceipts

# The report property a settled word rides to the terminal on. `user_properties` is pytest's own
# typed channel from an item to its report, which is exactly this trip, so nothing here has to
# hang an attribute off a report object and hope every consumer of it tolerates the extra field.
WORD = "mainboard_trials_word"


def params_of(item: pytest.Item) -> dict[str, JsonValue]:
    """A trial's own parametrize values as text, empty for a lane that takes no grid.

    Text because a receipt column has to be comparable across runs and a parametrize value is
    whatever object the grid held, which may not survive a round trip through parquet at all.
    """
    drawn = getattr(item, "callspec", None)
    return {name: str(value) for name, value in getattr(drawn, "params", {}).items()}


def lane_of(item: pytest.Item) -> tuple[str, str]:
    """A trial's lane and its key, the node id split at the parametrize bracket."""
    lane, _, key = item.nodeid.partition("[")
    return lane, key.removesuffix("]")


class Session:
    """One run of a declared universe: its identity, its provenance and its open stores.

    THE IDENTITY IS TWO FACTS AND THEY ARE NOT INTERCHANGEABLE. `run` names this run and nothing
    else: a uuid7 under the readable timestamp its partition directory is found by, 128 bits where
    there used to be 32, still lexically time-ordered because a uuid7 opens with its own
    millisecond. `opened` is the creation COORDINATE, in nanoseconds, and is what every recency
    question is answered from, because inferring an order from a name that ends in random hex is
    inferring it from the random hex.

    declared: what the consumer stated about its trials.
    """

    def __init__(self, declared: Declaration) -> None:
        self.declared = declared
        self.opened = time_ns()
        opened = datetime.fromtimestamp(self.opened // 1_000_000_000, UTC)
        self.run = f"{opened:%Y%m%dT%H%M%SZ}-{uuid7().hex}"
        self.taken = Preflight(
            declared.universe.root, declared.tree, probed=declared.universe.probed
        )
        self.baseline = reading(declared.flags)
        self.common: dict[str, JsonValue] = {
            **self.taken.stamp,
            OPENED: self.opened,
            **{f"session_{name}": value for name, value in self.baseline.items()},
        }
        self.writers: dict[str, TrialReceipts] = {}
        self.lanes: tuple[LaneStatus, ...] = ()
        self.leaked: dict[str, str] = {}
        self.staged = Stage("", resident=declared.resident)

    @property
    def card(self) -> str:
        """The device this run measures on, empty on a host that carries none."""
        return str(self.common.get("card", ""))

    @property
    def heading(self) -> str:
        """The line a session opens with, naming the machine and the tree its rows are scoped to.

        A run on a tree nobody can identify SAYS SO HERE, in the same line that names the card,
        because the rows it is about to write will not count toward any claim and a person who
        learns that from a coverage table three days later has already spent the card time.
        """
        named = str(self.common.get("card_name", "")) or "no card"
        where = f"{named}{f' ({self.card})' if self.card else ''}"
        stance = self.taken.admissibility
        if stance is Admissibility.ADMISSIBLE:
            return f"evidence on {where}:"
        return f"evidence on {where}, INADMISSIBLE ({stance}), these rows are scratch work:"

    def cell(self, params: Mapping[str, JsonValue]) -> Cell:
        """Where a trial sits on the declared axes, and why each axis reads what it does.

        An axis is read off the trial's own parameters when it names one and off the run's probed
        provenance otherwise, which is one rule covering both kinds: `model` comes from a
        parametrize grid and `card` from the machine, and neither is special-cased anywhere. The
        outcome rides beside the value, taken from the probe that produced it where there was one
        and `unasked` where nothing was ever asked, so an axis a lane simply does not use never
        looks like a machine nobody could identify.

        params: the trial's own parametrize values.
        """
        values, probing = {}, {}
        for axis in self.declared.universe.axes:
            drawn = params.get(axis)
            values[axis] = str(drawn) if drawn else str(self.common.get(axis, "") or "")
            outcome = str(self.common.get(f"{axis}{PROBED}", "")) if drawn is None else ""
            probing[axis] = Probed(outcome) if outcome else Probed.UNASKED
        return Cell(values=values, probing=probing)

    def close(self) -> str:
        """Release, compact, remint each store's readable ledger, then say why the run must fail.

        Compaction runs unconditionally, because a run that moved a flag still took every reading
        it took and the fragments are worth exactly as much either way.

        THE LEDGER IS REMINTED HERE SO IT IS NEVER OLDER THAN THE STORE IT SITS IN, BUT ONLY WHEN
        THIS RUN COVERS EVERY LANE THE STORE HAS EVER KNOWN. `latest.jsonl` is the one file in a
        receipts directory a person can open, and nothing wrote it after the store became parquet,
        so four universes of one workspace were found on 2026-08-29 handing a reader a generation
        their own coverage rule had superseded. Reminting unconditionally traded that defect for a
        second one: a run that only recollected some of a claim's lanes would remint the ledger
        from its own rows alone and every lane it did not touch would vanish from the one file a
        person reads, though the store underneath still held it. A partial run still lands as its
        own run and is still admissible evidence; it is written out beside the ledger instead of
        replacing it, so nothing this session measured goes unseen and nothing it did not measure
        is reported missing. `Dataset.retire` carries the ledger on its own path, which between it
        and this is every way the current view can move.
        """
        self.staged.drop()
        for node, writer in self.writers.items():
            writer.compact()
            store = self.declared.universe.dataset(node)
            if store.full(self.run):
                store.as_jsonl(store.root / LEDGER)
            else:
                store.as_jsonl(store.root / PARTIAL.format(self.run), self.run)
        drifted = moved(self.declared.flags, self.baseline)
        if not drifted:
            return ""
        lines = [
            f"  {name}: opened at {self.baseline[name]!r}, ended at {value!r}, first moved by "
            f"{self.leaked.get(name, 'a trial that settled no receipt')}"
            for name, value in drifted.items()
        ]
        return "\n".join(
            [
                f"trials REFUSE this session: {len(drifted)} tracked flag(s) ended off baseline, "
                "so every trial collected after the move measured a machine nobody can identify.",
                *lines,
                "Move a tracked flag only inside `mainboard.trials.held(...)`, which writes it "
                "back on the way out.",
            ]
        )

    def enter(self, node: str) -> None:
        """Make `node` the claim now running, releasing whatever the previous one held."""
        if node == self.staged.claim:
            return
        self.staged.drop()
        self.staged = Stage(node, resident=self.declared.resident)

    def trial(self, item: pytest.Item) -> Trial:
        """The evidence line for one collected trial, the claim it belongs to now open.

        Entering the claim here rather than in a fixture of its own is what makes the residency
        scope real: every lane that measures asks for its evidence line, so no claim can start
        without the previous one's holdings having been dropped first.
        """
        self.enter(self.declared.universe.node_of(Path(str(item.path))))
        return Trial(item, self)

    def writer(self, node: str) -> TrialReceipts:
        """One claim's store for this run, opened on first use and compacted at teardown.

        The claim's own registered rows are digested HERE rather than per trial, because
        `baselines/` is a fact about the claim and every receipt of it is scored against the same
        directory. A gate is only pre-registered if the rows it reads existed before the reading,
        and this is what lets a reader check that instead of taking it on trust.
        """
        if node not in self.writers:
            self.writers[node] = self.declared.universe.dataset(node).writer(
                self.run,
                {
                    "node": node,
                    "producer": "mainboard.trials",
                    "baselines_digest": self.taken.baselines(node),
                    **self.common,
                },
            )
        return self.writers[node]


class Trial:
    """One trial's evidence line: derived identity, host and commit, plus what the lane measured.

    A lane settles ONCE, with one of its workspace's declared words and its readings. A trial that
    settles nothing settles `failed` at teardown, so a broken instrument leaves a row rather than
    a hole, and the trial itself fails because a silent instrument is not a result.

    item: the running test. session: the run this row belongs to.
    """

    def __init__(self, item: pytest.Item, session: Session) -> None:
        self.item = item
        self.session = session
        self.lane, self.key = lane_of(item)
        self.settled = ""
        self.gated = ""

    def __getattr__(self, name: str) -> Callable[..., None]:
        """One declared word as a method, so a lane calls `trial.validated(...)` and reads well.

        The words are the consumer's, so the methods are too, and there is no table of them here
        to fall out of step with the vocabulary. An undeclared word refuses at the attribute
        rather than settling a row nothing can read.

        name: the word being reached for.
        """
        words = self.session.declared.words
        if name not in words:
            raise AttributeError(
                f"{name!r} is not a declared settle word; declared: {words.names}"
            )

        def settle(reason: str = "", **measured: JsonValue) -> None:
            self.settle(name, reason=reason, **measured)

        return settle

    def gate(self, registration: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Take the committed row this trial is scored against, digesting it on the way through.

        A LANE READS ITS GATE AND THE RECEIPT NEVER SAW IT. A claim registers an interval, a lane
        selects the row holding it and settles on whether today's reading falls inside, and the
        row that decided the verdict left no trace on the row that recorded it, so a reader could
        not tell a pre-registered gate from one edited into agreement afterwards. Reading the
        registration THROUGH here is what closes that: the digest rides on the receipt as
        `gate_digest` and the lane retypes nothing, since the row itself comes straight back.

        registration: the committed baseline row the verdict is taken against.
        """
        self.gated = digested(dict(registration))
        return registration

    def record(
        self,
        word: str,
        *,
        reason: str,
        measured: Mapping[str, JsonValue],
        outcome: Outcome,
    ) -> None:
        """Write this trial's one fragment and tell the terminal which word to print for it.

        Every tracked flag is read HERE rather than carried from the session baseline, so a lane
        that moved one names what was actually in force when it settled, and a value that differs
        from the baseline records this trial as the first suspect for the end-of-run check.

        ADMISSIBILITY IS PER ROW BECAUSE TRACKEDNESS IS PER LANE. A clean tree can still collect a
        lane nobody committed, and that row names a commit which does not contain the test that
        produced it, so the run-wide answer alone would call it evidence.
        """
        params = params_of(self.item)
        path = Path(str(self.item.path))
        live = reading(self.session.declared.flags)
        for name, value in live.items():
            if value != self.session.baseline[name]:
                self.session.leaked.setdefault(name, self.item.nodeid)
        self.settled = word
        self.item.user_properties.append((WORD, word))
        self.session.writer(self.session.declared.universe.node_of(path)).write(
            {
                "lane": self.lane,
                "key": self.key,
                "trial": self.item.nodeid,
                "case_id": self.item.nodeid.rpartition("::")[2],
                "kind": path.stem.removeprefix("test_"),
                ADMISSIBILITY: str(self.session.taken.admits(path)),
                "gate_digest": self.gated,
                **self.session.cell(params).filters,
                **live,
                "outcome": str(outcome),
                "verdict": word,
                "reason": reason,
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "params": params,
                "measured": dict(measured),
            }
        )

    def settle(self, word: str, *, reason: str = "", **measured: JsonValue) -> None:
        """Commit this trial under one of the declared words, with whatever it read.

        word: a word the consumer's own vocabulary declares.
        reason: one line saying what happened. measured: the readings behind it.
        """
        self.record(word, reason=reason, measured=measured, outcome=Outcome.PASSED)
