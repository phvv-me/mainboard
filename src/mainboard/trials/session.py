# One run and one trial, which is everything the hooks in `pytest_plugin` stand on.
#
# A `Session` is the run: its identity, the provenance on every row, each claim's store and the
# baseline of every tracked flag. A `Trial` is one row, the whole of what a lane touches. A lane
# supplies measurements only; who ran, on what card, at which source and under which claim are
# derived, because a fact a test has to retype is a fact it will eventually retype wrong.
#
# `session_<flag>` is what a flag read when the run opened; `<flag>` beside it is the live value
# when the trial settled. A review found 24 of 40 rows of one claim carrying a policy their reading
# was not taken under, because only the session value existed. A lane measuring under two policies
# in one trial carries the observed policy beside each reading inside `measured`.
#
# `case_id` is the last component of the node id. It was once named `run_id` beside a `run` column
# holding the actual run, so joins on it joined on the test case; the full node id is in `trial`.

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING
from uuid import uuid7

from pydantic import JsonValue

from ..dispatch.provenance import registered
from .artifacts import Artifact, Artifacts
from .coverage import PROBED, Cell, LaneStatus, Probed
from .dataset import ADMISSIBILITY, LEDGER, OPENED, PARTIAL
from .flags import moved, reading
from .lease import CardLease
from .provenance import Admissibility, Preflight, digested, parsed
from .stage import Stage
from .vocabulary import Outcome

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pytest

    from .declaration import Declaration
    from .ledger import TrialReceipts

# The report property a settled word rides to the terminal on, through pytest's own typed channel
# from an item to its report.
WORD = "mainboard_trials_word"


def params_of(item: pytest.Item, axes: Sequence[str] = ()) -> dict[str, JsonValue]:
    """A trial's own coordinates as text, empty for a lane that takes no grid and no marker.

    Text because a parametrize value may not survive a parquet round trip. A declared axis a lane
    names by marker (`@pytest.mark.phase("2")`) is a coordinate too, so a registration's second
    phase of the same grid is a second cell rather than a re-run of the first.

    axes: the declared coverage axes a marker of the same name may answer.
    """
    drawn = getattr(item, "callspec", None)
    values: dict[str, JsonValue] = {
        name: str(value) for name, value in getattr(drawn, "params", {}).items()
    }
    for axis in axes:
        marker = item.get_closest_marker(axis)
        if axis not in values and marker is not None and marker.args:
            values[axis] = str(marker.args[0])
    return values


def lane_of(item: pytest.Item) -> tuple[str, str]:
    """A trial's lane and its key, the node id split at the parametrize bracket."""
    lane, _, key = item.nodeid.partition("[")
    return lane, key.removesuffix("]")


class Session:
    """One run of a declared universe: its identity, its provenance and its open stores.

    `run` is IDENTITY: a uuid7 under a readable timestamp, 128 bits where a second-resolution
    stamp and eight hex characters once gave 32, still lexically time-ordered. `opened` is the
    creation coordinate in nanoseconds and answers every recency question, since ordering on a name
    ending in random hex orders on the hex.
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
        self.manifests: dict[str, Artifact] = {}
        self.lanes: tuple[LaneStatus, ...] = ()
        self.leaked: dict[str, str] = {}
        self.staged: Stage = Stage("", resident=declared.resident)
        self.leased: CardLease | None = None

    @property
    def card(self) -> str:
        """The device this run measures on, empty on a host that carries none."""
        return str(self.common.get("card", ""))

    def claim(self) -> None:
        """Take this run's exclusive hold on its card; a no-op on a host with no card.

        Collection decides whether to call this, so a run touching no `gpu` lane is never blocked.
        """
        if self.card:
            self.leased = CardLease.acquire(self.declared.universe.root)

    @property
    def heading(self) -> str:
        """The line a session opens with, naming the machine and whether its rows are evidence.

        An inadmissible tree says so here, before the card time is spent, not in a coverage table
        three days later.
        """
        named = str(self.common.get("card_name", "")) or "no card"
        where = f"{named}{f' ({self.card})' if self.card else ''}"
        stance = self.taken.admissibility
        if stance is Admissibility.ADMISSIBLE:
            return f"evidence on {where}:"
        return f"evidence on {where}, INADMISSIBLE ({stance}), these rows are scratch work:"

    def cell(self, params: Mapping[str, JsonValue]) -> Cell:
        """Where a trial sits on the declared axes, and why each axis reads what it does.

        An axis reads the trial's own parameter when it names one, else the run's probed
        provenance, with that probe's outcome; `unasked` where nothing asked, so an axis a lane
        does not use never looks like an unidentifiable machine.
        """
        values, probing = {}, {}
        for axis in self.declared.universe.axes:
            drawn = params.get(axis)
            values[axis] = str(drawn) if drawn else str(self.common.get(axis, "") or "")
            outcome = str(self.common.get(f"{axis}{PROBED}", "")) if drawn is None else ""
            probing[axis] = Probed(outcome) if outcome else Probed.UNASKED
        return Cell(values=values, probing=probing)

    def close(self) -> str:
        """Release the claim and card, remint each ledger, then return what must fail.

        The last claim's residue is returned, not raised: when `Stage.drop` escaped from
        `pytest_sessionfinish`, the lease stayed held, nothing was reminted and pytest never
        printed the failures behind it (on 2026-08-31 a GH200 wave read "did not release" over
        twelve hidden `ZeroDivisionError`s). The lease releases before anything else can raise.

        A store's `latest.jsonl` is reminted only when this run covers every lane the store has
        known (`Dataset.full`); a partial run lands beside it as `partial-<run>.jsonl`. Receipt
        fragments stay immutable, so a concurrent fetch never sees a rewritten part.
        """
        refusals = []
        try:
            self.staged.drop()
        except RuntimeError as residue:
            refusals.append(str(residue))
        if self.leased is not None:
            self.leased.release()
        for node in self.writers:
            store = self.declared.universe.dataset(node)
            if store.full(self.run):
                store.as_jsonl(store.root / LEDGER)
            else:
                store.as_jsonl(store.root / PARTIAL.format(self.run), self.run)
        drifted = moved(self.declared.flags, self.baseline)
        if not drifted:
            return "\n".join(refusals)
        lines = [
            f"  {name}: opened at {self.baseline[name]!r}, ended at {value!r}, first moved by "
            f"{self.leaked.get(name, 'a trial that settled no receipt')}"
            for name, value in drifted.items()
        ]
        return "\n".join(
            [
                *refusals,
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
        """The evidence line for one collected trial, its claim entered first.

        Every measuring lane asks for its evidence line, so no claim starts before the previous
        one's holdings are dropped.
        """
        self.enter(self.declared.universe.node_of(Path(str(item.path))))
        return Trial(item, self)

    def manifest(self, path: Path) -> Artifact | None:
        """One run manifest per node of an `experiments` universe, from the dispatch's listing.

        The adjacent `node.md` registration must be in the captured listing; no experiment keeps
        another source list or computes a second source seal.
        """
        universe = self.declared.universe
        if universe.root.name != "experiments":
            return None
        node = universe.node_of(path)
        if node in self.manifests:
            return self.manifests[node]
        source = self.taken.source
        if not source.digest or not source.closure:
            raise RuntimeError("research logging requires a captured Mainboard source bundle")
        registration = path.parent / "node.md"
        sources = parsed(Path(source.closure).read_text(encoding="utf-8"))
        registered(registration, sources, root=Path.cwd())
        manifest = {
            "run": self.run,
            "registration": registration.relative_to(Path.cwd()).as_posix(),
            "source": source.model_dump(mode="json", exclude={"closure"}),
            "files": [row.model_dump(mode="json") for row in sources],
            "environment": self.taken.versions,
            "inputs": {
                key: value.model_dump(mode="json") for key, value in self.declared.inputs.items()
            },
            "hardware": self.taken.card.model_dump(mode="json"),
            "arithmetic": self.baseline,
            "opened_at_ns": self.opened,
        }
        directory = universe.dataset(node).root.parent / "artifacts" / self.run
        self.manifests[node] = Artifacts(self.declared.tree, directory).write(
            json.dumps(manifest).encode(),
            media_type="application/json",
            schema_name="mainboard.run.v1",
        )
        return self.manifests[node]

    def writer(self, node: str) -> TrialReceipts:
        """One claim's append-only store for this run, opened on first use.

        The claim's `baselines/` digest is taken here, once, since every receipt of the claim is
        scored against the same directory.
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
    """One trial's evidence line: derived identity, host and source, plus what the lane measured.

    A lane settles with one of its workspace's declared words and its readings. One that settles
    nothing settles `failed` at teardown, leaving a row rather than a hole, and the trial fails.
    """

    def __init__(self, item: pytest.Item, session: Session) -> None:
        self.item = item
        self.session = session
        self.lane, self.key = lane_of(item)
        self.settled = ""
        self.gated = ""
        self.artifacts: dict[str, JsonValue] = {}

    def __getattr__(self, name: str) -> Callable[..., None]:
        """One declared word as a method, `trial.validated(...)`; an undeclared one refuses."""
        words = self.session.declared.words
        if name not in words:
            raise AttributeError(
                f"{name!r} is not a declared settle word; declared: {words.names}"
            )
        return partial(self.settle, name)

    def gate(self, registration: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Hand back the committed row this trial is scored against, digesting it on the way.

        The digest rides on the receipt as `gate_digest`, so a reader can tell a pre-registered
        gate from one edited into agreement afterwards.
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
        """Write this trial's fragment and tell the terminal which word to print for it.

        Tracked flags are read here, so the row names what was in force when it settled and a
        drifted value records this trial as the first suspect. Admissibility is per row because a
        verified tree can still collect a lane outside the captured source.
        """
        params = params_of(self.item, self.session.declared.universe.axes)
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
                "artifacts": dict(self.artifacts),
            }
        )

    def settle(self, word: str, reason: str = "", **measured: JsonValue) -> None:
        """Commit this trial under a declared word, with one line of reason and its readings."""
        self.record(word, reason=reason, measured=measured, outcome=Outcome.PASSED)
