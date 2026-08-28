# PROCESS-GLOBAL STATE, AND THE ONLY SHAPE ALLOWED TO MOVE IT.
#
# A measurement runs under whatever the process happens to be configured with, so a knob one lane
# moves and does not move back is measured by every lane collected after it. That is not a
# hypothetical: a suite whose lanes each set a numeric backend policy and left it wherever their
# last lane put it published a universe's readings against the wrong machine, twice, and both
# readings were internally consistent and unfalsifiable from the numbers alone.
#
# So a tracked knob is a `Flag`, a name with the two halves that read and write it, and the only
# way to move one is `held`, which records the value on the way in and writes it back on the way
# out. ENTER IS UNREPRESENTABLE WITHOUT EXIT: there is no `pin()` to call and forget, because the
# whole defect class is a call that was made and never paired.
#
# AND THE SESSION REFUSES RATHER THAN WARNS. A flag left off its recorded baseline at the end of a
# run means some trial measured a machine nobody can identify, so `moved` is what the harness asks
# at teardown and a non-empty answer fails the session naming the flag, the two values and the
# first trial that settled under the wrong one. A warning in a scroll-back is not a gate.

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from pydantic import JsonValue


@dataclass(frozen=True, slots=True)
class Flag:
    """One process-global value a trial's readings depend on, and how it is read and moved.

    A knob honored only at process start has no honest writable form: setting a workspace or an
    allocator policy after the library has read it changes the receipt and not the machine, which
    is a lie in a column. Such a knob is declared with no `write` at all, which makes it ASSERTED
    rather than held: it is recorded on every receipt and audited at the end of the run, and no
    code path can pretend to have moved it.

    name: the column a receipt records this value under, and the name a refusal prints.
    read: reads the live value, wherever it lives.
    write: sets it, taking a value `read` returned earlier, absent for an asserted knob.
    """

    name: str
    read: Callable[[], JsonValue]
    write: Callable[[JsonValue], None] | None = None


def reading(flags: Sequence[Flag]) -> dict[str, JsonValue]:
    """What every tracked flag reads right now, which is one row's worth of machine state."""
    return {flag.name: flag.read() for flag in flags}


def _restore(flags: Sequence[Flag], baseline: Mapping[str, JsonValue]) -> None:
    """Write every writable flag back to `baseline`, leaving an asserted one exactly as it is.

    flags: the tracked knobs. baseline: what each read before the block that may have moved them.
    """
    for flag in flags:
        if flag.write is not None:
            flag.write(baseline[flag.name])


@contextmanager
def held(*flags: Flag) -> Iterator[dict[str, JsonValue]]:
    """Hold every flag's current value for the block, writing each one back on the way out.

    Yields the recorded baseline, so a lane that wants to state what it ran under reads it from
    the same object that guarantees it will be restored. An asserted knob is recorded and never
    written, so it rides in the baseline and is simply skipped on the way out.

    flags: the tracked knobs this block may move.
    """
    baseline = reading(flags)
    try:
        yield baseline
    finally:
        _restore(flags, baseline)


def moved(flags: Sequence[Flag], baseline: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Every tracked flag whose live value has drifted off `baseline`, and what it reads now.

    Asserted knobs are audited exactly like held ones. A knob nothing was allowed to write that
    has moved anyway is the more alarming of the two findings, not the more forgivable.

    flags: the tracked knobs. baseline: what each read when the run opened.
    """
    return {name: value for name, value in reading(flags).items() if value != baseline[name]}
