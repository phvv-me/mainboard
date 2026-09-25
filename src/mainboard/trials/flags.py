# Process-global state, and the only shape allowed to move it.
#
# A knob one lane moves and does not move back is measured by every lane after it: a suite whose
# lanes each set a numeric backend policy published a universe's readings against the wrong
# machine, twice, and both were internally consistent. So a tracked knob is a `Flag` and the only
# way to move one is `held`, which writes it back on the way out; there is no `pin()` to call and
# forget. The session refuses, rather than warns, when a flag ends off its baseline, naming the
# flag, both values and the first trial that settled under the wrong one.

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from pydantic import JsonValue


@dataclass(frozen=True, slots=True)
class Flag:
    """One process-global value a trial's readings depend on, and how it is read and moved.

    A knob honored only at process start is declared with no `write`, which makes it ASSERTED:
    recorded on every receipt and audited at the end of the run, since setting it after the
    library read it would change the receipt and not the machine.

    name: the receipt column and the name a refusal prints.
    write: sets a value `read` returned earlier; absent for an asserted knob.
    """

    name: str
    read: Callable[[], JsonValue]
    write: Callable[[JsonValue], None] | None = None


def reading(flags: Sequence[Flag]) -> dict[str, JsonValue]:
    """What every tracked flag reads right now."""
    return {flag.name: flag.read() for flag in flags}


@contextmanager
def held(*flags: Flag) -> Iterator[dict[str, JsonValue]]:
    """Hold every flag's current value for the block, writing each writable one back on exit.

    Yields the recorded baseline, asserted knobs included.
    """
    baseline = reading(flags)
    try:
        yield baseline
    finally:
        for flag in flags:
            if flag.write is not None:
                flag.write(baseline[flag.name])


def moved(flags: Sequence[Flag], baseline: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Every tracked flag, asserted ones included, whose live value drifted off `baseline`."""
    return {name: value for name, value in reading(flags).items() if value != baseline[name]}
