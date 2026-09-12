import os
from types import SimpleNamespace

from mainboard.probe.occupancy import Occupancy, holder, rows


class Unit:
    """A probed card stand-in with the sensors the occupancy probe reads."""

    def __init__(self, index: int, label: str, holders: tuple[int, ...], util: int) -> None:
        self.index = index
        self.label = label
        self.memory = SimpleNamespace(used_bytes=3 * 10**9, total_bytes=24 * 10**9)
        self._holders = holders
        self._util = util

    def snapshot(self):
        return SimpleNamespace(
            utilization=SimpleNamespace(gpu_pct=self._util, memory_pct=0),
            processes=[SimpleNamespace(pid=pid, used_bytes=10**9) for pid in self._holders],
        )


def test_occupancy_names_each_card_and_who_holds_it() -> None:
    mine = os.getpid()
    reading = Occupancy.collected(
        [Unit(0, "NVIDIA GB10", (mine,), 0), Unit(1, "NVIDIA GeForce RTX 3090", (), 2)]
    )
    assert [card.free for card in reading.cards] == [False, True]
    held = reading.cards[0].holders[0]
    assert held.pid == mine and held.user and held.command
    listed = rows("gold", reading)
    assert listed[0]["host"] == "gold" and listed[0]["card"] == "0: NVIDIA GB10"
    assert str(mine) in str(listed[0]["holders"])
    assert listed[1]["free"] is True and listed[1]["holders"] == ""
    assert Occupancy.model_validate_json(reading.model_dump_json()) == reading


def test_a_vanished_holder_keeps_its_pid_alone() -> None:
    gone = holder(2**22 - 1, 5)
    assert gone.pid == 2**22 - 1 and gone.user == "" and gone.used_bytes == 5
