import threading
import time


class Ledger:
    """The work `in_waves` runs over a graph: what started, and what had finished by then.

    failing: the item whose work raises `RuntimeError` instead of finishing.
    """

    def __call__(self, item: int) -> None:
        with self.lock:
            self.started.append(item)
            self.seen[item] = frozenset(self.finished)
        time.sleep(0.0005)
        if item == self.failing:
            raise RuntimeError(f"item {item} failed")
        with self.lock:
            self.finished.add(item)

    def __init__(self, *, failing: int | None = None) -> None:
        self.failing = failing
        self.lock = threading.Lock()
        self.started: list[int] = []
        self.finished: set[int] = set()
        self.seen: dict[int, frozenset[int]] = {}
