import sys
from pathlib import Path

from .profiler import Profiler
from .spans import span
from .target import Target
from .trace import Activity


def main() -> None:
    """Run one target inside local collectors for an optional Tachyon parent."""
    mode, features, activities, output, target, *args = sys.argv[1:]
    profiler = Profiler(
        features=Profiler.Feature(int(features)),
        activities=Activity(int(activities)),
    )
    try:
        with profiler, span("program"):
            Target(name=target, module=mode == "module", args=tuple(args)).run()
    finally:
        profiler.result().save(Path(output))


if __name__ == "__main__":
    main()
