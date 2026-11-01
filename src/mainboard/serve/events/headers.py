"""Claude Code's `otelHeadersHelper`: the sink's token as an Authorization header, from its file.

Claude runs it by path with the tool's own interpreter, `python -I headers.py TOKEN_FILE`, when a
session starts and every 29 minutes after, so the token lives in its 0600 file alone and never in
a settings file. Without a readable token it prints no header: the sink then refuses the
session's records, which telemetry may lose by rule, and a `claude --print` keeps a clean
standard error instead of warning that its helper failed.

It imports the standard library alone, as the keeper does, so a session starts without importing
the tool; that is also why it writes its one JSON object to standard output itself, the helper's
protocol, rather than through the project logger.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _main(arguments: Sequence[str]) -> None:
    try:
        token = Path(arguments[0]).read_text(encoding="utf-8").strip()
    except IndexError, OSError:
        token = ""
    sys.stdout.write(json.dumps({"Authorization": f"Bearer {token}"} if token else {}))


if __name__ == "__main__":
    _main(sys.argv[1:])
