from datetime import UTC, datetime

# The instant forged frames claim, the ssh host the runs are on, and the scheduler's handle of the
# run under test, which the runner on the host could not know.
AT = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
HOST = "gold"
HANDLE = "42"
