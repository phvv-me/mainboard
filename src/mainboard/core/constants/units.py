"""Units of measure and the second and byte budgets every layer shares, below all of them."""

NANOSECONDS_PER_SECOND = 1_000_000_000

MILLISECONDS_PER_SECOND = 1000

NANOSECONDS_PER_MILLISECOND = 1e6

BYTES_PER_GB = 1e9

SECONDS_PER_HOUR = 3600

SECONDS_PER_DAY = 86400

BYTES_PER_KIBIBYTE = 1024

BYTES_PER_GIBIBYTE = 2**30

# The free bytes every large local writer leaves on its disk (`Disk.require`): the center reached
# 0 bytes about eight times in three weeks, each time from another writer, and every write then
# failed at once, the half-written file included.
DISK_FLOOR = 30 * BYTES_PER_GIBIBYTE

# How long, in seconds, one tool may take to answer a one-line question before the answer is
# unknown. A manager whose daemon is unreachable keeps retrying for minutes (`snap` without snapd's
# socket), so a pass takes at most this long per tool, whatever the machine can reach.
ASK_SECONDS = 30

# Seconds a release takes to stop a rented meter, billed after the command ends. Every lease keeps
# this much clear of its budget (`Lease.priced`), and every score bills it (`Scorecard.score`).
RELEASE_SECONDS = 120

# The command seconds a job that states no walltime is ranked at: one useful hour.
DEFAULT_RUN_SECONDS = 3600.0

# The free bytes a job host keeps where its workspace lives: dispatch refuses a job below it, and
# `host list --facts` warns. No job declares how much it writes, so the bar is one flat figure.
JOB_HOST_DISK = 20 * 10**9
