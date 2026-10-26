from typing import Final

NANOSECONDS_PER_SECOND = 1_000_000_000
MILLISECONDS_PER_SECOND = 1000
BYTES_PER_GB = 1e9
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
BYTES_PER_KIBIBYTE = 1024
# How long, in seconds, one tool may take to answer a one-line question before the answer is
# unknown. A manager whose daemon is unreachable keeps retrying for minutes (`snap` without snapd's
# socket), so a pass takes at most this long per tool, whatever the machine can reach.
ASK_SECONDS = 30

SSH_FAMILY: Final = "ssh-family"
