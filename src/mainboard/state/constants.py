"""The constants of `state` other modules share."""

# The schema's own version, recorded in `schema_log` by `Lake.create` beside the DuckLake spec
# and by `Lake.evolve` when an older lake gains what this one added. 2 added `blobs`, 5 chunked
# them (`ordinal`) and added `evidence_log`, 6 added `checksums`, 7 bounded the row groups of
# blob files; 8 added artifact/operation-keyed reverse-engineering findings (`re_evidence`); 9
# added the cloud scoring columns of `costs` (`tier`, `datacenter`, `verdict`, `ready_s`,
# `probe_mbps`).
VERSION = 9

# The DuckDB extensions that reach or serve a lake over Quack.
QUACK = ("httpfs", "quack")

# How many times a commit that lost a race to another writer's snapshot is retried.
RETRIES = 100

# How many evidence files are read at once: opening a file is what costs on a scanned disk, and
# the opens overlap.
READERS = 16

# How many tree copies an eviction moves aside before it asks the lake about them together: each
# read of the lake scans every data file, so one read per file outlasts a tree of hundreds of
# thousands, and the files wait aside, off their names, for the read.
EVICTED_TOGETHER = 256

# The most object bytes one replicating or recalling read gathers: DuckDB holds a read's whole
# answer (half the center lake in one read peaked at 32 GB), and a window is one scan of `blobs`.
WINDOW_BYTES = 1 << 30
