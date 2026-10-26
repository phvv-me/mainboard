# EXPERIMENT EVIDENCE KEPT IN THE LAKE, so its files can leave version control and every reader
# still finds their bytes.
#
# An evidence file's bytes are one content-addressed object in `blobs` (chunked, see
# `state.blobs`), and `evidence_log` records which workspace-relative path held which object; the
# `evidence` view is each path's latest record. Bytes are kept exactly as written, never decoded
# or re-encoded, so every SHA-256 a `receipts.json` or an artifact reference pins keeps verifying
# whether the bytes come from disk or from the lake. An ingest commits in groups of whole files,
# each group one insert, so small files share Parquet data files instead of one each.
#
# INTEGRITY. `verify` puts the object of every indexed path through the one blob verifier
# (`state.blobs`), a window of objects at a time, and reports each path whose object is missing,
# short or damaged. It never writes. A chunk with no checksum gets none recorded, and `ingest`
# repairs.
#
# CUSTODY. A tree copy leaves only on proof. `inventory` reads where every indexed file stands
# against the lake and changes nothing, and `evict` consumes that same inventory for the files
# their producer has `finalize`d: each is moved by name into a private folder beside it, hashed
# and checked against the lake once more, and only then removed, so a file replaced or written
# meanwhile is put back and never unlinked. Without that handoff nothing is evicted, since
# renaming fences no writer that still has the file open. The parts of an artifact archive are
# never evicted: readers open its members by path. `materialize` goes the other way as carefully,
# through `OwnedFile`, and refuses a changed file already there, naming both digests.
#
# READERS. What used to glob a directory asks an `EvidenceTree` instead: the files on disk, and
# every file the lakes of the workspaces holding that directory index under it, materialized on
# first use into the lake's read-through cache (`<state>/evidence/<path>`), outside the tree and
# outside version control. A read pinned by digest needs no path at all: the object is found by
# its SHA-256 in any of those lakes. A nested workspace (a research repository with its own
# manifest inside the monorepo) is served by its own lake first and the enclosing one after.
#
# REPLICATION. Once evidence leaves version control the lake is its only copy, so a second one
# on another disk is what makes that safe. `Evidence.replicate` fills a `DirectoryReplica` on a
# mounted disk (the center's external drive): every object it lacks, then the whole path index, so
# both the bytes and which path held them survive the lake's disk.

from .core import Evidence
from .keeping import within
from .models import Finalized, Inventory, Kept, Located
from .replica import DirectoryReplica
from .tree import EvidenceTree

__all__ = [
    "DirectoryReplica",
    "Evidence",
    "EvidenceTree",
    "Finalized",
    "Inventory",
    "Kept",
    "Located",
    "within",
]
