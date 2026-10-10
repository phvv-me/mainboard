# Bugs found after the design (2026-10-10)

B56. Archive members are unreadable once eviction removes their archive parts (ISSUES 13o).
`ParquetArtifacts.read` opens `artifacts.parquet/<digest[:2]>/` only on disk and
`Artifact.read`'s fallback recalls only standalone blobs, so the 10-09 eviction made 53,444
September artifacts unreadable though the lake held them. Owners: M01 (eviction refuses an archive
container whose members readers address by path) and M27 (verified object access reaches archive
members through the lake). Acceptance: the lake holds only the parts, the tree is empty, and
`Artifact.read` returns every member sha256-verified.
