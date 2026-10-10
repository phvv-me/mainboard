# The v26 per-call step: diagnosis of 2026-10-11 and the measured fixes (not applied)

Most of the v26 per-call slowdown on the GH200 (NFC 0.5-0.66 ms, card scaling's calls 1-6 1.33 ms
above call 0) is the instrument. Since the Polars removal, mainboard's trial `Log.table` opens a
fresh in-memory DuckDB database per row. A harness that writes a row, or runs the thread census,
between timed calls leaves the host cold for the next call. The proof is in the GH200 receipts.
Call 0, with no row before it, equals E-stages' undisturbed timing. The steady state equals
September's, and the unchanged Hugging Face NFC arm slowed too. The mechanism was measured on gold
and pedro-cvlab (0.1-1.5 ms) and is not yet measured on the GH200 (gh200_aftermath.py does that).
A smaller real engine cost is also present: a ledger built per call plus two syncing reads in
`RewriteStage.stage`, about 0.1-0.35 ms on Grace.

- mainboard-shared-row-database.diff: rows staged through one cached database (instrument).
- harness-nfc-rows-after-loop.diff, harness-card-rows-after-loop.diff: rows and census once per
  arm, after the timed loop. Both need registration amendments before any re-acquisition.
- engine-stage-overhead.diff: a lazy `Ledger.memory`, one ledger per stage, and an early return
  when nothing was rewritten. Recovers 0.10-0.17 ms per NFC cell on gold with byte-equal outputs.
  It is an engine change after the v26 freeze.
