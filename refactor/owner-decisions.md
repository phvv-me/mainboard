# Owner decisions on the mainboard refactor (2026-10-10, answering design-v2)

1. Cutover: **now**. Do not wait for the MLSys deadline. Each wave reaches the live tool as soon as
   its gates pass, by drain-and-switch (below), while v26 GH200 jobs stay queued. Jobs already
   dispatched keep the code they shipped, and the new settler must settle them.
2. Plan size: **design-v2, with drain-and-switch instead of the service-principal fence (Q13)**. One
   user owns the center and every host, so the old controllers are fenced by draining them: stop
   the launchd settler, wait for or stop running `mb` processes, switch the center's editable
   checkout (and later the host installs) in one step, then start the new settler. No separate
   principal, no credential rotation. A wave that cannot drain safely waits.
3. Job state (Q01, Q02): **DuckLake with a PostgreSQL catalog**. The owner proposed it, and
   DuckLake's documentation recommends PostgreSQL for multi-client lakes. One PostgreSQL server
   runs on the center, its data directory on the internal disk, managed by mainboard as a service
   with its binaries from the workspace environment (conda-forge postgresql), never a system
   install. It holds the DuckLake catalog (replacing lake.sqlite) and, in a separate schema, the
   job and control state through SQLAlchemy Core. Locks are PostgreSQL advisory locks: released
   when their session dies, holders visible in pg_locks and pg_stat_activity. Parquet data files
   and blobs stay under the lake's DATA_PATH on the external volume. list and show are passive
   readers that report freshness, and the durable settler (or an explicit refresh or wait) acts.
4. GPU sharing (Q05): **one host-wide card lease that everyone takes**, meaning mainboard jobs,
   refactor gates and agent runs. Registered exclusive runs wait for the lease and refuse an
   unmanaged card.
5. Every other question (Q04, Q06 to Q12, Q14, Q15): the design's recommendation stands.

## Owner decisions of 2026-10-11

6. The center's data goes on /Volumes/PORTABLE, with no watcher. The PostgreSQL data directory of
   M36 and the center's large state move off the 96%-full internal disk onto PORTABLE. Services
   staying down after a reboot until a console login is accepted, and no watcher is added. launchd
   services reach PORTABLE through the ssh-localhost route that the settler already uses, or through
   a Full Disk Access grant.
7. Logging is zero-cost. The trial harness defect of 2026-10-11 (a DuckDB database opened per
   table row, rows and censuses between timed calls) is fixed so that logging adds nothing inside
   a timed region. Rows and censuses are buffered in memory and written after the timed loop, and
   the harnesses keep outputs to verify after the loop. The affected v26 GH200 cells are acquired
   again (E-grid, card scaling, NFC, cooperative offline). The engine stays frozen at v26, and its
   per-call ledger cost (0.1-0.35 ms) is fixed after the paper.
