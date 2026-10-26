"""The constants of `trials` other modules share."""

# What a read of table artifacts answers when none holds the schema: no rows, and the `_trial`
# provenance column every such read adds beside the table's own.
NO_TABLE = "SELECT NULL::VARCHAR AS _trial WHERE false"

# The suffix naming an axis's probe outcome column, so a store never has to be told which columns
# are outcomes and which are values.
PROBED = "_probed"

# The default marker table. `gpu`, `slow` and `paid` decide whether a trial runs; only `paid` is
# wired to an option, since its cost is money. `adversarial` and `search` name an adaptive lane
# kind, whose result is a candidate rather than coverage: `-m "not adversarial and not search"`
# is a run that claims only what a declared grid measured.
MARKERS = {
    "gpu": "needs a real card, skipped where there is none",
    "slow": "runs for minutes rather than seconds",
    "paid": "could bill money, skipped unless --paid is passed",
    "adversarial": "hunts a counterexample by shrinking, so what it finds is a candidate",
    "search": "proposes its own points adaptively, so what it finds is a candidate",
    "phase": "the registered phase its receipts belong to, a coverage axis when one is declared",
}

# The fields holding a whole object, stored as JSON text so one lane's measurement shape is not
# forced into every other lane's fragment. Writer and reader both read this one tuple.
NESTED = ("params", "measured", "versions", "gates", "artifacts")

# Which package drives each lane kind; the key is also the marker and the `mainboard[...]` extra.
DRIVERS = {"adversarial": "hypothesis", "search": "optuna"}

# The report property a settled word rides to the terminal on, through pytest's own typed channel
# from an item to its report.
WORD = "mainboard_trials_word"

# The creation coordinate every recency question reads, and the field deciding evidence or scratch.
OPENED = "opened_at_ns"

ADMISSIBILITY = "admissibility"

# What a claim's registered rows live under, one directory per node, and what counts as a source
# file when the universe root is digested.
BASELINES = "baselines"
