"""The constants of `lab` other modules share."""

# The whole published contract between a trial and anything reading its output: one JSON line
# under this key, printed by whatever drives the trial (nothing here prints it). The key names
# the shape, not the harness, so any framework printing the same line reads the same, and a
# reader (a proof-bookkeeping tool turning a claim's run into evidence naming its trial and
# gates) parses it instead of importing this package.
RECEIPT = "trial_receipt"
