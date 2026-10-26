"""The constants of `schema` other modules share."""

# What a workspace holds as data rather than source, in the sync filter language: never sealed or
# archived as a local trial's source, and pinned through `needs` or `resources` by what reads it.
# The same places `[git] never-commit` keeps out of commits, so a source package named `datasets`
# or `evidence` stays source.
DATA = ("/datasets/", "/research/*/datasets/", "/references/", "**/experiments/**/evidence/")
