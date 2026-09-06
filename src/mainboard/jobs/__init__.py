# A job is a file and a name inside it, dispatched with exactly the code it imports.
#
# Only the declaration is exported here, because this is the module a job file imports and a job
# file is imported on the node with the environment's GPU libraries around it. The walk, the
# spelling and the runner live in their own modules and are reached by name.

from .declare import Declaration, job

__all__ = ["Declaration", "job"]
