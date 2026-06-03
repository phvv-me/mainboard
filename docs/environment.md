# Environment

`Machine.environment`, also carried in the snapshot, reports who is running and what job scheduler the host offers, so a tool can route work without re-detecting it.

```python
from mainboard import Machine

env = Machine().environment
print(env.user, env.scheduler)
```

| field | type | meaning |
|---|---|---|
| `user` | `str` | login name of the current user |
| `group` | `str` | primary group name |
| `groups` | `tuple[str, ...]` | every group the user belongs to |
| `scheduler` | `Scheduler` | job scheduler on PATH: `slurm`, `pbs`, `pueue`, or `none` |

Cluster schedulers take priority over pueue when more than one is on PATH. Every field is probed defensively: a host where the user, group, or scheduler cannot be resolved yields empty values rather than an error.
