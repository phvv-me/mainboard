# Environment

`Machine.environment`（也包含在快照中）报告谁在运行以及主机提供哪种作业调度器，因此工具无需重新检测即可路由工作。

```python
from mainboard import Machine

env = Machine().environment
print(env.user, env.scheduler)
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `user` | `str` | 当前用户的登录名 |
| `group` | `str` | 主组名称 |
| `groups` | `tuple[str, ...]` | 用户所属的每一个组 |
| `scheduler` | `Scheduler` | PATH 上的作业调度器：`slurm`、`pbs`、`pueue` 或 `none` |

当 PATH 上同时存在多个调度器时，集群调度器优先于 pueue。每个字段都经过防御式探测：在无法解析用户、组或调度器的主机上，会产生空值而非错误。
