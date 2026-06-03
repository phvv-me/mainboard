# Environment

`Machine.environment` は、スナップショットにも含まれており、誰が実行していて、ホストがどのジョブスケジューラを提供しているかを報告します。これにより、ツールは再検出することなく作業をルーティングできます。

```python
from mainboard import Machine

env = Machine().environment
print(env.user, env.scheduler)
```

| フィールド | 型 | 意味 |
|---|---|---|
| `user` | `str` | 現在のユーザーのログイン名 |
| `group` | `str` | プライマリグループ名 |
| `groups` | `tuple[str, ...]` | ユーザーが所属するすべてのグループ |
| `scheduler` | `Scheduler` | PATH 上のジョブスケジューラ: `slurm`、`pbs`、`pueue`、`none` |

複数が PATH 上にある場合、クラスタスケジューラが pueue より優先されます。すべてのフィールドは防御的に探査されます。ユーザー、グループ、スケジューラを解決できないホストでは、エラーではなく空の値が返されます。
