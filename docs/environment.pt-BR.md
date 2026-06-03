# Environment

`Machine.environment`, também presente no snapshot, informa quem está executando e qual job scheduler o host oferece, para que uma ferramenta possa rotear o trabalho sem ter que detectá-lo de novo.

```python
from mainboard import Machine

env = Machine().environment
print(env.user, env.scheduler)
```

| campo | tipo | significado |
|---|---|---|
| `user` | `str` | nome de login do usuário atual |
| `group` | `str` | nome do grupo primário |
| `groups` | `tuple[str, ...]` | todos os grupos aos quais o usuário pertence |
| `scheduler` | `Scheduler` | job scheduler no PATH: `slurm`, `pbs`, `pueue` ou `none` |

Schedulers de cluster têm prioridade sobre o pueue quando mais de um está no PATH. Todo campo é sondado de forma defensiva: um host em que o user, o group ou o scheduler não podem ser resolvidos retorna valores vazios em vez de um erro.
