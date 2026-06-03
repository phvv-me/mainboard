# Entorno

`Machine.environment`, también incluido en la instantánea, informa quién está ejecutando y qué planificador de trabajos ofrece el host, para que una herramienta pueda enrutar el trabajo sin volver a detectarlo.

```python
from mainboard import Machine

env = Machine().environment
print(env.user, env.scheduler)
```

| campo | tipo | significado |
|---|---|---|
| `user` | `str` | nombre de inicio de sesión del usuario actual |
| `group` | `str` | nombre del grupo primario |
| `groups` | `tuple[str, ...]` | cada grupo al que pertenece el usuario |
| `scheduler` | `Scheduler` | planificador de trabajos en PATH: `slurm`, `pbs`, `pueue` o `none` |

Los planificadores de clúster tienen prioridad sobre pueue cuando hay más de uno en PATH. Cada campo se sondea de forma defensiva: un host donde el usuario, grupo o planificador no se pueden resolver produce valores vacíos en lugar de un error.
