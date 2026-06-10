# Profiling

`mainboard.profiling` times your code and shows where time and memory go, the same API
on CUDA, macOS, and AMD. Annotation is near-free when no profiler is attached, and the
deep per-kernel tier is off by default.

![Profiler capabilities](assets/capabilities.svg)

## How it works

You run a **`Profiler`**, and it produces a **`Profile`**, an immutable result. Everything
you do with a measurement is a verb on that `Profile`.

1. **Annotate** regions with `region()` / `@profile`, or auto-annotate a whole package.
2. While it runs, a background sampler records memory/power/utilization. With
   `trace=True`, CUPTI captures every GPU kernel (asynchronously, one device sync at the
   end, minimal impact).
3. **`p.result()`** returns the `Profile`: `show()`, `bottlenecks()`, `diff()`,
   `perfetto()`, `save()` / `load()`.

## Examples

Each [example](https://github.com/phvv-me/mainboard/tree/main/examples) is a few lines:

| File | Shows |
|---|---|
| `regions.py` | time named regions, print a table |
| `decorator.py` | `@profile` every call of a function |
| `diff.py` | compare two runs (what got faster) |
| `deep_trace.py` | CUDA per-kernel trace + Perfetto export |

## Annotate

```python
from mainboard.profiling import Profiler, region, profile

with Profiler() as p:
    with region("load"):
        ...
    p.show()
```

`@profile` wraps a function so each call is a region. To profile code you don't want to
edit, auto-annotate by package (`Profiler(auto=["mypkg"])`) or from the shell with no
code at all:

```sh
mainboard profile your.module --auto your.package
```

## Read the result

```python
prof = p.result()
print(prof)             # the profile prints itself (rich table in a rich console)
prof.bottlenecks(5)     # the slowest regions, aggregated by name
prof.diff(baseline).show()   # per-region deltas: green where faster, red where slower
```

Save a run and diff against it later to track an optimization:

```python
prof.save("before.mbprof")
# ... optimize ...
after.diff(Profile.load("before.mbprof")).show()
```

## One-call bottleneck report

When you just want the verdict on a single hot function, `profile` runs it under the
profiler and hands back a structured `ProfileReport`: the dominant kernel, a
memory-versus-compute `bound`, achieved versus peak bandwidth, and the per-kernel
breakdown. It is exported straight from the top level.

```python
from mainboard import profile

report = profile(lambda: model(batch), iters=50, warmup=5, sync=torch.cuda.synchronize)
print(report)                 # the report prints itself as a rich table
report.bound                  # Bound.MEMORY or Bound.COMPUTE
report.dominant_kernel        # the kernel eating the most time
report.kernels                # the full per-kernel KernelStat breakdown
```

Pass a zero-arg callable (bind args with a lambda or `functools.partial`). `sync` is a
device barrier run after each pass, so async GPU work is captured rather than just the
launch. The report degrades per device: it asks only for the activity kinds the GPU
supports, and any dropped kind lands in `report.unavailable` instead of raising.

To profile against a clean window rather than someone else's load, gate on GPU
contention first:

```python
from mainboard import gpu_busy, wait_for_idle

if not gpu_busy():            # is another job using GPU 0 right now?
    report = profile(work)

wait_for_idle(timeout=60)     # block until the GPU is free, True if it became idle
```

`gpu_busy` reads live NVML utilization and memory (busy means compute over 10 percent or
memory over 90 percent of capacity) and reads `False` on a CPU-only host. `wait_for_idle`
polls it until the device is free or the timeout elapses.

## Deep trace and Perfetto

`trace` is one knob: `False` (off), `True` (kernels + memcpy), or an `Activity` flag for
exactly the CUPTI kinds you want:

```python
from mainboard.profiling import Activity

with Profiler(trace=Activity.ALL) as p:   # or trace=True for just kernels + memcpy
    with region("matmul"):
        ...
prof = p.result()
prof.trace_report()           # compute-vs-copy split, hot regions and kernels
prof.perfetto("trace.json")   # open at https://ui.perfetto.dev
```

Regions and kernels share the GPU clock, so they line up on one timeline. The same
export works wherever the deep tier is available (CUDA today, see the roadmap above).

### What this device supports

CUPTI activity kinds vary by GPU and driver, so mainboard probes the device and adapts:

```python
Profiler().supported()        # e.g. Activity.KERNEL|MEMCPY|MEMSET|...|MEMORY_POOL
```

`trace=Activity.ALL` means "everything this device offers", so it collects the supported
subset and logs anything dropped. Asking for a specific kind the device can't collect
*fails fast* instead of silently returning nothing:

```python
Profiler(trace=Activity.MEMORY)   # ValueError if this GPU has no MEMORY activity kind
```
