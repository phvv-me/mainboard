# Examples

Tiny, runnable profiler examples. Each is a few lines.

```sh
python examples/regions.py     # time named regions, print a table
python examples/decorator.py   # @span every call of a function
python examples/diff.py        # compare two runs (what got faster)
python examples/deep_trace.py  # CUDA: per-kernel trace + Perfetto export
```

No code at all — profile any module or script from the shell:

```sh
mainboard profile run your.module
```

See the [profiling tutorial](https://phvv.me/mainboard/profiling/) for how it works.
