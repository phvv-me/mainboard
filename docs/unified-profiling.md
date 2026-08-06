# One config for the whole profiling surface

Profiling a system means choosing three independent things, and mainboard currently spreads them
across thirty seven parameter slots on four entry points with no object that holds a choice.

    __init__          6  features, activities, device_index, sample_interval_ms, max_spans, auto
    run              15  target, module, args, features, activities, mode, format, output,
                         duration, sampling_rate, all_threads, blocking, executable, timeout,
                         strict
    run_instrumented  6  target, tachyon, executable, features, activities, timeout
    attach           10  pid, mode, format, output, duration, sampling_rate, all_threads,
                         blocking, executable, timeout

`features` and `activities` appear three times, `executable` and `timeout` three times, and the
eight Tachyon knobs are duplicated wholesale between `run` and `attach`. A caller who wants the
same collection policy in two places passes it twice and nothing checks that they agree.

## The three concerns, which are genuinely orthogonal

**Collection.** What evidence to gather and at what cost. `features`, `activities`,
`device_index`, `sample_interval_ms`, `max_spans`, `auto`. This is the one that already has a
type, `Profiler.Feature`, and the Flag is honest about its own contract: independent costs
combinable with `|`.

**Reach.** How to get at the thing being measured. In-process, launch a target, or attach to a
pid. `target`, `module`, `args`, `executable`, `timeout`, `pid`, `blocking`. Today these decide
which of three entry points you call, which is why the knobs repeat.

**Sampler policy.** How the external Python profiler is driven. `mode`, `format`, `output`,
`duration`, `sampling_rate`, `all_threads`, `strict`. `Tachyon` is already a model holding most
of these, and then `run` and `attach` take them again as loose arguments and rebuild it.

Three models, one per concern, and the entry points take those instead. `Tachyon` shows the
shape already works; the other two have no equivalent.

## The fourth concern, which has no home at all

What varies about the input. Nothing in mainboard represents it, so every study that needs it
builds its own loop, and a number gets recorded without the conditions that produced it.

`cunicode.workload.Workload` is one instance of this and nothing about it is tokenizer-specific.
It states a byte budget, a mean document length, a length spread, a mean single-script run length
and a character-width mix, and it can synthesise input to that or select the matching slice of a
real corpus. What makes it work is that a specification separates axes a corpus ties together, so
a measurement that moves can say which axis moved it.

On its first run that model corrected a diagnosis that had been written down twice. A document
taking sixteen seconds had been attributed to Chinese script; sweeping run length showed ASCII and
CJK tracking within noise at every length, so the variable was pretoken length and script never
entered. It also showed the effect was not a threshold but a curve, throughput falling as roughly
one over run length across the whole range, which means every corpus sits somewhere on it rather
than one document being pathological.

That is the argument for putting this in mainboard rather than leaving it per project. A study
that varies nothing cannot distinguish a property of the system from a property of the input it
happened to use, and that failure produced three wrong conclusions in one session.

## What the central config is

A `Study` holding one `Collection`, one `Reach`, one `Sampler` and one grid of input points, and
returning one row per point. The grid is the cartesian product of whatever axes the caller
names, so a study is declared rather than looped.

Two properties matter more than the shape.

Rows are the artifact, not a chart. The same run has to be re-readable as new questions arrive,
which means every point keeps its distributions rather than a mean. The document that cost
sixteen seconds is invisible in any average and obvious in a spread, and the same is true of the
warp imbalance that `cunicode bench skew` reports, which is a ratio between what a schedule
serves and what the work needs and cannot be recovered from a total.

And a point's conditions travel with its measurement. A throughput number whose input
specification is not attached is not evidence, because the axis that explains it may not be the
one the caller thought they were varying.

## Sequencing

The correlation id landing on `KernelTrace` and `MemcpyTrace` is done and is the prerequisite for
joining a host-side sample to a device kernel, since it is CUPTI's own link from a runtime launch
to the kernel it produced.

Next is the three collapse models, which is a refactor with no behaviour change and can be
verified by the existing suite. Then `Study` and the grid. Then the view, which is a small
multiple over the axes rather than a single plot, because the surface has four dimensions and no
single plot shows one.

Metrics stays out. The installed CUPTI binding has no Profiling API, and counter collection needs
kernel replay so it cannot share a pass with Activity, which is exactly the independence the
`Feature` flag promises. If it ever arrives it is its own pass behind its own entry point.

## Progress

Two of the three collapse models are in.

`Tachyon` is now what `run` and `attach` take, instead of the eight loose fields they used to
take and rebuild it from. `run` went from fifteen parameters to eight and `attach` from ten to
three, and the duplication between them is gone. The flag-to-model conversion moved to
`sampler_from_flags` in the CLI, which is where it belongs: a command line wants flat flags, a
library wants one value, and the boundary is the one place that should know both. One
consequence worth having is that `run` now takes the interpreter the target runs under from
`sampler.executable`, so the sampler and the target can no longer disagree about which Python
they mean, which two separate arguments allowed.

`Collection` holds the six collection choices and the constructor is a flat façade over it, so
no call site changed. It is reusable, comparable by value and serialisable, which is what makes
it storable beside the measurement it produced. Until now a collection policy existed only as
arguments that had already been consumed by the time there was a number to record.

`Feature` moved to module scope to let `Collection` name it, with `Profiler.Feature` kept as an
alias exactly as `Profiler.Activity` already was. Its docstring now says why independence is a
contract rather than a convenience, since that is the property a sixth member would break.

`Reach` is in too, and it is the one that changes the shape rather than tidying it. Its three
constructors name the three ways of getting at a target, `here`, `launch` and `attaching`, and
`measure` dispatches on the model instead of the caller dispatching by choosing a function. So
which way a measurement was taken is now data that serialises alongside it, which is what a study
needs and what three separate methods could never provide. `run` and `attach` stay as the two
shorthands a caller writes by hand, façades over `measure` the way the constructor is a façade
over `Collection` and the CLI is over `Tachyon`.

One test earned its keep during this. Adding `from __future__ import annotations` to let `Reach`
name itself broke `test_profiler_signature_resolves_runtime_annotations`, which pins the
constructor's annotations as real type objects because the command line introspects them. The
future import turns every annotation in the module into a string, so only the three genuine
forward references are quoted and the rest stay resolvable.

What is left is `Study`, holding one `Collection`, one `Reach`, one `Sampler` and a grid of input
points. Every piece it composes now exists.
