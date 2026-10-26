# Tests

Every test of mainboard lives here. A stranger should find the tests of a piece of source by its
path, read any test module in the same order, and add a test without choosing a style. The rules
below are enforced by `mb refactor` (the id after each rule) and fixed where a fix is mechanical.

The convention is not Python's alone. Where it is about layout and naming (rules 1 to 3 and 5) it
is enforced by language-independent `ALL-` rules, with one layout adapter per ecosystem (a
mirrored folder tree for Python, `Tests/<Target>Tests` beside `Sources/<Target>` for Swift, `tests/`
beside the crate for Rust, a colocated `*.test.ts` for TypeScript). Where it is about the test
framework (rules 6 to 9: pytest, Hypothesis, `monkeypatch`) the rule is a `PY-` one, and the
other languages get their analogue (`#[should_panic]` and proptest, Swift Testing's
`#expect(throws:)` and `@Test(arguments:)`, Vitest's `toThrow` and `vi.spyOn`) as their test
facts allow.

## Where files live

1. **The first folder is the source package under test.** `integration/<package>/...` holds the
   tests of `src/mainboard/<package>/` (a top-level module such as `cli.py` counts as one).
   Below it, folders name the subject (`state/lake/`, `state/evidence/`), mirroring the
   subpackage where there is one. Two folders are not packages: `scenarios/` (several packages
   through the real `mb`, end to end) and `gates/` (the harness itself). No test module sits
   directly in `integration/`. `ALL-TEST0004`
2. **One module per subject, `test_<subject>.py`.** A subject is a public class, verb or
   behavior family. A module holds at most 20 members, a folder at most 5 modules; a subject
   past that becomes a folder named for it. `ALL-TEST0003`, `ALL-FILE0003`, `ALL-FILE0005`
3. **Non-test modules come from a closed vocabulary, one of each per folder:** `conftest.py`
   (fixtures), `constants.py` (named literals the folder's tests share), `fakes.py` (test doubles),
   `builders.py` (factories of inputs), `readers.py` (parsers and parsed-result models for what
   the code printed or wrote). Never `tools.py`, `helpers.py`, `utils.py`. A fixture
   used by one module stays in it; used by two or more, it moves to the folder's `conftest.py`.
   `ALL-TEST0005`

## What a test looks like

4. **A test is arrange, one act, assert.** At most 10 statements, no conditionals, no loops over
   cases. `PY-TEST0011`, `PY-TEST0012`, `PY-TEST0013`
5. **The name states the behavior** in at least five words
   (`test_a_lake_verb_refuses_a_path_outside_the_workspace`). Parametrize ids are kebab-case
   descriptions of the situation (`a-symlink-leaving-the-bundle`). `ALL-TEST0006`, `PY-TEST0023`
6. **The input space picks the kind of test.** A pure function over a wide input space is a
   Hypothesis property with an `@example` for every regression literal it replaces. A finite list of
   named cases is a `pytest.param` table. A sequence of operations on state is a
   `RuleBasedStateMachine`. A CLI contract is one subprocess test per verb; Hypothesis never wraps
   a subprocess. `PY-TEST0003`, `PY-TEST0014`, `PY-TEST0017`
7. **Hypothesis settings come from one profile** registered in the root `conftest.py`
   (`database=None`, `deadline=None`); a test states only what it changes (`max_examples`).
   `PY-TEST0020`

## Doubles and hermeticity

8. **A test that expects a refusal also asserts nothing else happened:** the fake's record is
   empty, nothing was written. `PY-TEST0021`
9. **Doubles sit at the external seam only** (process, network, clock, files outside `tmp_path`).
   A test never patches a private name of mainboard: a rename then breaks a test that pinned no
   behavior. `PY-TEST0022`
10. **Real external tools (`xcodebuild`, `git`, `ssh`) run only in tests marked `native`.**
    Markers are a closed set declared in `pyproject.toml` (`native`, `served`, `slow`).

## Changing the suite

`integration/gates/baseline.json` lists the cases the gate must still collect. A merge, split or
move changes ids, so it ends with a rebaseline entry naming the old ids and the tests that took
their place, and `MB_GATE_COVERAGE=1 sh integration/gates/run.sh target integration/<area>`
shows no line or branch lost against the area's earlier run.
