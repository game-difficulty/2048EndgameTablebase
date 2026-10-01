# Isolated merge-goal search experiment

This directory is an opt-in CLI experiment. It is not linked into AIPlayer,
Python bindings, WASM, the cloud service, or the live Worker. Production search,
table routing, scoring and configuration remain untouched.

## Goal and model

- Explicit target tile, fixed throughout one search. The target must be absent
  initially. Default board `330043598671da10`, target 2048: existing 8192 stays
  on the board; all smaller tiles total 2060.
- Success means creating the target by a move. Search ends immediately, before
  the next spawn. This does NOT certify safe table handoff or later survival.
- Actual no-legal-move states are deaths, including at the depth boundary.
- A nonterminal depth boundary or budget cutoff is unresolved, not death.
- Uniform empty-cell spawn, 90% 2 / 10% 4 by default. All nonzero-probability
  outcomes are enumerated; no sampling, masking, evaluation pruning or tasking.
- Reuses the repository BoardMover directly. Its 4-bit encoding does not merge
  two 32768 tiles; 65536 goals and ambiguous 0xf representations are unsupported.

## Values and policy

Each player state returns:

- `success_lower`: probability of hitting the goal in the explored tree under
  the selected contingent policy. With `complete=true`, this is the optimal
  success probability within the requested move horizon.
- `policy_death`: death probability seen under that SAME selected policy.
- `policy_unresolved`: 1 minus the above two probabilities. It is not a loss.
- `success_upper`: optimistic upper bound on eventual goal reachability, using
  1 for unresolved leaves. It can use a different policy from success_lower.
- `success_steps`: conditional mean steps on the already successful paths,
  NOT expected eventual completion time across unresolved paths.

At player nodes, maximize horizon success; break exact ties by lower observed
death, then lower success-weighted steps. Chance nodes probability-average.
Upper bounds are separately maximized over legal actions. Terminal success has
bounds [1,1], death [0,0], and unresolved [0,1]. Floating-point arithmetic is used;
these are numerical bounds, not interval arithmetic with directed rounding.

`best_observed` ranks these partial results. It is not necessarily the safest
eventual move. `dominance_certified` is true only if its lower bound exceeds all
other legal root upper bounds by more than 1e-12 (vacuously true for one move).
Overlapping intervals must not be treated as equal proven success rates.

## Cache and budgets

Single thread; each root direction has its own fresh cache, wall-time limit and
node limit. Full 64-bit board plus EXACT remaining depth is checked on every hit.
Goal and spawn parameters are immutable per search instance. No cross-depth or
cross-root reuse. Cache collisions only replace entries, never identify boards.
Only fully processed horizon results are cached; budget-cut results are not.

`complete=true` means the horizon was fully evaluated, NOT that eventual success
or failure was determined. Budget cutoffs preserve unexplored probability as
unknown. A node count includes calls/cache hits, not unique boards.
Time is checked every 1024 player calls and is a soft per-direction limit.
Reported seconds exclude root cache allocation; OS/process overhead is additional.

This first baseline is depth-first finite-horizon enumeration, not yet selective
bound-guided deepening/BRTDP. It deliberately exposes horizon/budget limitations
before implementing an adaptive allocator. No heuristic value claims probability.
Different budgets may explore different partial policies; incomplete-run lower
bounds need not improve just because a larger requested depth is supplied.

## Run

Requires Python 3 and a C++17 compiler. Set CXX if g++ is not on PATH.
The build script also detects C:/Apps/mingw64/bin/g++.exe.

```powershell
python experiments/merge_search/test_probe.py -v
python experiments/merge_search/run_experiment.py --depths 3 4 5 6 7 8 9 10
python experiments/merge_search/run_experiment.py --depths 7 8 9 10 --seconds 10 --nodes 1000000000 --cache-mib 128 --label larger_budget
```

The runner adds the compiler's runtime directory to child PATH. To invoke the
executable directly on Windows, add that directory to PATH as well:

```powershell
$env:PATH = 'C:/Apps/mingw64/bin;' + $env:PATH
experiments/merge_search/merge_probe.exe --board 330043598671da10 --target 2048 --depth 7 --seconds 10 --nodes 1000000000 --cache-mib 128
```

`--reverse` reverses only the order of independent root searches. `--cache-mib 0`
disables caching. Full-horizon values should remain unchanged under either option.
Tests compare with an independent Python move generator and exhaustive small-tree
oracle, verify terminal/depth/budget semantics, cache/root-order invariance, and
monotonic complete-horizon bounds. Tests do not mutate production binaries.

Raw results and source hashes live under `results/`; labels prevent separate
configurations from overwriting one another. Reusing a label overwrites its run.
