# Merge-goal depth protection validation (2026-10-06)

## Release Status

The measurements below describe the earlier uncapped-protection experiment,
not the final v13.6.1 policy. The shipped policy caps pending-goal searches
at 7 with five empty cells and 6 with six or more; ordinary/completed-goal
caps remain 4/3. Existing masking-count conditions and search time budgets
remain unchanged. See `ai_merge_depth_caps_experiment_20261006.md` for the
selected implementation. Referenced test scripts and outputs are retained
locally and are not distributed with the repository or release packages.

## Implementation

- Shared `native_core/include/MergeDepthPolicy.h` is used by native and WASM
  search in both repositories. Scoring, pruning thresholds, cache entries and
  task scheduling are unchanged.
- At each search start, virtually carry pairs in the initial board's tile
  counts. The largest newly producible tile is the fixed goal for that search.
  Only goals 256 and above enable protection, matching the existing mid-size
  merge-depth adjustment boundary and avoiding ordinary 2/4 merges.
- This tests numerical completeness, not whether a physical merge is certain.
  It does not assume future spawns or assign an unearned merge bonus.
- Until the goal is achieved, empty-cell count does not cap depth to 3 or 4.
  Normal depth decrement, existing pruning and cancellation remain active.
- Completion means mass in tiles at or above the goal exceeds its root value.
  That mass is preserved when existing large tiles merge, and grows only when
  smaller tiles cross the goal boundary. Subsequent merges cannot undo it.
  Spawning 2/4 cannot cross a 256+ goal. Completion is board-derived, so no
  path-dependent cache key or additional cache storage is needed.
- Dynamic masking preserves all sub-goal constituents. Root reference mass
  is computed after masking, and rebased if search falls back to the real board.
  `start_search` already clears the cache between search contexts.
- After completion, the original live-branch depth cap resumes. With no goal,
  the original depth/masking behavior is retained.
- `protect_merge_depth` defaults to true in both Python and WASM bindings;
  setting it false provides an exact old-policy control. Python also exposes
  read-only `merge_target`. No caller-side parameter changes are required.

## Test Coverage

1. 10,000 deterministic random boards, 232,977 assertions: independent
   pair-reduction goal oracle, mass monotonicity, spawning invariance, existing
   large tiles, masking/rebasing, 32768 saturation, reset, disabled mode and
   completed-goal equivalence to the original cap.
2. 306 A/B search cases, each with protection disabled/enabled: 128 samples
   reconstructed from actual live run `1e0600a6-bc10-43ed-851b-a7facf187adf`
   (10,765 moves), 64 distinct-tile controls, 64 shuffled complete chains,
   eight reported problem boards at depths 5-8, prune/urgency variants and
   four-thread controls. Corpus seed: 20261006.
3. 173 ordinary cases: scores and node counts identical. 133 merging cases:
   33 changed moves. No 30-second per-search timeouts in either arm.
4. First 64 cases also compared to an independently compiled pre-change
   native source snapshot: disabled-policy scores and node counts identical.
5. Local/cloud native implementations and local/cloud WASM-source
   implementations agree within their respective variants on 32 selected
   cases (parity comparisons capped at depth 6). Native/WASM are not assumed
   to have identical task traversal or merge-depth behavior.
6. Actual Emscripten WASM binary: 36 searches, including the reported board
   at depth 8 in both modes, plus timeout and reset smoke checks. Passed.
7. Isolated native DLL builds from both repositories: 20 existing/new AI
   tests each passed. Cache-signature fixed-score regression explicitly
   disables this new policy, so its original score expectations still verify
   the cache change alone. Native cancellation also checked independently.
8. Seeded playouts: 384 runs on six boards (32 seeds per arm, fixed depth 5,
   single thread, at most 20 moves); 128 additional runs on the borderline
   sample (64 seeds per arm, at most 40 moves); 32 runs isolating the original
   board's depth-8 first decision, followed by depth-5 play (four threads).
   Total: 544 bounded playouts. These are merge completion tests, not complete
   game win-rate estimates.

## Key Results

For `c862b74095422e00`, goal 1024, fixed depth 8:

| Variant | Right | Up | Down | Selected |
| --- | ---: | ---: | ---: | --- |
| Native, old, 1 thread | 1192 | 627 | 632 | Right |
| Native, new, 1 thread | 1228 | 4448 | 4628 | Down |
| Native, old, 4 threads | 1185 | 627 | 632 | Right |
| Native, new, 4 threads | 1257 | 4451 | 4629 | Down |
| Actual WASM, old | 1191 | 638 | 633 | Right |
| Actual WASM, new | 1191 | 4457 | 4628 | Down |

The bad child `c8621b749542002e` remains negative and selects left.
In 32 single-thread depth-5 playouts, both arms completed 2, died 16 and
remained unresolved 14: the change does not erase its existing danger.

For the original board's depth-8 first decision, followed by depth-5 play,
16 four-thread seeds per arm completed the goal within 20 moves in 12 old
versus 16 new runs. Neither arm died within that finite observation window.

`39107a1266d813b1` completed 16 old versus 15 new runs in the initial 20-move
window. Extending to 64 seeds and 40 moves gave identical outcomes: 40
completed, 7 dead, 17 unresolved per arm. This distinction prevents treating
a window-boundary delay as a proven eventual success-rate regression.

## Cost And Limits

- Median node ratio among merging cases: 1.88x. Maximum ratio: 361.76x on a
  cheaply truncated control; relative cost alone does not describe latency.
  This is a corpus statistic, not the reported board's multiplier. Follow-up
  same-tree overhead experiments are in `ai_merge_depth_overhead_20261006.md`.
- Original sample depth 8: approximately 34.2M -> 338.9M nodes (1 thread),
  and 34.2M -> 344.5M (4 threads). Four-thread wall time was approximately
  0.52s -> 3.13s. Native single-thread and actual WASM were approximately
  7.79s and 9.69s respectively with protection.
- Benchmarks did not have exclusive CPU access. Wall times are illustrative;
  node counts are the primary controlled cost comparison.
- Equalizing the pre-goal horizon does not fix all other heuristic pruning,
  cache approximation or finite-horizon limitations. No claim of universally
  increased eventual merge probability is made from these samples.
- Existing iterative-search time budgets are unchanged. Protection is not
  an instruction to force depth 8 when the allotted time cannot afford it.

## Reproduction And Artifacts

```powershell
python -m unittest discover -s tests -p test_ai_merge_depth.py -v
python tests/benchmark_ai_merge_depth.py --output output/merge-depth-validation-20261006
```

Detailed per-case results, bounded playout records, actual WASM results and
compiled native/WASM test artifacts are under
`output/merge-depth-validation-20261006/`. Production frontends were not
deployed and the live Worker was not restarted; its local DLL remains
unchanged (SHA-256 `C02143B1CE8EC99E40861810FFF1344CEC87D5D1B9563E1288D644C886312D3F`).

The cloud repository's older CMake configuration initially wrote the test
DLL to its source directory. The new binary was preserved in the isolated
`modules-cloud/` directory, and the cloud source-directory DLL was restored
from `output/backend-review-20261001/` (SHA-256
`B2633AC6CA081F16D7307533285CFB7B71485760E51011CBF04F964C2DD773D8`). Subsequent
validation output is redirected for both generic and Release properties.
