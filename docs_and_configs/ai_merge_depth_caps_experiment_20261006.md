# Merge-depth cap alternatives

These are isolated native experiments, not production changes or deployments.
The uncapped protected search is the experiment's reference, not a proven
optimal policy.

## Selected Implementation

After the experiment, the user selected the minimal `pending76` scheme.
Local/cloud native and WASM source files now use 7 with five empty cells,
6 with six or more, only while the goal is pending. Completed goals and
ordinary positions keep 4/3. Existing `masked_count < 4` and empty-cell
conditions remain unchanged; already-shallower depths are not increased.
No new parameter, chain-length calculation, scoring, pruning, cache or
task change is introduced. Production DLL/WASM files have not been replaced.
The four merge-depth regression tests pass, including 10,000 random boards,
cap boundaries, the reported depth-8 move, ordinary-stage invariance, and
local/cloud parity for native and host-compiled WASM source probes.

## Variants

- `old43`: disable protection; retain the previous 4/3 empty-slot cap.
- `full`: current protection; no empty-slot cap until the goal is produced.
- `global65`: previous unprotected search with its cap changed globally to 6/5.
- `pending65`: protected goal/masking, cap 6/5 while pending, then restore 4/3.
- `pending76`: protected goal/masking, cap 7/6 while pending, then restore 4/3.
- `chain_root`: pending cap based on the root's merge chain length.
- `chain_board`: recompute remaining merge chain length on the branch board.

In each pair, the first cap applies with exactly five empty cells, the second
with six or more. All caps retain the existing `masked_count < 4` condition.
Boards with fewer empty cells are unaffected. A cap is a ceiling, not a request
to deepen an already shallower node.

Pending variants differ from `full` only in the empty-slot depth guard. The
legacy/global variants disable protection and use the old masking behavior;
they should not be interpreted as isolating the cap from goal-safe masking
on every position. Compiler flags, cache, scheduling, scoring and pruning are
unchanged. Native single-thread comparisons are deterministic; four-thread
results retain existing task/shared-cache variation.

## Chain Definition

For each tile below the goal, an existing tile has cost zero. A theoretically
available merged tile has cost `1 + max(child costs)`. Pair the cheapest costs
at each exponent to find the shortest dependency chain that produces one new
goal tile; ignore already-existing tiles at or above the goal.

This measures sequential merge layers, not total merge count, Manhattan
distance or the number of moves needed to arrange tiles. Two independent
merges may happen in the same move. Movement restrictions and hostile spawns
are deliberately not represented, so this is only a lower bound.

The tested chain formula is `high = clamp(L + 2, 6, 8)`, with `low = high - 1`.
Ordinary/post-goal caps remain 4/3. This formula is experimental, not a
calibrated upper bound on required search depth.

## Reported Position

`c862b74095422e00`, depth 8, one thread, prune 0, urgency 1.5:

| Variant | Right | Up | Down | Selected | Seconds |
| --- | ---: | ---: | ---: | --- | ---: |
| Old 4/3 | 1192 | 627 | 632 | Right | 0.69 |
| Full protection | 1228 | 4448 | 4628 | Down | 7.34 |
| Pending 6/5 | 1192 | 2385 | 2186 | Up | 1.18 |
| Pending 7/6 | 1188 | 4032 | 4096 | Down | 2.80 |
| Root chain | 1228 | 4448 | 4628 | Down | 7.17 |
| Branch chain | 1188 | 4032 | 4096 | Down | 2.73 |

The initial chain length is six. Root-only length gives an 8/7 cap that
does not save nodes here. As merges occur, recomputing remaining length can
lower the cap, reducing nodes from 338,898,697 to 130,475,166 while preserving
the selected direction in this example. Scores do not remain identical.

Five interleaved four-thread repetitions have median times of 2.854s for
full protection, 0.455s for pending 6/5, 1.176s for pending 7/6 and 1.150s
for branch-chain caps. All five full/7-6/branch-chain runs select down;
all five 6/5 runs select up. The 7/6 saving is about 59% on this position,
not a claim of 59% across the workload.

For `330043598671da10`, depth 8, pending 6/5 preserves down, with score
7682 -> 7520 and time 15.63s -> 7.72s. Pending 7/6 has identical scores and
nodes to full protection here, so it does not save time on this position.

For `e931a88f102d0013`, depth 8, pending 6/5 preserves left (7469 vs the
reference 7687); legacy global 6/5 selects right instead. Thus blindly
replacing the old global cap is not interchangeable with a stage-aware cap.

## Validation Scope And Limits

The experiment compares six reported boards at depths 7 and 8, five
four-thread repetitions for the main position, and 276 additional cases
at depth 6. Depth-6 comparisons mainly stress 6/5 and short-chain caps;
7/6 cannot usually truncate a depth-6 search, so equality on that cohort
is not evidence of safety at deeper horizons.

Some additional boards change direction under 6/5 or the short-chain
formula. No cap guarantees the reference ordering, and the reference
ordering itself is not proof of maximum eventual merge probability.
Scores are heuristic values, not success probabilities. Reduced horizon
can lower scores even when the same action remains preferred.

No query times out under the 40-second per-query limit. All 12 focused
depth-7/8 comparisons retain the full-reference direction under pending
7/6 and both chain formulas. Pending 6/5 differs in two comparisons
(the main position at both depths). Across the 276 depth-6 cases, 103
have a pending goal: 6/5 changes seven selected directions, and each
chain formula changes six. Pending 7/6 changes none on that cohort, but
its cap does not truncate these depth-6 queries, as noted above.

Over the 12 focused cases, pending 7/6 uses 86.5% of the full-reference
aggregate nodes; branch-chain caps use 81.6%. For some positions the
7/6 cap is never reached and no time is saved. A conservative fixed
7/6 stage-aware cap is the simplest next candidate; chain length alone
is not enough to authorize more aggressive truncation on short chains.

## Bounded First-Decision Playouts

For the reported board, the first direction is fixed to each variant's
deterministic depth-8 decision, without rerunning that search for every seed.
Subsequent decisions use depth 5, for up to 20 moves. The same 32 PRNG seeds
are used per arm, uniformly selecting empty cells with a 10% chance of a 4.
This is a counterfactual first-move comparison, not a complete depth-8 or
iterative-search policy evaluation.

| Variant | First move | Completed | Dead | Unresolved | Total moves |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | Down | 32 | 0 | 0 | 335 |
| Pending 6/5 | Up | 32 | 0 | 0 | 349 |
| Pending 7/6 | Down | 32 | 0 | 0 | 335 |
| Branch chain | Down | 32 | 0 | 0 | 335 |

These samples do not establish that up is a losing move or that any arm
has 100% eventual success probability. They distinguish a changed heuristic
ranking from demonstrated deaths, and do not resolve rare hostile spawns.

## Reproduction

```powershell
python tests/experiment_merge_depth_caps.py --output output/merge-depth-caps-20261006 --repeats 5 --corpus
python tests/experiment_merge_depth_caps.py --output output/merge-depth-caps-20261006 --rollouts-only
```

Source copies, binaries and per-case results remain in the output directory.
The live Worker remains active, so timings do not have exclusive CPU access.
