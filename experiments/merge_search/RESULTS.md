# Initial experiment results (2026-09-23)

Board `330043598671da10`, merge target 2048. One native thread, real board,
no evaluation pruning or masking. Existing 8192 is retained. See README.md for
definitions: successful target creation is not a claim of safe table handoff.

## Complete depth 7

128 MiB cache, fresh per root, 10 seconds / 1 billion calls allowed per root.
All four directions finished their full seven-move horizon.
No direction can reach the merge goal in this horizon under the model.
Since all horizon-success probabilities are zero, the selected policy minimizes
death probability; its death probability equals the unavoidable horizon risk.

| Root | Time seconds | Success by 7 | Minimum death by 7 | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Left | 0.2833 | 0% | 0.00008859375% | 99.99991140625% |
| Right | 0.1120 | 0% | 0.078900390625% | 99.921099609375% |
| Up | 0.2331 | 0% | 0.003116909722% | 99.996883090278% |
| Down | 4.0080 | 0% | 0% | 100% |

The horizon-risk tiebreak selects Down. Zero observed/optimal seven-move death
risk does NOT prove eventual merge success. The success bounds still overlap,
so eventual-success dominance is NOT certified.

## Deeper horizons

Same budget and cache; percentages below are lower bounds from explored policy
branches, not estimates of the real eventual success rate.

| Depth | Up lower | Down lower | Up complete | Down complete | Total root search seconds |
| --- | ---: | ---: | --- | --- | ---: |
| 8 | 0% | 0.430617665% | yes | no | 17.922 |
| 9 | 0.058207600% | 0.777420114% | no | no | 40.000 |
| 10 | 0.152916366% | 0.197541507% | no | no | 40.000 |

At depth 8, Left/Right/Up finish; Up's minimum death probability in eight moves
is 0.010706158%, while no Up branch reaches the goal within eight moves.
Down finds goal branches, but its search is truncated. At depths 9 and 10 every
root is truncated. Upper bounds for Up/Down there remain 100%.

All four runs rank Down as best observed; none proves eventual-success dominance.
Lower bounds need not increase between budget-truncated depths: ordinary DFS
spends its budget on different partial trees and policies. Depth 10's smaller
Down lower bound is NOT evidence of a lower real success probability.

## What this establishes

- The independent objective exposes rare failure without mixing it with score
  bonuses or post-merge positional evaluation.
- Complete shallow horizons cannot distinguish slow merge from no merge, but
  they can already expose unavoidable death branches.
- Successful-leaf termination alone does not solve branch explosion. In the
  complete depth-7 Down search, roughly 159 million player calls were made.
- This baseline is not production-ready: even 10 seconds PER DIRECTION does not
  resolve the deeper cases. Adaptive bound-guided work allocation and retention
  of prior-depth information are the next experiments, not implemented here.
- These timings are not directly comparable to production AI depths: this probe
  performs unmasked full enumeration with different values and exact-depth cache.

## Reproducibility and validation

- `results/larger_budget/raw.jsonl`: all four directions and detailed counters.
- `results/larger_budget/metadata.json`: build command and input hashes.
- `results/raw.jsonl` and `results/metadata.json`: initial smaller-budget pilot
  (32 MiB, 20 million calls per root, depths 3 through 10).
- Five unittest groups passed, including an independent Python move/tree oracle,
  goal/death/budget boundaries, cache and reverse-root invariance, monotonic
  complete-depth bounds, and invalid-target rejection.
- No production source, native module, WASM asset or Worker configuration changed.
