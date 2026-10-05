# Merge-depth protection: tree growth versus policy overhead

## Question

The earlier report placed a 1.88x corpus-wide median node ratio beside
0.52s -> 3.13s for one depth-8 position. They describe different populations.
For `c862b74095422e00`, the earlier four-thread node counts were actually
34,211,133 -> 344,526,976: 10.07x, versus 6.04x elapsed time.

## Controlled Experiment

All variants use the same compiler flags, scoring, 32-bit cache signature,
task structure, pruning, urgency 1.5 and search depth. Builds are isolated.

Three guards were compared:

1. Original implementation: scans the goal at empty-slot depth-cap sites,
   including depths already below the cap; sometimes scans twice.
2. Single scan: consolidates the two empty-slot conditions.
3. Depth guard: only scans when current depth exceeds the applicable 3/4 cap.

Variant 3 is mathematically equivalent: a cap cannot change a depth already
at or below it. Goal definition, masking and post-goal behavior are unchanged.

Seven interleaved repetitions per variant, toggle and thread count, without
instrumentation, gave these medians for `c862b74095422e00`, depth 8:

| Protection / guard | 1 thread, seconds | 4 threads, seconds |
| --- | ---: | ---: |
| Off, original guard | 0.6810 | 0.4444 |
| On, original guard | 7.2701 | 2.9244 |
| On, single scan | 7.4151 | 2.8124 |
| On, depth guard | 7.2842 | 2.8482 |

The same-tree depth guard removes almost all policy scans without materially
changing elapsed time. The four-thread median improvement is about 2.6%,
not enough to explain the several-second difference. This is not evidence
of zero overhead, nor a guarantee of a speedup on every device.

## Actual Call Counts

Separate instrumented single-thread builds count function entry and scans.
Instrumentation timings are not used for speed claims.

| Metric | Protection off | On, original | On, depth guard |
| --- | ---: | ---: | ---: |
| `search_branch` calls | 13,637,570 | 123,099,037 | 123,099,037 |
| `search_ai_player` calls | 16,898,079 | 176,583,947 | 176,583,947 |
| Combined calls | 30,535,649 | 299,682,984 | 299,682,984 |
| Goal checks | 2,120,416 | 23,121,608 | 596 |
| Reported nodes | 34,210,135 | 338,898,697 | 338,898,697 |

Actual recursive entry counts grow 9.81x. Reported nodes are a different
accounting metric, so these direct counters additionally verify the tree
expansion rather than relying solely on the public node count.

Single-thread on-guard variants have identical four-direction scores:
`[-131072, 1228, 4448, 4628]`, selecting down. Four-thread runs have small
existing task/shared-cache variations; exact parity is checked single-thread.

The 306-case corpus was checked with protection both off and on: 612 paired
comparisons (1,224 searches), at depths capped at 5, preserve best move,
four scores, target and nodes exactly. The first 32 cases in both modes
also verify the optimized production sources against the diagnostic build
and local/cloud native/WASM-source parity (64 four-copy comparisons).
Higher-depth parity is separately covered by the four instrumented cases
above, including the reported position at depth 8.

## Changes And Reproduction

The depth guard is synchronized in local/cloud, native/WASM source files.
No scoring, pruning, cache or scheduling change is made. Existing production
DLLs and frontend WASM files are not replaced; the live Worker is not restarted.

```powershell
python tests/benchmark_merge_policy_overhead.py --output output/merge-depth-speed-20261006 --repeats 7
python tests/benchmark_merge_policy_overhead.py --output output/merge-depth-speed-parity-20261006 --check-corpus
python -m unittest discover -s tests -p test_ai_merge_depth.py -v
```

Timing runs are sequential but the production Worker remains active; CPU
access is not exclusive. Results and mechanical diagnostic source copies are
preserved under the indicated output directories.
