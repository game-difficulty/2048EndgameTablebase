# 3x3 Analysis Result Cards

Profile `3x3-v1` applies to Play analyses with variant/pattern `3x3` and
target `1024` or `sum-1790`. The 4x4 grading formula is unchanged.

## Metrics and Points

| Points | Combo | Perfect rate | Geometric accuracy |
| --- | --- | --- | --- |
| 1 | 13 | .63 | .9940000 |
| 2 | 17 | .67 | .9979000 |
| 3 | 27 | .73 | .9992000 |
| 4 | 90 | .85 | .9999300 |
| 5 | 158 | .88 | .9999840 |
| 6 | 186 | .895 | .9999925 |
| 7 | 203 | .91 | .9999957 |
| 8 | 225 | .92 | .9999976 |
| 9 | 290 | .93 | .9999990 |

Interpolate linearly between adjacent anchors. Below the first anchor gives
zero points; values above the last anchor are capped at nine. No rounding is
applied before summing points or assigning the grade.

Use the product of included stage fits to the power `1 / N`, calculated via
logarithms, where `N` is the number of categorized moves across those stages.
The tile-goal analyzer's skipped certain moves and ungraded warmup moves are
excluded from `N`. A zero fit yields zero geometric accuracy. Valid short
stages are included without the 4x4 minimum of twenty moves.

## Time Bonus

Use the whole recorded game duration and final board sum, not just analyzed
steps. Exactly 1024 and 1536 enter the higher progress band.

| Final sum | +1 | +2 | +3 |
| --- | --- | --- | --- |
| <1024 | 4:30 | 3:15 | 2:00 |
| 1024 to <1536 | 11:00 | 8:00 | 5:00 |
| >=1536 | - | - | Always |

Faster times score more. Interpolate between time anchors, cap at three, and
award zero for slower-than-first-anchor or missing/incomplete timing. The
sum >=1536 bonus does not require timing. Never interpret missing time as zero.

## Grades and Existing Records

Sum the three metric scores and time bonus (maximum 30). Lower inclusive
grade bounds: F=0, E=1, D=3, C=5, B=7, A=9, S=12, SS=15, SSS=17.

Existing paid analysis summaries are upgraded without reanalysis or charges.
Legacy native runs can supply full timing from their recorded elapsed field;
legacy imported runs without full timing receive no timed bonus. The grading
version also refreshes the summary-based strength index. Missing mean move
time leaves a result unranked under the existing index schema.

Cards show geometric accuracy (five decimal percent places), Perfect rate,
max combo, evaluated moves, full duration, and final board sum. Internal point
totals remain server-side, matching the existing 4x4 policy.
