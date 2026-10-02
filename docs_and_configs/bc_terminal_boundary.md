# BC terminal spawn-4 boundary fix

BC previously generated through `final_sum - 2`, omitting the final spawn-4
output retained by EX. The solver then supplied a virtual empty layer at a
sum that should contain real terminal wins, reducing upstream probabilities.

Generate through `final_sum` and restrict both last layers to successful
boards. Apply the terminal boundary to resident and single carry outputs
as well as the shared family path. Keep the solver's empty sentinel beyond
the real endpoint. Python's expected generated-layer count increases by one.
This does not change the meaning of `extra steps`.

Existing solved tables are not automatically invalidated. Recompute affected
backward layers; merely adding the missing terminal file does not repair
probabilities already propagated from the empty boundary.

## Validation (2026-10-03)

`tests/test_bc_terminal_boundary.py` forces all nine combinations of resident,
single and family generation/solving. It checks actual routes in CSV statistics,
nonempty terminal payload, identical terminal row counts, terminal values of 1,
and effective query values against EX with tolerance 6e-9. The old boundary
fails the nonempty-payload assertion. The fixed module passes all nine cases.

An independent full free10-128 run without compression or pruning matched
Classic and EX exactly for board `00201211ff30ffff`:

| Direction | Corrected BC / EX / Classic |
| --- | --- |
| down | 0.999998944 |
| left | 0.999998234 |
| right | 0.999991755 |
| up | 0.996939723 |

Layer 98 contains 76,779,067 rows; layer 99 contains 53,984,387 rows.
Both terminal payloads contain only probability 1. Before the fix, layer 99
was virtual empty, and the up/right discrepancies exceeded tolerance.
Changing only the success-check starting threshold did not change those
incorrect query values; that hypothesis was ruled out by a separate full run.

Evidence is retained under
`C:/2048_tables/test/free10-validation-20261003/bc-terminal-comparison.json`
and `bc-terminal-recheck/02_128_plain_bc`. Original failed results are preserved.
Full production-size forced-route runs are scheduled separately after the
ongoing 13-case validation; the nine-route regression above uses a small case.

Run the focused regression from the repository root with the built native module:

```powershell
C:/Anaconda/python.exe -m pytest tests/test_bc_terminal_boundary.py -q
```
