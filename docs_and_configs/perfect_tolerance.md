# Perfect Tolerance by Success-Rate Dtype

The analysis policy lives in `docs_and_configs/performance_evaluations.json`.
`perfect.tolerance` is the fallback for missing/unknown dtype;
`perfect.tolerance_by_dtype` provides overrides:

| Reader dtype | Default absolute success-rate tolerance |
| --- | --- |
| uint32, float32, 1-float32 | 3e-10 |
| uint64, float64, 1-float64 | 1e-14 |

The resolver also accepts f32/f64 and 1-f32/1-f64 aliases. No formation-specific
override is used, including 3x3_1024 and 3x3_sum-1790. Values must be finite and
nonnegative; invalid settings fall back to the corresponding built-in default.

Batch analysis uses the dtype returned by the Reader for each step. Perfect,
combo and goodness of fit use the same comparison. A Perfect step contributes
a ratio of exactly 1; otherwise its ratio is selected probability / best
probability. Complement-format rates are restored before comparison. Existing
certain-step skipping rules are unchanged.

Configuration is loaded at Python process startup. The cloud browser-side
tester imports the same JSON at build time, so config changes require a
frontend rebuild as well as restarting affected Python processes.

Legacy .rpl files still store uint32 probabilities scaled by 4e9. Reanalysis
of those stored values explicitly uses the uint32 policy, regardless of the
source table's current dtype. They cannot recover float64 precision discarded
during export. New batch-analysis summaries and posters use the Reader's
full-precision values directly. Existing saved analyses are not retroactively
recomputed; run analysis again to obtain results under the new policy.
