# Pre-study report (ECUSTFD, hold-out)

| Model | Strategy | n | MAPE | Gain vs B0 | β | Bias | Validity | Repeat CV | $/1000 | P50 ms | P95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| google/gemini-3.5-flash | S1 | 25 | 0.199 | -4.6% | 0.33 | +6.7% | 100.0% | — | $9.27 | 6913 | 17835 |
| anthropic/claude-sonnet-4.6 | S1 | 25 | 0.277 | -45.7% | 0.17 | -0.7% | 100.0% | — | $5.70 | 3935 | 5542 |

## Decision

No model passes all gates (gain>=30%, beta in [0.7,1.3], |bias|<=10%, validity>=98%). Stop and rethink before Phase 0.

**Result: stop.**
