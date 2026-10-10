# Pre-study report (ECUSTFD, hold-out)

| Model | Strategy | n | MAPE | Gain vs B0 | β | Bias | Validity | Repeat CV | $/1000 | P50 ms | P95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| google/gemini-3.5-flash | BBOX | 25 | 0.119 | 40.2% | 1.14 | +0.8% | 100.0% | — | $8.44 | 4387 | 11581 |
| anthropic/claude-sonnet-4.6 | S1 | 25 | 0.277 | -39.0% | 0.17 | -0.7% | 100.0% | — | $5.70 | 3935 | 5542 |

## Decision

Default/fallback chosen from models passing all gates, ranked by MAPE.

**Default:** `google/gemini-3.5-flash` (BBOX)
**Fallback:** none (no passing model from a different vendor).
