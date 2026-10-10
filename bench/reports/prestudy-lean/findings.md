# Pre-study findings (ECUSTFD, lean hold-out sweep)

**Superseded, 2026-10-10 — see §6.** Sections 1–5 below are the original
write-up for the direct-mass-estimation strategies (S1/S3) and still stand
as an accurate description of *those* strategies: no model clears the
decision gates asking a VLM to estimate mass directly. But the
"resolution floor" explanation in §3 does not hold as a limit on the
architecture as a whole — §6 shows the size signal is recoverable from the
same photos once localisation (find the coin and the fruit) is separated
from measurement (compute mass in code from the boxes).

See `report.md` for the S1/S3 metrics table and `decision.md` for the
mechanical gate output on those strategies. This file adds the *why*
behind the numbers, using targeted diagnostic calls made after the
hold-out sweep completed.

## 1. What was run

Hold-out sweep, strategy S1 and S3, 25 hold-out objects (apples, bananas,
oranges, pears, kiwis; mean of side+top views per object), repeats=1:

| Model | Provider pin | Strategy used | Cost (hold-out sweep) |
|---|---|---|---|
| `anthropic/claude-sonnet-4.6` | `amazon-bedrock/global` | S1 (S3 did not improve β) | $0.57 |
| `google/gemini-3.5-flash` | `google-vertex/global` | S1 (S3 did not improve β) | $0.93 |

Both ran at 100% validity (no refusals, no schema failures, no truncation) —
the models answer confidently and in-schema; they are simply not tracking
true mass.

| Model | n | MAPE | Gain vs B0 | β | Bias | $/1000 |
|---|---|---|---|---|---|---|
| gemini-3.5-flash | 25 | 19.9% | -4.6% | 0.33 | +6.7% | $9.27 |
| claude-sonnet-4.6 | 25 | 27.7% | -45.7% | 0.17 | -0.7% | $5.70 |

Both fail the primary gate (gain ≥30% vs. the trivial "predict the
fruit-type mean" baseline B0) and the β gate (0.7–1.3) by a wide margin.
Neither is close to passing; this is not a borderline result.

Four further candidates were screened for basic viability (schema support,
routing, cost) but not run on the full hold-out set once the pattern below
was established: `openai/gpt-4o-mini` (works, but returned a *flat* 150.0g
prediction across every test image — the most extreme version of the same
failure), `openai/gpt-4.1` and `google/gemini-3.1-flash-lite` (both weak,
similarly clustered), and two next-gen reasoning models
(`anthropic/claude-sonnet-5`, `openai/gpt-5`) that required a proxy fix
(§2 below) to even route and were not evaluated on accuracy — both are
materially slower (up to 83s/call) and pricier, and nothing in the evidence
so far suggests reasoning effort would fix a perception problem. Two
candidates (`z-ai/glm-4.5v`/`4.6v`) have no OpenRouter endpoint supporting
strict structured output; `qwen/qwen3-vl-8b-instruct` is schema-compatible
but was blocked by an upstream shared-pool rate limit on Parasail's only
endpoint for that model.

## 2. Why: diagnostic evidence beyond the metrics

The β values alone say "not tracking size." Three follow-up diagnostics on
the same three hold-out apples (212.5g / 238.0g / 293.5g true weight — a
38% spread) show *why*, using the actual `observations` text the models
produce (not persisted by the sweep pipeline; recovered by replaying the
same request path — `imaging.preprocess.normalize` →
`observation.adapter.observe` — outside the committed CLI, ~$0.03 total).

**(a) S1, single view.** Claude's stated diameter judgement is identical
across all three apples: *"roughly 3-3.5 coin diameters wide (75-85mm)"* —
for an object that is 38% heavier than another. Gemini's diameter estimate
*decreases* as true weight *increases* (90mm → 95mm → 80mm-ish framing for
212.5g → 238.0g → 293.5g) — an inverse relationship, not noise.

**(b) S2, explicit coin-ratio measurement.** `pre-study.md` §3 drops S2 for
ECUSTFD by design (no real-world length/diameter ground truth to calibrate
or validate against — only pixel bounding boxes). For this one diagnostic
question only — does *forcing* an explicit measurement change the pattern,
regardless of calibration — a scratch S2 prompt was used outside the
committed pipeline (not added to `observation/prompts/`, not run through
`prestudy run`) asking for an explicit coin-width ratio before converting
to mass. It did not help: Gemini gave the lightest and heaviest apple the
*same* length, diameter, and derived mass, bit-for-bit. No S2 support is
added to the codebase as a result — the spec's exclusion stands.

**(c) S1, two views (side + top) combined.** Claude's prediction became
*more* flat, not less: all three apples get the identical 220.0g estimate.
Gemini kept the same inverse pattern (heaviest apple → lowest estimate).

## 3. Working hypothesis: a resolution floor, not a prompt problem

Mass scales roughly with the cube of linear size for self-similar shapes.
The 38% weight spread used in the diagnostic corresponds to only a ~11%
diameter difference ((293.5/212.5)^(1/3) ≈ 1.11×) — a few millimetres on
an apple ~8cm wide, in a single photo, at the resolution and framing used
here. Three independent attempts to extract that signal (free-form
estimate, forced explicit coin-ratio, two combined views) all converged on
the same outcome for both of the two best-performing candidates: a stable
category judgement ("this is a normal-sized apple") rather than a
measurement that tracks the specific specimen. This reads as a real
limit on what single/dual-photo VLM size discrimination can resolve for
same-type produce, not as an artifact of strategy S1 specifically.

This is consistent with, and explains, the uniformly low β across every
model actually screened in this pre-study (including the flat 150.0g
gpt-4o-mini result) — it is not one weak model, it is a pattern.

## 4. What this does not settle

- **Not tested**: a prior-informed strategy that leans on typical-weight
  ranges rather than asking for a fresh per-image measurement (S3 was run
  mechanically but its prior source and tuning were not iterated on once
  the S1/S2 pattern was clear).
- **Not tested**: whether a much larger or more distinctive size spread
  (e.g. comparing a 150g apple to a 400g apple, rather than two
  mid-range specimens) is resolvable — the diagnostic deliberately used
  typical within-type variation, which is the harder and more realistic
  case, but a wider-spread check would show whether there is *any*
  resolvable signal at all.
- **Not run to completion**: `claude-sonnet-5` / `gpt-5` accuracy, and
  `qwen3-vl-8b-instruct` (rate-limited every attempt). None of the
  evidence gathered suggests these would behave differently, but that is
  an assumption, not a measured result.

## 5. Recommendation

Stop before committing to Phase 0 production architecture on the current
evidence. Before spending further model-sweep budget, prioritise:

1. The wider-spread diagnostic in §4 — cheap (a handful of calls) and
   would directly test whether the resolution-floor hypothesis is right,
   or whether these two candidates are specifically weak.
2. If the resolution floor holds, revisit the product design itself: a
   strategy that does not require single-photo absolute-size perception
   (e.g. leaning harder on class-typical priors plus a cheap
   measured-reference input, rather than photo-derived linear
   dimensions) may be more productive than continuing to screen models.

## 6. Correction (2026-10-10): the resolution floor does not hold — it's a strategy problem, not a signal problem

An external review (`review.md` §4) ran the coin-scaled geometry formula
(top-view `(w·h)^1.5`, side-view `w·h·min(w,h)`, combined top+side, §4.1
 of `review.md`) against the dataset's *ground-truth* annotation boxes
instead of a VLM's mass estimate: **8.1% MAPE, β=0.98** on the same 25
hold-out objects. β≈1 means the size signal **is** present in the pixels
at a resolution far better than §3's "resolution floor" hypothesis
claimed — §3's reasoning was sound given what was tested (direct mass
estimation), but the conclusion it supported (a perceptual ceiling) was
wrong. The real bottleneck is that asking a VLM for a holistic mass
number lets it fall back to a category-typical guess instead of
measuring; asking it to *localise* two objects is a different, and much
easier, task.

**E1** (`review.md` §5) tested the realistic version of that 8.1% bound:
ask each VLM for normalised `[0,1]` bounding boxes of the coin and the
fruit (new `BBOX` strategy, `src/food_vision/observation/schema.py`),
then compute mass in code from those *predicted* boxes using the same
formula and the same dev-fitted per-type constants as the ground-truth
oracle (`bench/scripts/bbox_vlm_eval.py`). 100/100 calls `ok`, cost $0.78,
full results in `bbox_vlm_eval.json`:

| Model | n | MAPE | MedAPE | Bias | β |
|---|---|---|---|---|---|
| Ground-truth-box oracle (upper bound) | 23 | 8.1% | 7.5% | -3.3% | 0.98 |
| `google/gemini-3.5-flash`, VLM boxes | 25 | **11.9%** | 8.6% | +0.8% | 1.14 |
| `anthropic/claude-sonnet-4.6`, VLM boxes | 25 | 89.3% | 80.3% | +72.2% | 1.14 |

For comparison, direct mass estimation (S1) on the same 25 objects scored
19.9% (Gemini) and 27.7% (Claude) MAPE with β=0.33/0.17 — **Gemini's
box-then-compute MAPE (11.9%) beats its own direct-estimate MAPE (19.9%)
and comes within 3.8 points of the ground-truth oracle**, with β rising
from 0.33 to 1.14. This is a materially different, more actionable result
than §3's "stop and rethink": the architecture the external review
proposed (VLM localises, code measures) works, at least for this model.

**Claude's 89.3% MAPE is a specific, diagnosed failure, not a second
counter-example to the architecture** — but this section's original
diagnosis of *why* was wrong; see §7's correction below.

**Revised recommendation**: do not stop. The direct-mass strategies
(S1/S3) are the wrong strategy, not evidence that VLM-assisted size
estimation is infeasible. See §7's revised next-steps list.

## 7. Second correction (2026-10-10): §6's "foreshortened coin" diagnosis was factually wrong, and E1 is now a formal, reproducible pipeline strategy

A second external review of §6 corrected the claimed mechanism and
changed the priority order of what to do next. Both corrections were
independently re-verified against the real ground-truth annotation XML
(not taken on faith) before acting on them.

**The coin does not foreshorten to an ellipse in side-view photos — this
was checked and is false.** Ground-truth coin-box aspect ratio across
all 1,422 side-view and 1,421 top-view annotations: median 1.050 (side),
1.044 (top), p10–p90 of 1.01–1.12 in both. The coin reads as
near-circular in *every* real photo, side or top. Claude's VLM-predicted
median aspect ratio of 2.29 in side views is therefore a genuine
localisation or coordinate-convention bug, not a physical effect — §6's
explanation is retracted. The review's working hypothesis: different
providers are trained on different "native" box conventions (e.g.
Gemini's `[ymin,xmin,ymax,xmax]`-on-1000 vs. pixel coordinates with an
explicit image size), and asking every model for the same shared `[0,1]`
fraction format may systematically penalise models whose native
convention differs — not yet tested; see the next-steps list below.

**"E1" is no longer a one-off script result — it's now computed by the
same formal `analysis.py`/`report.py`/`decide()` pipeline S1/S3 already
used** (`analysis.bbox_predictions_per_object`, `compute_metrics(...,
strategy="BBOX", kfit=...)`, `choose_strategy_per_model` generalised to
a 3rd candidate). This needed two additions the original E1 script
didn't have: `run.ResultRecord` now persists `image_width_px`/
`image_height_px` at run time (no more re-deriving dimensions from raw
image files at analysis time), and the ground-truth-fitted per-type `k`
is a committed artifact (`bench/scripts/fit_k.py` → `kfit.json`), not
recomputed from the Annotations directory on every analysis run. The
existing 100 real `BBOX` rows (from before these fields existed) were
backfilled in place (`bench/scripts/backfill_image_dims.py`,
sha256-verified against the actual sent bytes, 100/100 recovered) rather
than discarded or re-paid-for.

Running the real pipeline on the real, backfilled data reproduces §6's
numbers exactly and now auto-generates a decision:

| Model | Strategy | n | MAPE | Gain vs B0 | β | Bias | Validity |
|---|---|---|---|---|---|---|---|
| `google/gemini-3.5-flash` | BBOX | 25 | 0.119 | 40.2% | 1.14 | +0.8% | 100.0% |
| `anthropic/claude-sonnet-4.6` | S1 | 25 | 0.277 | -39.0% | 0.17 | -0.7% | 100.0% |

**Decision: default `google/gemini-3.5-flash` (BBOX)** — passes every
§5 gate (gain ≥30%, β∈[0.7,1.3], |bias|≤10%, validity ≥98%). No
fallback: Claude's BBOX metrics exist but lose to its own S1 on MAPE
(`choose_strategy_per_model` picks S1 for Claude), and Claude's S1 fails
the gain/β gates, so no second vendor passes. This is the formal
pipeline's own output, not a hand-edited `decision.md`.

**Revised next steps, in order** (supersedes §5 and §6's lists):

1. Investigate Claude's (and, once its rate limit clears, Qwen3-VL's)
   box coordinates in their own native convention rather than the
   shared `[0,1]` fraction format, converting to a common representation
   in code — not yet attempted; the current `[0,1]`-for-everyone schema
   is unchanged pending this.
2. Re-run E1 across all photo variants (`--max-variants-per-object
   all`, already implemented per §E3) rather than one variant per
   (object, view) — cheap, not yet done for BBOX specifically.
3. Fit `k` per model from that model's own dev-split BBOX predictions
   (not just the one ground-truth-fit reference value) — corrects
   systematic box bias per model; needs dev-split BBOX attempts, which
   the current hold-out-only sweep doesn't have. Documented as a real
   gap, not implemented.
4. Own-photo transfer check with a real card (review's E6) — the
   dev-fitted `k` is specific to this dataset's camera distance/coin and
   won't carry over; needs real photos only the product's actual user
   can supply.
