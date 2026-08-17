# Plumb — Benchmark & Scoring Contracts (FIXED)

These are the fixed contracts of the Plumb benchmark. Changes require a
DECISIONS.md entry with measured evidence. The generator, verifier, scorer,
contamination monitor, and eval harness all conform to this file.

## 1. Task definition

A task is one food label presented as:

- **image**: rasterized label render (front panel + Nutrition Facts + ingredients
  + net wt, or one composite panel), 224–1568 px on the long edge,
  PNG/JPEG, with optional OCR-noise/rotation/glare augmentation.
- **metadata**: optional structured fields as rendered (for oracle tracing).

The model must emit a verdict JSON object:

```json
{
  "verdict": "PASS" | "FLAG",
  "violations": [
    {
      "type": "VIOLATION_TYPE",
      "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
      "cfr": "21 CFR 101.12(a)",
      "observed": "single serving; 1/4 cup (60g)",
      "expected": "1 cup (245g) per 21 CFR 101.12(b)",
      "correction": "restate serving size as 1 cup (245g)"
    }
  ]
}
```

`PASS` ⟺ `violations == []`. At most one violation object per distinct
(location, type) on a label; a label may legitimately carry several.

## 2. Violation taxonomy + severity weights

| Class | id | Severity | Weight |
|---|---|---|---|
| Undeclared / mislabeled allergen | `ALLERGEN` | CRITICAL | 1.0 |
| Unauthorized health claim | `HEALTH_CLAIM` | CRITICAL | 1.0 |
| Nutrient-content claim above threshold | `CLAIM_THRESHOLD` | CRITICAL | 1.0 |
| Wrong serving size / RACC category | `SERVING_SIZE` | CRITICAL | 1.0 |
| Missing mandatory nutrient line | `MISSING_NUTRIENT` | HIGH | 0.8 |
| Wrong %DV (off by ≥1 pp) | `DV_ERROR` | HIGH | 0.8 |
| Non-compliant "healthy" criteria (2026) | `HEALTHY_RULE` | HIGH | 0.8 |
| FOP rule violation (2026) | `FOP_RULE` | HIGH | 0.8 |
| Missing top-9 allergen statement line | `ALLERGEN_STATEMENT` | HIGH | 0.8 |
| Metric/unit formatting | `FORMATTING` | MEDIUM | 0.5 |
| Statement of identity missing | `IDENTITY` | MEDIUM | 0.5 |
| Net quantity declaration error | `NET_QUANTITY` | MEDIUM | 0.5 |
| Minor DV rounding | `DV_ROUNDING` | LOW | 0.2 |
| Font size / legibility flag | `LEGIBILITY` | LOW | 0.2 |

Severity-weighted recall `R_w = Σ w·caught / Σ w` over violation-bearing tasks;
a violation is **caught** iff the model emits a violation object whose
`type` matches the oracle type (critical classes must also match `severity`).
HIGH-severity recall and CRITICAL-severity recall are reported separately.
Precision = caught / emitted (each emitted object matched to at most one
oracle violation; unmatched = false positive).

## 3. Oracle (verifier-as-oracle)

The oracle is the deterministic rule engine `plumb_forge.verify`.
Given the label's structured ground-truth fields, it recomputes every
violation independently of the model. The generator injects violations from
real patterns and runs every candidate task through the oracle with a
**self-check gate**: if oracle and generator disagree, the task is
regenerated (bounded tries, deterministic fallback). 100% oracle agreement is
a property, not an accident.

Oracle sources (public law):

- 21 CFR 101.12 — RACC serving-size reference amounts
- 21 CFR 101.9 — Nutrition Facts panel requirements, %DV
- 21 CFR 101.13 — nutrient content claims, implied claims
- 21 CFR 101.54–101.60 — specific nutrient content claims
- 21 CFR 101.14 / 101.71–83 — health claims, authorized list
- FALCPA + 2025 sesame addition — top-9 allergen declaration
- 2026 "healthy" final rule; 2026 front-of-package (FOP) final rule
- 21 CFR 101.105 — net quantity; 21 CFR 101.3 — statement of identity

## 4. Data & splits

- **Train/val**: stratified by (difficulty decile × violation class) under a
  seeded RNG (seed 7). Train/val disjoint on `task_id` AND on SHA-256
  value-level task signatures (contamination guard refuses to build on
  overlap > 0).
- **Benchmark**: ~1,000 held-out tasks from a different seed (777). Zero
  signature overlap with train verified.
- **Difficulty** is a 0–1 scalar: number of decoys, OCR noise level,
  violation subtlety (near-threshold claim values, near-boundary serving
  sizes, low-contrast text).

## 5. Contamination monitor

Task signature = SHA-256 over sorted (field, value) pairs of the label's
ground-truth fields (metadata excluded). A leak probe fires when any eval
signature matches the train set; must fire 1.0 on intentionally leaked sets
and 0.0 false-fire on clean sets (ROC check).

## 6. Benchmark rules

- Same system prompt for every model (worker contract, non-thinking mode for
  the verdict, `reasoning_effort` free for borderline cases).
- Frontier models scored with identical inputs and scoring code.
- Unparseable/empty model output = parse miss, counted honestly in parse rate.
- Run benchmark with concurrency + JSONL checkpointing (resumable).

## 7. Judge calibration (residual prose only)

Core verdicts use the oracle. Any LLM judge (e.g., for reason quality) is
pairwise, position-swapped, temp 0, majority over ≥3 samples, bootstrap CIs,
thresholds calibrated on a labeled seed. Target Cohen's kappa ≥ 0.85 vs a
golden-100 oracle-labeled set.