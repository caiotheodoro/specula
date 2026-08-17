# Plumb

**Fine-tune a multimodal Qwen3.8-27B (28B, Apache-2.0) on Modal/GCP free
credits so it beats frontier models at FDA food-label compliance review.**

A label image goes in → a cited compliance report comes out
(`PASS | FLAG` + per-violation severity, exact 21 CFR citation, correction).
Full process automation: intake → rule application → verdict → report.

This is the first of four parallel repos (`plumb`, `seam`,
`tally`, `attest`) built on one shared methodology (see `docs/methodology.md`):
seeded synthetic generator + deterministic verifier-as-oracle + RLVR, the
recipe that already made a 1.7B LoRA beat a frontier model on financial
reconciliation (ReconForge). Here the model is 16× bigger, multimodal, and
trained in the cloud on free credits.

## Why this niche

- **The multimodal 27B is the weapon, not a coincidence.** Labels are images.
  Frontier VL models hallucinate exact thresholds and citations; we fine-tune
  against the regulation as a deterministic oracle.
- **"Perfect level" is approachable.** The whole rule surface is public law:
  21 CFR Part 101 (labeling) + 21 CFR Part 104 (nutrient content claims) +
  FALCPA allergen rules + 2026 "healthy" and front-of-package (FOP) rules.
  A verifier engine recomputes every flag from the label's structured fields,
  so the RLVR reward is exact, not judged.
- **Public ground truth exists.** openFDA recalls, CAERS, and published FDA
  warning letters give real violation patterns — zero proprietary data.
- **Recurring market.** Every CPG launch needs label review ($500–5k/label
  consulting); rule churn (2026 "healthy", FOP) makes it recurring; pre-market
  labels are confidential → on-prem deployability is a moat.

## The process (end-to-end, fully automatable)

1. Intake label image(s) (front panel, Nutrition Facts, ingredients, net wt).
2. Normalize (OCR where needed; the model reads pixels directly).
3. Extract structured fields: product name, serving size + household measure,
   servings per container, Nutrition Facts panel (calories + all mandatory
   nutrients + %DV), ingredient list (allergens), claims (nutrient content,
   health), net quantity, statement of identity, FOP symbols.
4. Apply rule engine (the oracle): RACC serving-size category (§101.12),
   %DV calculation, nutrient-content claim thresholds (§101.13, §101.54–60),
   authorized health-claim list, top-9 allergen declarations (FALCPA +
   sesame), "healthy" criteria (2026), FOP rule (2026), formatting/units.
5. Emit verdict JSON: `PASS | FLAG`, list of `{violation_type, severity,
   cfr_citation, observed, expected, correction}`. Human signs off.

## Benchmark contracts

See `CONTRACTS.md` — fixed taxonomy (violation classes + severity weights),
scoring functions, contamination scheme, benchmark rules. Nothing in this repo
may change the contracts without a DECISIONS.md entry and measured evidence.

## Directory layout

```
forge/      seeded label generator + CFR verifier-as-oracle + contamination
            monitor + golden benchmark (uv project, package plumb_forge)
model/      multimodal dataset builder, QLoRA SFT, GRPO RLVR, ReST-EM,
            benchmark eval, frontier head-to-head, HF model card
            (package plumb_model)
cloud/      Modal app + Dockerfile + checkpoint sync + GCP spot scripts
eval/       deterministic golden harness (no LLM judges for core metrics)
docs/       corpus (CFR reference), DECISIONS.md, BENCHMARK.md, HANDOFF.md,
            methodology.md, study results
```

## Training stack (SOTA 2026)

1. **SFT cold-start** — QLoRA (4-bit NF4) on verifier-filtered traces
   distilled from a stronger model (Qwen3.8-2.4T-A95B / DeepSeek). RLVR data
   is never mixed into SFT.
2. **RLVR** — GRPO with the Dr. GRPO fix (drop per-token length norm),
   DAPO-style decoupled clip (high 1.0 / low 0.2) + dynamic sampling of
   non-saturated prompts, G≈8–16, outcome-verifier-as-oracle only.
3. **Self-play** — ReST-EM / expert iteration (sample → verifier-filter →
   SFT) plus one s1-style curation pass (~1k high-difficulty traces).
4. **Judges** — pairwise, position-swapped, temp 0, majority of 3, bootstrap
   CIs; only for residual prose, never for core verdicts.

## Targets (golden benchmark, held-out seed, zero contamination)

| Metric | Target |
|---|---|
| Critical-violation recall (undeclared allergen, false claim, wrong serving size) | **> 98%** |
| Overall violation precision | **> 95%** |
| Citation exact-match accuracy | **> 95%** |
| Parse rate (schema-valid JSON) | 100% |
| Severity-weighted recall (weights in CONTRACTS.md) | **> 0.95** |

Head-to-head vs **Qwen3.8-2.4T-A95B**, **DeepSeek v4-flash**, base
**Qwen3.8-27B** zero-shot — same inputs, same scoring.

## Cloud (free credits)

- **Modal (primary):** `your-modal-profile` profile active, $30/mo free credit,
  L4 24GB ≈ 37 GPU-hrs/mo. SFT (~15–20 GPU-hr) + RLVR (~20–40 GPU-hr) fits in
  ~2 monthly credit cycles. Checkpoints on a Modal Volume (free 1 TiB).
- **GCP (fallback):** `your-gcp-project` (you@example.com,
  Compute API enabled, GPU quota limit 1000). Spot L4/A100 marathon under
  `cloud/gcp_spot.sh`.
- **Verify before training:** smoke LoRA on the base model to confirm
  Qwen3.8-27B DeltaNet fine-tuning tooling (model released Aug 2026).
  Fallbacks: TRL+PEFT, or Qwen3.8-27B-FP8 base.

## Phases

- **P0** scaffold + CONTRACTS (this repo state)
- **P1** CFR oracle engine (`forge` verifier) + hand-verified CFR test cases
- **P2** label generator (planted violations from real openFDA patterns) +
      golden benchmark
- **P3** multimodal dataset builder + QLoRA SFT on Modal
- **P4** GRPO RLVR against the oracle
- **P5** self-play + judge calibration
- **P6** full benchmark head-to-head + model card + writeup

## Run

```sh
cd forge && uv sync && uv run pytest -q      # oracle + generator green
make sync && make validate                   # exit gate
```

## Risks (honest)

- **Tooling maturity** for the 3-day-old DeltaNet architecture — smoke LoRA
  first (P1 gate).
- **Label-rendering fidelity** — synthetic labels must look like real scanned
  packs; augment with OCR noise, rotation, glare, perspective.
- **Legal** — review-assist only, human sign-off; the benchmark demonstrates
  capability, not a compliance guarantee.