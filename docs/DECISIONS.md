# Labelforge — Decision Log

Every architectural decision is recorded here with rationale and evidence.
Agents append entries as they make decisions (format below). The P1 oracle
rule set, severity weights, and benchmark seeds are the fixed defaults from
CONTRACTS.md; revise only with measured evidence.

## Format

```
## YYYY-MM-DD — <study id> — <title>
- Decision: ...
- Rationale: ...
- Evidence: (file/metrics)
- Alternatives rejected: ...
```

---

## 2026-08-17 — P0 — Base model: Qwen/Qwen3.8-27B (multimodal 28B)
- Decision: Qwen3.8-27B (Apache-2.0, 262k context, VL) as the fine-tune base,
  QLoRA 4-bit, on Modal L4 then GCP spot fallback.
- Rationale: released 2026-08 (3 days before P0), the model the user named;
  vision-native input is the decisive weapon for label review; dense 28B fits
  L4 24GB in 4-bit; Apache-2.0 for a public portfolio repo.
- Evidence: HF model card (V1 architecture, OmniDocBench 91.1).
- Alternatives rejected: Qwen3.6-27B (mature but text-only), Qwen3.8-27B-FP8
  (fallback if tooling lacks DeltaNet support).

## 2026-08-17 — P0 — Cloud: Modal primary, GCP fallback
- Decision: Modal profile `your-modal-profile` ($30/mo free credit, L4 24GB,
  ~37 GPU-hrs/mo) is the training platform; GCP `your-gcp-project`
  (you@example.com, Compute API enabled, GPU quota 1000) is the
  $300-credit marathon fallback.
- Rationale: free tier elsewhere cannot fit a 28B QLoRA (Colab/Kaggle T4
  = 16GB wall); Modal's recurring credit + free 1 TiB volume makes 2-cycle
  SFT+RLVR feasible without money.
- Evidence: `modal token set` verified 2026-08-17; gcloud services enable
  compute.googleapis.com finished; GPU quota limit 1000.

## 2026-08-17 — P0 — Oracle design: printed vs true split
- Decision: Label carries both `printed` (as-rendered fields) and `true`
  (ProductFacts). The oracle compares printed vs true/regulatory; the
  generator derives printed from true + injected violations and runs the
  self-check gate.
- Rationale: a verifier cannot catch a wrong serving size unless it knows the
  true one; the split makes every injected violation independently detectable
  and keeps 100% oracle agreement structural.
- Evidence: forge tests (oracle gate holds on pilot sweep).
- Alternatives rejected: single "ground truth label" object (cannot detect
  deviations from truth).

## 2026-08-17 — P0 — RACC/DV reference tables: placeholders pending eCFR
- Decision: RACC subset (21 CFR 101.12(b)) and DV reference values
  (21 CFR 101.9(c)(8)(iv)) are embedded as constants; each entry must be
  re-verified against eCFR during P1 before dataset generation.
- Rationale: scaffold needs working tables now; exact values are public law
  and cheap to confirm; mis-specified RACC entries would silently poison the
  oracle.
- Evidence: verify.py constants; test_dv_math_anchor.