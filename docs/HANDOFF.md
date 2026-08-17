# Plumb — Handoff

## Status (2026-08-17)

- **P0 scaffold**: complete. CONTRACTS, README, forge (schema, oracle,
  generator, score, contamination, CLI), model (schema, dataset builder,
  benchmark eval), cloud (Modal app, Dockerfile, GCP script), docs.
- **P1 (oracle engine)**: first cut live in `forge/src/plumb_forge/verify.py`
  with the printed-vs-true design. **Pending**: re-verify RACC + DV reference
  tables against eCFR; add FORMATTING and LEGIBILITY rules; expand claim and
  health-claim condition checks.
- **P2 (generator)**: first cut live (`generate.py`) — renders composite PNGs,
  injects 11 violation classes, oracle gate enforced. **Pending**: OCR-noise /
  rotation / glare augmentation; real openFDA violation-pattern seeding;
  adversarial near-threshold difficulty parameterization.
- **P3+**: not started (dataset builder stub, benchmark eval stub, Modal
  train/RLVR stubs).

## Next actions

1. eCFR verification pass on `RACC` and `DV_REF` (and the 2026 healthy/FOP
   thresholds) — gate for dataset generation.
2. Build the train/val/benchmark sets (P2) and run the smoke LoRA on Modal to
   validate Qwen3.8-27B DeltaNet fine-tuning tooling.
3. Wire a real provider adapter into `benchmark_eval._predict_one` (local
   vLLM/MLX or frontier API).

## Environment

- Modal: profile `your-modal-profile` (token verified 2026-08-17).
- gcloud: active account `you@example.com`, project
  `your-gcp-project`, Compute API enabled, GPU quota 1000.
- uv + Python ≥3.11; `make sync` then `make validate` in repo root.