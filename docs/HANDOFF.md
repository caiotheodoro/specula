# Specula — Handoff (fresh-agent bootstrap)

**What this is:** a fine-tune of Qwen3.8-27B (28B multimodal, Apache-2.0) that
reviews FDA food labels → cited compliance report. Specs: `README.md`
(atomic spec), `CONTRACTS.md` (fixed benchmark contracts), `docs/methodology.md`
(shared recipe across the four repos: specula, suture, plumb, habeas).

## Repo map

```
forge/    specula_forge: seeded label generator + CFR verifier-as-oracle +
          contamination monitor + CLI (pilot/split/leakprobe)
model/    specula_model: dataset builder (stub), benchmark eval (stub),
          schema.py = model-output contract + parser + SYSTEM_PROMPT
cloud/    Modal app (L4 24GB, QLoRA) + Dockerfile + GCP spot script
eval/     deterministic golden harness notes
docs/     DECISIONS.md (decision log), BENCHMARK.md (report template),
          HANDOFF.md, methodology.md
```

## Status (2026-08-17)

- **P0 scaffold**: complete — CONTRACTS, README, all packages, cloud, docs.
- **P1 oracle**: live in `forge/src/specula_forge/verify.py` (printed-vs-true
  design). RACC table verified against eCFR 21 CFR 101.12(b); 2026 "healthy"
  rule verified against 101.65(d)(3)(iii) Table 4 (see DECISIONS.md). DV
  reference values are the standard 2020 values; full-table re-check of
  101.9(c)(8)(iv) still queued.
- **P2 generator**: live (`generate.py`) — renders composite PNGs, injects 11
  violation classes, oracle gate enforced, deterministic per seed.
- **P3+**: not started — dataset builder + benchmark eval + Modal train/RLVR
  are stubs.
- **Verified green**: `make validate` (14 tests); 400-task pilot (305 FLAG),
  stratified split overlap 0, contamination leak-probe ROC clean.

## Next actions

1. **P1 finish**: confirm DV_REF against full 101.9(c)(8)(iv) table; add
   FORMATTING/LEGIBILITY extras (currently flag-based); expand claim and
   health-claim condition checks.
2. **P2 finish**: OCR-noise / rotation / glare augmentation; seed from real
   openFDA violation patterns; adversarial near-threshold difficulty.
3. **P3**: build train/val/benchmark (seed 7 / 777) via the CLI; then run the
   **smoke LoRA on Modal** to validate Qwen3.8-27B DeltaNet fine-tuning
   tooling (the gating risk — model is days old).
4. Wire a real provider adapter into `benchmark_eval._predict_one` (local
   vLLM/MLX or frontier API) for head-to-head.

## Bootstrap (fresh agent)

```sh
cd ~/Documents/personal/specula
make sync          # uv sync --no-editable --extra dev (forge)
make validate      # forge pytest
cd forge && uv run python -m specula_forge.cli pilot --seed 7 --n 400 --out data/pilot.jsonl
cd forge && uv run python -m specula_forge.cli split --pilot data/pilot.jsonl --out-train data/train.jsonl --out-val data/val.jsonl
cd forge && uv run python -m specula_forge.cli leakprobe --train data/train.jsonl --eval-file data/val.jsonl
```

## Environment (already configured)

- **Modal**: profile `your-modal-profile` (token verified 2026-08-17, $30/mo free
  credit, L4 24GB ≈ 37 GPU-hrs/mo). Volume `specula-checkpoints`.
- **gcloud**: active account `you@example.com`, project
  `your-gcp-project`, Compute API enabled, GPU quota 1000 (spot L4
  fallback).
- uv + Python ≥3.11; macOS note: `make sync` runs `chflags -R nohidden .venv`
  (uv's UF_HIDDEN flag otherwise breaks editable .pth — see DECISIONS F7/I1).

## Parallel repos

suture (policy-issuance QC), plumb (AIA pay-app review), habeas (I-9
validation) — same scaffold/methodology, independently operable. Specula is
the lead repo; its P3 smoke-LoRA outcome (DeltaNet tooling) gates all four.
