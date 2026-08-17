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
  rule verified against 101.65(d)(3)(iii) Table 4. DV_REF confirmed against
  21 CFR 101.9 (c)(8)(iv) RDIs + (c)(9) DRVs (see DECISIONS.md). Claim and
  health-claim *condition* expansion still open.
- **P2 generator**: live (`generate.py`) — all 14 taxonomy classes in `ALL`,
  oracle gate enforced, health-claim / FORMATTING / LEGIBILITY now visible
  on the PNG. `difficulty` is still unused. OCR/glare/openFDA not started.
- **P3**: dataset builder has a CLI; eval uses SYSTEM_PROMPT + real pixels +
  resumable JSONL + oracle-mock; citation exact-match is scored. Modal smoke
  path is 4-bit + dummy records + `max_steps=4` — **not yet run on GPU**.
- **Verified green**: `make validate` (25 forge + 13 model). Existing 400-task
  pilot JSONL is stale vs the new ALL/render; regenerate before a P3 freeze.

## Next actions

1. **P1 finish**: expand claim and health-claim condition checks; hand-built
   CFR fixtures per class; tighten allergen tokens if needed.
2. **P2 finish**: drive `difficulty`; near-threshold injectors; OCR-noise /
   rotation / glare; static openFDA mix; then regenerate seed-7 / 777.
3. **P3 smoke**: `modal run cloud/modal_train.py --smoke` (≪1 GPU-hr). This
   still gates all four forge repos. Do not start full SFT until it steps.
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
