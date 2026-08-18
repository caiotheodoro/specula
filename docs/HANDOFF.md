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

## Status (2026-08-18)

- **P0 scaffold**: complete — CONTRACTS, README, all packages, cloud, docs.
- **P1 oracle**: live in `forge/src/specula_forge/verify.py` (printed-vs-true
  design). RACC table verified against eCFR 21 CFR 101.12(b); 2026 "healthy"
  rule verified against 101.65(d)(3)(iii) Table 4. DV_REF confirmed against
  21 CFR 101.9 (c)(8)(iv) RDIs + (c)(9) DRVs (see DECISIONS.md). Claim slice
  includes free / very low sodium; 101.72 and 101.74 have condition checks.
- **P2 generator**: live (`generate.py`) — all 14 taxonomy classes in `ALL`,
  oracle gate enforced, health-claim / FORMATTING / LEGIBILITY now visible
  on the PNG. `difficulty` drives near-threshold claims, near-RACC servings,
  and OCR/rotation/glare. `--mix openfda` uses a static class-weight fixture.
- **P3**: dataset builder has a CLI; eval uses SYSTEM_PROMPT + real pixels + OpenAI-compat provider +
  resumable JSONL + oracle-mock; citation exact-match is scored. Modal smoke
  path ran on Modal L4: 4 optimizer steps, then full SFT on seed-7 train
  (204 rows, 2 epochs, 102 steps, train_loss 0.443 text-only). Then VL
  SFT on rendered label PNGs (shard-0000, 102 steps, train_loss 2.366,
  40 min GPU, ap-asAxCdeOO7cB3q7LBc2LQO). Checkpoint `/checkpoints/sft-final`.
- **P4 GRPO**: forge `reward()` is the oracle scalar (recall − FP penalty +
  verdict bonus). `specula_model.rlvr_reward.oracle_reward_func` is the TRL
  hook. Prompts from `dataset_builder --rlvr-out` have no assistant gold;
  SFT JSONL is rejected. `modal_rlvr.py` runs 4-bit Dr. GRPO + DAPO clip
  (`epsilon=0.2`, `epsilon_high=1.0`) from `/checkpoints/sft-final`.
  Smoke ran on Modal L4: 2/2 steps, train_runtime 50s, checkpoint
  `/checkpoints/rlvr-final` (ap-pES3JzaVBU18Qyp3SA0VDx). Dummy 1x1 PNG +
  clipped 64-token completions all scored reward −1 (unparseable); that
  proves the loop, not learning. Marathon: `cloud/gcp_spot.sh` with real
  `--rlvr-out` prompts. Do not mix RLVR into SFT.
- **Verified green**: `make validate` (forge + model). Seed-7 pilot
  400 (295 FLAG) → train 204 / val 196 overlap 0; seed-777 benchmark 1000
  (734 FLAG); leakprobe clean vs train (0 false-fire, 10/10 on planted leaks).

## Next actions

1. **P4 marathon**: build `dataset_builder --rlvr-out` from train.jsonl,
   then `SMOKE=0 GROUP_SIZE=4 ITERS=200 ./cloud/gcp_spot.sh` after copying
   `sft-final` onto the VM. L4 G=8 VL may OOM; keep G=2–4 on 24GB.
2. **P6 head-to-head**: set `SPECULA_LLM_BASE_URL` and fill
   `docs/BENCHMARK.md`. Serve `/checkpoints/sft-final` (now VL).

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
