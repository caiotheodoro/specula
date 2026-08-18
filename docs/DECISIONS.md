# Specula — Decision Log

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
## 2026-08-17 — P1 — RACC table verified against eCFR 21 CFR 101.12(b)
- Decision: RACC entries re-verified live against eCFR Table 2 (General Food
  Supply). Corrections: snack_chips 28→30 g, yogurt 225→170 g, candy 40→30 g,
  pizza 145→140 g, cereal_ready_to_eat 30→40 g, beverages re-anchored to
  juices/nectars (8 fl oz / 240 mL). Frozen dessert gram weight is a working
  estimate (2/3 cup per eCFR; no gram value published).
- Rationale: mis-specified reference amounts would silently poison the oracle
  and every SERVING_SIZE label; public law is the ground truth.
- Evidence: eCFR 101.12(b) fetched 2026-08-17; forge tests + pilot green.
- Alternatives rejected: keeping scaffold estimates.

## 2026-08-17 — P1 — 2026 "healthy" criteria verified (21 CFR 101.65(d))
- Decision: healthy-rule thresholds updated to the Dec-27-2024 final rule as
  codified: mixed-product path (Table 4) = added sugars ≤10% DV, sodium ≤15%
  DV, saturated fat ≤10% DV. FOP_RULE stays coupled to healthy criteria;
  exact FOP-symbol rule placement pending a follow-up pass.
- Rationale: the scaffold used pre-final-rule thresholds; the codified final
  rule is authoritative and fetched live from eCFR.
- Evidence: eCFR 101.65(d)(3)(iii) fetched 2026-08-17; tests + pilot green.

## 2026-08-17 — P1 — DV reference values (status: standard values, table pending)
- Decision: DV_REF uses the 2020-update reference daily values (78 g fat,
  300 mg cholesterol, 2300 mg sodium, 275 g carb, 28 g fiber, 50 g added
  sugars, 50 g protein, 20 mcg vit D, 1300 mg calcium, 18 mg iron,
  4700 mg potassium, 2000 kcal). The 101.9 fetch truncated before
  (c)(8)(iv); a full-table re-check is queued but these match the published
  2020 table.
- Rationale: correctness of %DV depends on these; flagged honestly as
  standard-values pending the full-table confirmation.
- Evidence: verify.py DV_REF; test_dv_math_anchor.

## 2026-08-17 — P1 — DV_REF confirmed against 21 CFR 101.9 (c)(8)(iv) + (c)(9)
- Decision: keep current DV_REF numbers; they match the adult ≥4 tables. Note that HANDOFF called this a 101.9(c)(8)(iv) re-check, but macronutrient DVs live in (c)(9) DRVs; (c)(8)(iv) is the vitamin/mineral RDI table. `total_sugars` has no statutory DV and remains a non-statutory 50 g alias of added sugars.
- Rationale: wrong DVs would silently poison DV_ERROR, source-claim %DV, and healthy/FOP limits while the oracle gate still passed.
- Evidence: Cornell LII eCFR text of 21 CFR 101.9 fetched 2026-08-17 (eCFR.gov HTML was CAPTCHA-blocked); forge test_dv_ref_matches_101_9_tables.
- Alternatives rejected: adding optional (c)(8)(iv) vitamins to NutritionFacts; deleting total_sugars from DV_REF in this pass.

## 2026-08-17 — P3 — Modal smoke LoRA on Qwen3.8-27B (L4)
- Decision: keep Qwen/Qwen3.8-27B 4-bit QLoRA on Modal L4. Smoke uses
  `loss_type=nll`, `max_length=256`, LoRA r=8, torchvision + python3-dev
  in the image, and a `specula-hf-cache` volume. Full runs stay 4-bit
  with `max_length=1024` and LoRA r=32.
- Rationale: first GPU attempts failed on (1) Modal needing `python` on
  PATH, (2) Qwen3 VL AutoProcessor needing torchvision, (3) Triton JIT
  needing Python.h, (4) TRL default `chunked_nll` upcasting the 248k
  lm_head to fp32 (~4.7 GiB) which is incompatible with PEFT+VLM on 24GB.
  Standard `nll` at 256 tokens completed 4/4 steps.
- Evidence: `modal run cloud/modal_train.py --smoke` exit 0,
  ap-i2MSEyzsNuWHLUX35f9srK, train_loss 3.279, 4 steps in 60s GPU.
- Alternatives rejected: Qwen3.8-27B-FP8 (not needed; 4-bit loaded);
  keeping chunked_nll (OOM); max_length 4096 on L4 (CE/activation wall).

## 2026-08-17 — P3 — Full SFT on Modal; GCP is the marathon wallet
- Decision: run this SFT on Modal L4, not GCP. Keep GCP $300 credits
  for P4+ GRPO/self-play. Do not use `gcp_spot.sh` as-is.
- Rationale: smoke already proved 4-bit Qwen3.8-27B on Modal; HF cache
  and the train image were warm. 204×2 epochs is ~26 min GPU (~$0.35
  at $0.80/hr L4), well inside the $30 credit. GCP spot L4 is cheaper
  per hour (~$0.40–$0.56) but `gcp_spot.sh` clones GitHub, installs
  unused flash-attn/unsloth/vllm, and calls a missing
  `specula_model.train` module. First GCP boot would also re-download
  28B weights and can be preempted. $300 GCP is the right budget for
  15–40 GPU-hr RLVR, not for a sub-hour SFT.
- Evidence: `modal run cloud/modal_train.py --epochs 2 --data forge/data/train.jsonl`
  exit 0, ap-whNEgDfAuVtv68Dss3aSkC, 102/102 steps, train_loss 0.443,
  train_runtime 1569s. Loss 4.1 → 0.02 on text-only JSON is expected
  memorization on n=204, not a VL result.
- Alternatives rejected: jumping to GCP for this run; on-demand G2
  (~$1.00/hr, more expensive than Modal for a short job).

## 2026-08-17 — P3 — VL SFT with label PNGs
- Decision: SFT chats carry a thumbnail (max 384px) of `render_png` plus
  the verdict JSON. Dummy/smoke records use a 1x1 PNG so the VL collator
  is always exercised. Assistant content is typed `[{type:text,...}]` so
  Arrow does not mix list and string message parts.
- Rationale: the text-only SFT never saw a label; that adapter cannot be
  a VL reviewer. Loss 10.86 → 2.366 on n=204 with images is the real
  cold start; the earlier 0.443 text-only number was JSON memorization.
- Evidence: VL smoke ap-bGq5nkkTiVPWx8uT9ZWfho exit 0; full VL SFT
  ap-asAxCdeOO7cB3q7LBc2LQO, 102/102, train_runtime 2419s.
- Alternatives rejected: keeping text-only SFT as champion; max_length=None
  (VRAM risk on L4); sending full 600x900 PNGs without thumbnail.

## 2026-08-18 — P4 — GRPO RLVR against the forge oracle
- Decision: reward = severity-weighted recall − 0.3·FP + 0.2 verdict bonus
  (unparseable = −1.0; clean PASS scored on verdict only). Train with TRL
  `GRPOConfig(loss_type="dr_grpo", epsilon=0.2, epsilon_high=1.0)`, 4-bit
  QLoRA on the VL SFT adapter. Modal smoke G=2 / 2 steps; GCP spot L4 is
  the marathon (`gcp_spot.sh` rsyncs this checkout and runs
  `python -m specula_model.rlvr_train`). RLVR prompts are structurally
  distinct from SFT traces (no assistant gold); SFT JSONL is rejected.
- Rationale: methodology §3 (Dr. GRPO, DAPO clip, outcome-verifier only).
  Citation exact-match stays a report metric, not the RLVR reward, so the
  policy cannot hack CFR strings. Dynamic zero-variance resampling is not
  native in GRPOTrainer — monitor `frac_reward_zero_std` rather than
  subclassing preemptively. Unquantized 28B bf16 (habeas stub) cannot fit
  L4 24GB next to G completions; keep 4-bit. G=8 on L4 is likely an OOM;
  smoke G=2, marathon G=4 until an A100 is justified.
- Evidence: forge `test_score.py` reward cases; model `test_rlvr_reward.py`,
  `test_train_config.py` GRPO kwargs + SFT-mix rejection; `make validate`.
  Modal smoke `modal run cloud/modal_rlvr.py --smoke` exit 0,
  ap-pES3JzaVBU18Qyp3SA0VDx, 2/2 steps, train_runtime 50.11s, reward −1
  on dummy 1×1 PNG (all completions clipped at 64, frac_reward_zero_std=1).
  First attempt died on Modal TRL 1.10 dropping `max_prompt_length`;
  `grpo_trainer_kwargs` now filters unknown fields.
- Alternatives rejected: mixing train.jsonl SFT traces into GRPO; loading
  full bf16 28B on L4; putting citation exact-match in the reward; cloning
  GitHub from `gcp_spot.sh` (this branch is unpushed; the old script also
  called a missing `specula_model.train` and installed flash-attn/vllm).

## 2026-08-18 — P4 — GCP GPU quota is zero; marathon stays on Modal
- Decision: do not wait on GCP for this GRPO marathon. Run 4-bit VL GRPO
  on Modal L4 with G=2, 256px thumbs, max_completion 128, and
  `chat_template_kwargs={"enable_thinking": False}`.
- Rationale: `gcloud compute instances create` on
  `your-gcp-project` fails with Quota `GPUS_ALL_REGIONS` exceeded
  (limit 0 globally) even though regional `NVIDIA_L4_GPUS=1`. A first
  real-prompt run without thinking-off filled 256 tokens, scored reward
  −1, and left ~5MB free VRAM. Disabling thinking made completions
  terminate as JSON (sample `{"verdict":"PASS","violations":[]}`).
- Evidence: quota error from account `you@example.com`;
  8-step probe ap-gsQ5DepGTEmt48N6vzN7Go exit 0, train_runtime 446.7s,
  train_loss −0.01694, reward std up to 0.93.
- Alternatives rejected: burning 200 steps at reward −1; G=4 on L4
  (OOM wall); filing a quota ticket in this loop (needs a human in the
  GCP console).
