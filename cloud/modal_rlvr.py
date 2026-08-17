"""RLVR trainer (GRPO/Dr.GRPO, DAPO-style) on Modal — P4.

Uses the forge oracle as the reward function: an answer's verdict JSON is
parsed, scored against the task's oracle Verdict per CONTRACTS §2, and the
reward is severity-weighted recall minus false-positive penalty. Outcome
verifier only — no PRM (CONTRACTS.md / methodology).

Run:
  modal run cloud/modal_rlvr.py --prompts data/rlvr-prompts.jsonl --iters 200
"""

from __future__ import annotations

import modal

app = modal.App("labelforge-rlvr")
vol = modal.Volume.from_name("labelforge-checkpoints", create_if_missing=True)
image = modal.Image.from_dockerfile("Dockerfile")
GPU_L4 = modal.gpu.L4(count=1)


@app.function(image=image, gpu=GPU_L4, volumes={"/checkpoints": vol},
              timeout=60 * 60 * 12)
def rlvr(prompts: bytes, base_adapter: str, iters: int = 200, group_size: int = 8):
    # TODO(P4): GRPO with Dr. GRPO fix (no per-token length norm), DAPO
    # decoupled clip (high=1.0, low=0.2), dynamic sampling of non-saturated
    # prompts; reward = forge oracle severity-weighted recall - FP penalty.
    # Rollouts G=8 per prompt, verifier-filter, advantage update.
    raise NotImplementedError("P4: GRPO RLVR against the forge oracle")
    return "done"