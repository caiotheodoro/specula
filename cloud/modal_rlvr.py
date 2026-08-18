"""Modal RLVR app: GRPO (Dr. GRPO + DAPO decoupled clip) against the forge
oracle. Short iteration on L4; GCP is the marathon wallet. See
docs/HANDOFF.md.

Run:
  modal run cloud/modal_rlvr.py --smoke
  modal run cloud/modal_rlvr.py --prompts forge/data/rlvr-prompts.jsonl --iters 200

Prompts are prompt-only JSONL from `dataset_builder --rlvr-out` — never SFT
traces (methodology: RLVR data never mixed into SFT).

Dynamic sampling of zero-variance groups is not native in GRPOTrainer
(it only logs frac_reward_zero_std). v1 monitors that metric; a custom
subclass is a follow-up if saturation shows up, not built preemptively.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_SRC = _ROOT / "model" / "src"
_FORGE_SRC = _ROOT / "forge" / "src"
if str(_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(_MODEL_SRC))

app = modal.App("specula-rlvr")
vol = modal.Volume.from_name("specula-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("specula-hf-cache", create_if_missing=True)
_DOCKER = Path(__file__).resolve().parent / "Dockerfile"
image = (
    modal.Image.from_dockerfile(str(_DOCKER))
    .add_local_python_source("specula_model")
    .add_local_dir(str(_FORGE_SRC), remote_path="/opt/forge-src")
)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/checkpoints": vol, "/root/.cache/huggingface": hf_cache},
    timeout=60 * 60 * 12,
)
def rlvr(prompts: bytes, smoke: bool = False, adapter: str = "/checkpoints/sft-final",
         iters: int = 200, group_size: int = 2) -> str:
    import sys as _sys
    if "/opt/forge-src" not in _sys.path:
        _sys.path.insert(0, "/opt/forge-src")
    from specula_model.rlvr_train import run_grpo
    out = run_grpo(prompts, smoke=smoke, adapter=adapter, iters=iters,
                   group_size=group_size, checkpoint_dir="/checkpoints")
    vol.commit()
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main(smoke: bool = False, prompts: str = "", adapter: str = "/checkpoints/sft-final",
         iters: int = 200, group_size: int = 2) -> None:
    blob = Path(prompts).read_bytes() if prompts else b""
    print(rlvr.remote(blob, smoke=smoke, adapter=adapter, iters=iters,
                      group_size=group_size))
