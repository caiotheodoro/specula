"""Modal P6 eval: 4-bit Qwen3.8-27B + PEFT adapter, oracle-scored.

Batch (the measurement path):

  modal run cloud/modal_eval.py --smoke --tasks forge/data/val.jsonl
  modal run cloud/modal_eval.py --tasks forge/data/val.jsonl \
      --adapter /checkpoints/rlvr-final --name rlvr-val
  modal run cloud/modal_eval.py --tasks forge/data/benchmark.jsonl \
      --adapter /checkpoints/rlvr-final --name rlvr-bench

Local serve (GPU machine): `make serve` with `--adapter-path`, then
`SPECULA_LLM_BASE_URL=http://127.0.0.1:8000/v1`. Batch eval on Modal is
the P6 measurement; Modal 1.2.6 cannot serde str class parameters for a
web Cls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_SRC = _ROOT / "model" / "src"
_FORGE_SRC = _ROOT / "forge" / "src"
if str(_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(_MODEL_SRC))

app = modal.App("specula-eval")
vol = modal.Volume.from_name("specula-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("specula-hf-cache", create_if_missing=True)
_DOCKER = Path(__file__).resolve().parent / "Dockerfile"
image = (
    modal.Image.from_dockerfile(str(_DOCKER))
    .add_local_python_source("specula_model")
    .add_local_dir(str(_FORGE_SRC), remote_path="/opt/forge-src")
)

_VOLS = {"/checkpoints": vol, "/root/.cache/huggingface": hf_cache}


def _forge_path() -> None:
    import sys as _sys
    if "/opt/forge-src" not in _sys.path:
        _sys.path.insert(0, "/opt/forge-src")


@app.function(
    image=image,
    gpu="L4",
    volumes=_VOLS,
    timeout=60 * 60 * 6,
)
def eval_adapter(tasks: bytes, adapter: str = "/checkpoints/rlvr-final",
                 name: str = "eval", limit: int = 0) -> str:
    _forge_path()
    from specula_model.infer import generate_from_png, load_vl_adapter, run_eval

    tmp = Path("/tmp/tasks.jsonl")
    tmp.write_bytes(tasks)
    out = Path("/checkpoints") / "eval" / f"{name}.jsonl"
    model, processor = load_vl_adapter(adapter)

    def generate(_task, png):
        return generate_from_png(model, processor, png)

    cap = None if limit <= 0 else limit
    summary = run_eval(tmp, generate, out, limit=cap)
    vol.commit()
    return json.dumps({"adapter": adapter, "name": name, "summary": summary}, indent=2)


@app.local_entrypoint()
def main(smoke: bool = False, tasks: str = "", adapter: str = "/checkpoints/rlvr-final",
         name: str = "eval", limit: int = 0) -> None:
    if smoke:
        limit = 2 if limit <= 0 else limit
        name = name if name != "eval" else "smoke"
    blob = Path(tasks).read_bytes() if tasks else b""
    if not blob:
        raise SystemExit("pass --tasks <jsonl> (forge Task records)")
    print(eval_adapter.remote(blob, adapter=adapter, name=name, limit=limit))
