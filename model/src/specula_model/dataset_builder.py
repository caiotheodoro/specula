"""Multimodal dataset builder: forge Task JSONL -> Qwen3.8 VL chat dataset.

One record per task: {"image": <png bytes>, "messages": [{role, content}],
"assistant": <verdict json>}. Images are stored alongside (arrow shards in
P3). Runs on CPU; images come from forge's render_png.

specula_forge must be on PYTHONPATH (pytest sets ../forge/src via
pyproject; otherwise PYTHONPATH=src:../forge/src).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import base64
from PIL import Image
from pydantic import BaseModel

from .schema import SYSTEM_PROMPT
from .train_config import USER_PROMPT


class Example(BaseModel):
    image_b64: str
    user_text: str
    assistant_json: str


def _b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()


def build(tasks_jsonl: Path, out_dir: Path, rng_seed: int = 7) -> None:
    """Convert forge task JSONL into a sharded multimodal dataset."""
    from specula_forge.generate import render_png
    from specula_forge.schema import Task

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(rng_seed)
    lines = [l for l in tasks_jsonl.read_text().splitlines() if l.strip()]
    rng.shuffle(lines)
    shard_size = 256
    for shard_i, start in enumerate(range(0, len(lines), shard_size)):
        shard = []
        for line in lines[start:start + shard_size]:
            task = Task.model_validate_json(line)
            img = render_png(task.label)
            ex = Example(
                image_b64=_b64(img),
                user_text="Review this food label for FDA compliance. "
                          "Emit the verdict JSON.",
                assistant_json=task.expected.model_dump_json(),
            )
            shard.append(ex.model_dump())
        out = out_dir / f"shard-{shard_i:04d}.jsonl"
        with open(out, "w") as f:
            for ex in shard:
                f.write(json.dumps(ex) + "\n")
        print(f"wrote {out} ({len(shard)} examples)")


def build_rlvr_prompt(task) -> dict:
    """Prompt-only record for GRPO. No assistant gold — not an SFT trace."""
    from specula_forge.generate import render_png
    img = render_png(task.label)
    return {
        "task_id": task.task_id,
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ],
        "images": [_b64(img)],
        "expected_verdict": task.expected.model_dump_json(),
    }


def build_rlvr_prompts(tasks_path: str, out_path: str) -> int:
    from specula_forge.schema import Task
    src = Path(tasks_path)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dest, "w") as f:
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            f.write(json.dumps(build_rlvr_prompt(Task.model_validate_json(line))) + "\n")
            n += 1
    return n


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--rlvr-out", default="",
                    help="Write prompt-only RLVR JSONL instead of SFT shards")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.rlvr_out:
        n = build_rlvr_prompts(args.tasks_file, args.rlvr_out)
        print(f"wrote {n} RLVR prompts to {args.rlvr_out}")
        return
    if not args.out_dir:
        ap.error("--out-dir is required unless --rlvr-out is set")
    build(Path(args.tasks_file), Path(args.out_dir), rng_seed=args.seed)


if __name__ == "__main__":
    main()