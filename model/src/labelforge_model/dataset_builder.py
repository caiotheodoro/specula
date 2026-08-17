"""Multimodal dataset builder: forge Task JSONL -> Qwen3.8 VL chat dataset.

One record per task: {"image": <png bytes>, "messages": [{role, content}],
"assistant": <verdict json>}. Images are stored alongside (arrow shards in
P3). Runs on CPU; images come from forge's render_png.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import base64
from PIL import Image
from pydantic import BaseModel


class Example(BaseModel):
    image_b64: str
    user_text: str
    assistant_json: str


def _b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()


def build(tasks_jsonl: Path, out_dir: Path, rng_seed: int = 7) -> None:
    """Convert forge task JSONL into a sharded multimodal dataset."""
    from labelforge_forge.generate import render_png
    from labelforge_forge.schema import Task

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