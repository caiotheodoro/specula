"""Dataset builder: forge Task JSONL -> multimodal shard."""

import base64
import json
import random
from pathlib import Path

from specula_forge.generate import task
from specula_model.dataset_builder import build
from specula_model.schema import parse


def test_build_writes_png_shard(tmp_path: Path):
    t = task(random.Random(7), "cookies", seed=7, n_violations=1)
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(t.model_dump_json() + "\n")
    out_dir = tmp_path / "out"

    build(tasks_file, out_dir, rng_seed=7)

    shard = out_dir / "shard-0000.jsonl"
    assert shard.exists()
    rec = json.loads(shard.read_text().splitlines()[0])
    png = base64.b64decode(rec["image_b64"])
    assert png.startswith(b"\x89PNG")
    assert parse(rec["assistant_json"]) is not None
