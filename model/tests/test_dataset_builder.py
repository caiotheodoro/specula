"""Dataset builder: forge Task JSONL -> multimodal shard."""

import base64
import json
import random
from pathlib import Path

from specula_forge.generate import task
from specula_model.dataset_builder import build, build_rlvr_prompt, build_rlvr_prompts
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



def test_build_rlvr_prompt_has_no_assistant_turn():
    t = task(random.Random(9), "cookies", seed=9, n_violations=1)
    prompt = build_rlvr_prompt(t)
    roles = [turn["role"] for turn in prompt["prompt"]]
    assert roles == ["system", "user"]
    assert "assistant_json" not in prompt
    assert "expected_verdict" in prompt
    json.loads(prompt["expected_verdict"])
    assert prompt["images"]
    user = prompt["prompt"][1]["content"]
    assert any(part.get("type") == "image" for part in user)


def test_build_rlvr_prompts_writes_jsonl_distinct_from_sft(tmp_path: Path):
    t = task(random.Random(10), "cookies", seed=10, n_violations=1)
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(t.model_dump_json() + "\n")
    out = tmp_path / "rlvr-prompts.jsonl"
    n = build_rlvr_prompts(str(tasks_file), str(out))
    assert n == 1
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["task_id"] == t.task_id
    assert [turn["role"] for turn in rec["prompt"]] == ["system", "user"]
    assert "assistant_json" not in rec
