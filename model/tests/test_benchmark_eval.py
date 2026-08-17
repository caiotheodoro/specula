"""Benchmark eval: oracle-mock, pixels, resume, CLI flags."""

import json
import random
import sys
from pathlib import Path

from specula_forge.generate import task
from specula_model.benchmark_eval import main, run_benchmark
from specula_model.schema import SYSTEM_PROMPT


def _write_tasks(path: Path, *tasks) -> None:
    path.write_text("".join(t.model_dump_json() + "\n" for t in tasks))


def test_oracle_mock_parsed_rate(tmp_path: Path):
    t = task(random.Random(7), "cookies", seed=7, n_violations=1)
    tasks_file = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_file, t)
    out = tmp_path / "results.jsonl"

    run_benchmark(tasks_file, "oracle-mock", 1, out)

    row = json.loads(out.read_text().splitlines()[0])
    assert row["score"]["parsed"] == 1.0


def test_resume_does_not_duplicate_task_id(tmp_path: Path):
    rng = random.Random(7)
    t1 = task(rng, "cookies", seed=7, n_violations=1)
    t2 = task(rng, "bread", seed=8, n_violations=1)
    tasks_file = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_file, t1, t2)
    out = tmp_path / "results.jsonl"
    out.write_text(json.dumps({"task_id": t1.task_id, "marker": True}) + "\n")

    run_benchmark(tasks_file, "oracle-mock", 1, out)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    ids = [row["task_id"] for row in rows]
    assert ids.count(t1.task_id) == 1
    assert t2.task_id in ids
    kept = next(row for row in rows if row["task_id"] == t1.task_id)
    assert kept.get("marker") is True


def test_cli_accepts_adapter_path(tmp_path: Path, monkeypatch):
    t = task(random.Random(7), "cookies", seed=7, n_violations=1)
    tasks_file = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_file, t)
    out = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_eval",
            "--tasks-file",
            str(tasks_file),
            "--out",
            str(out),
            "--model",
            "oracle-mock",
            "--adapter-path",
            "adapters/champion",
        ],
    )
    main()
    assert out.exists()


def test_eval_payload_uses_system_prompt_and_png():
    from specula_model.benchmark_eval import _eval_payload

    t = task(random.Random(7), "cookies", seed=7, n_violations=1)
    payload = _eval_payload(t)
    assert payload["system"] == SYSTEM_PROMPT
    assert payload["image"].startswith(b"\x89PNG")
