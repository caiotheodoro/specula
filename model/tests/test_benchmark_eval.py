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


def test_complete_chat_posts_openai_compat_and_returns_content(monkeypatch):
    from specula_model.benchmark_eval import complete_chat

    captured = {}

    def fake_post(url, headers, body):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return {"choices": [{"message": {"content": '{"verdict":"PASS","violations":[]}'}}]}

    monkeypatch.setenv("SPECULA_LLM_BASE_URL", "http://llm.local/v1")
    monkeypatch.setenv("SPECULA_LLM_API_KEY", "sk-test")
    out = complete_chat("qwen-local", {"system": "sys", "image": b"\x89PNG"}, post=fake_post)
    assert out == '{"verdict":"PASS","violations":[]}'
    assert captured["url"] == "http://llm.local/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "qwen-local"
    assert captured["body"]["messages"][0]["role"] == "system"


def test_predict_one_uses_complete_chat_for_non_mock(monkeypatch):
    from specula_model import benchmark_eval as be

    t = task(random.Random(7), "cookies", seed=7, n_violations=1)
    monkeypatch.setattr(be, "complete_chat", lambda model, payload, post=None: '{"verdict":"FLAG","violations":[]}')
    assert be._predict_one(t, "deepseek-v4-flash") == '{"verdict":"FLAG","violations":[]}'


def test_complete_chat_accepts_full_chat_completions_url(monkeypatch):
    from specula_model.benchmark_eval import complete_chat

    captured = {}

    def fake_post(url, headers, body):
        captured["url"] = url
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setenv("SPECULA_LLM_BASE_URL", "https://modal.example/chat/completions")
    complete_chat("m", {"system": "s", "image": b"x"}, post=fake_post)
    assert captured["url"] == "https://modal.example/chat/completions"
