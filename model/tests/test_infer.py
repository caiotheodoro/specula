"""CPU tests for adapter eval: prompt shape, scoring, OpenAI-compat, resume."""

import json
import random
from pathlib import Path

from specula_forge.generate import task
from specula_model.schema import SYSTEM_PROMPT
from specula_model.train_config import USER_PROMPT


def test_eval_messages_match_rlvr_prompt():
    from specula_model.infer import eval_messages

    msgs = eval_messages()
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"][0]["text"] == SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"][0]["type"] == "image"
    assert msgs[1]["content"][1]["text"] == USER_PROMPT


def test_eval_thumb_is_256():
    from specula_model.infer import EVAL_THUMB

    assert EVAL_THUMB == (256, 256)


def test_score_raw_completion_uses_oracle():
    from specula_model.infer import score_raw_completion

    t = task(random.Random(7), "cookies", seed=7, n_violations=1)
    row = score_raw_completion(t, t.expected.model_dump_json())
    assert row["score"]["parsed"] == 1.0
    assert row["task_id"] == t.task_id
    assert row["predicted"]["verdict"] == t.expected.verdict


def test_run_eval_resumes_and_summarizes(tmp_path: Path):
    from specula_model.infer import run_eval

    rng = random.Random(7)
    t1 = task(rng, "cookies", seed=7, n_violations=1)
    t2 = task(rng, "bread", seed=8, n_violations=1)
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(t1.model_dump_json() + "\n" + t2.model_dump_json() + "\n")
    out = tmp_path / "results.jsonl"
    out.write_text(json.dumps({"task_id": t1.task_id, "marker": True}) + "\n")

    def generate(_task, _png):
        return _task.expected.model_dump_json()

    summary = run_eval(tasks_file, generate, out, limit=None)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    ids = [r["task_id"] for r in rows]
    assert ids.count(t1.task_id) == 1
    assert t2.task_id in ids
    assert summary["n_tasks"] == 1.0
    assert summary["parse_rate"] == 1.0
    assert "precision" in summary
    assert "critical_recall" in summary
    assert "high_recall" in summary


def test_run_eval_honors_limit(tmp_path: Path):
    from specula_model.infer import run_eval

    rng = random.Random(7)
    tasks = [task(rng, "cookies", seed=7 + i, n_violations=1) for i in range(3)]
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("".join(t.model_dump_json() + "\n" for t in tasks))
    out = tmp_path / "results.jsonl"
    n = {"calls": 0}

    def generate(_task, _png):
        n["calls"] += 1
        return '{"verdict":"PASS","violations":[]}'

    run_eval(tasks_file, generate, out, limit=1)
    assert n["calls"] == 1
    assert len(out.read_text().splitlines()) == 1


def test_openai_compat_reply_shape():
    from specula_model.infer import openai_compat_reply

    body = openai_compat_reply("qwen-local", '{"verdict":"PASS","violations":[]}')
    assert body["choices"][0]["message"]["content"] == '{"verdict":"PASS","violations":[]}'
    assert body["object"] == "chat.completion"


def test_parse_chat_request_reads_png_data_url():
    from specula_model.infer import parse_chat_request

    png = b"\x89PNG\r\n\x1a\n"
    import base64
    b64 = base64.b64encode(png).decode()
    req = {
        "model": "specula",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Review this food label."},
            ]},
        ],
    }
    parsed = parse_chat_request(req)
    assert parsed["image"] == png
    assert parsed["model"] == "specula"


def test_benchmark_row_maps_contract_metrics():
    from specula_model.infer import benchmark_row

    summary = {
        "severity_weighted_recall": 0.5,
        "critical_recall": 0.25,
        "high_recall": 0.4,
        "precision": 0.9,
        "parse_rate": 1.0,
        "n_tasks": 10.0,
    }
    row = benchmark_row("rlvr", summary)
    assert row["model"] == "rlvr"
    assert row["severity_weighted_recall"] == 0.5
    assert row["critical_recall"] == 0.25
    assert row["high_recall"] == 0.4
    assert row["precision"] == 0.9
    assert row["parse_rate"] == 1.0
