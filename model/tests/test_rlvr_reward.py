"""RLVR reward plumbing: raw completion -> forge oracle reward. No GPU."""

import json
import random

from specula_forge import generate
from specula_model.rlvr_reward import oracle_reward_func, rlvr_reward


def _raw_from_task(task) -> str:
    return json.dumps({
        "verdict": task.expected.verdict,
        "violations": [
            {
                "type": v.type.value,
                "severity": v.severity.value,
                "cfr": v.cfr,
                "observed": v.observed,
                "expected": v.expected,
                "correction": v.correction,
            }
            for v in task.expected.violations
        ],
    })


def test_rlvr_reward_perfect_match_beats_one():
    rng = random.Random(20)
    task = generate.task(rng, "cookies", seed=20, n_violations=1)
    assert rlvr_reward(_raw_from_task(task), task.expected) > 1.0


def test_rlvr_reward_unparseable_is_minus_one():
    rng = random.Random(21)
    task = generate.task(rng, "cookies", seed=21, n_violations=1)
    assert rlvr_reward("not json", task.expected) == -1.0


def test_oracle_reward_func_batch_aligned():
    rng = random.Random(22)
    tasks = [generate.task(rng, "cookies", seed=22 + i, n_violations=1)
             for i in range(3)]
    completions = [_raw_from_task(t) for t in tasks]
    expected_verdict = [t.expected.model_dump_json() for t in tasks]
    rewards = oracle_reward_func(completions, expected_verdict)
    assert len(rewards) == 3
    assert all(r > 1.0 for r in rewards)


def test_oracle_reward_func_accepts_conversational_completions():
    rng = random.Random(23)
    task = generate.task(rng, "cookies", seed=23, n_violations=1)
    raw = _raw_from_task(task)
    completions = [[{"role": "assistant", "content": [{"type": "text", "text": raw}]}]]
    expected = [task.expected.model_dump_json()]
    rewards = oracle_reward_func(completions, expected)
    assert len(rewards) == 1
    assert rewards[0] > 1.0
