"""RLVR reward: raw completion -> forge oracle scalar.

CPU-only. TRL GRPOTrainer(reward_funcs=...) compatible. Outcome verifier
only — no PRM. Does not import Modal/GPU.
"""

from __future__ import annotations

from specula_forge.schema import Verdict
from specula_forge.score import reward

from .schema import to_forge_verdict


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return str(completion)
    parts: list[str] = []
    for msg in completion:
        if isinstance(msg, str):
            parts.append(msg)
            continue
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("text") or "")
    return "".join(parts)


def rlvr_reward(raw_completion: str, expected: Verdict) -> float:
    predicted = to_forge_verdict(raw_completion)
    return reward(expected, predicted)


def oracle_reward_func(completions: list, expected_verdict: list[str],
                       **kwargs) -> list[float]:
    """TRL GRPOTrainer reward_funcs callable.

    `expected_verdict` is a dataset column (JSON forge Verdict). TRL repeats
    other columns to match num_generations, so these lists are 1:1.
    """
    out = []
    for completion, exp_json in zip(completions, expected_verdict):
        expected = Verdict.model_validate_json(exp_json)
        out.append(rlvr_reward(_completion_text(completion), expected))
    return out
