"""Model-output contract for Specula. Mirrors forge Verdict (camelCase).

The worker emits exactly this JSON (no thinking block in the answer, per
CONTRACTS.md §6). `parse` is the canonical JSON parser used by every eval.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field


class ViolationOut(BaseModel):
    type: str
    severity: str
    cfr: str
    observed: str = ""
    expected: str = ""
    correction: str = ""


class VerdictOut(BaseModel):
    verdict: str
    violations: list[ViolationOut] = Field(default_factory=list)


VIOLATION_TYPE_IDS = (
    "ALLERGEN", "HEALTH_CLAIM", "CLAIM_THRESHOLD", "SERVING_SIZE",
    "MISSING_NUTRIENT", "DV_ERROR", "HEALTHY_RULE", "FOP_RULE",
    "ALLERGEN_STATEMENT", "FORMATTING", "IDENTITY", "NET_QUANTITY",
    "DV_ROUNDING", "LEGIBILITY",
)
SEVERITY_IDS = ("CRITICAL", "HIGH", "MEDIUM", "LOW")

SYSTEM_PROMPT = (
    "You are a FDA food-label compliance reviewer. Given the label image, "
    "emit a JSON object {\"verdict\": \"PASS\"|\"FLAG\", \"violations\": "
    "[{type, severity, cfr, observed, expected, correction}]}. "
    "type must be one of: " + ", ".join(VIOLATION_TYPE_IDS) + ". "
    "severity must be one of: " + ", ".join(SEVERITY_IDS) + ". "
    "Verify against 21 CFR 101.9, 101.12, 101.13, 101.54-101.83 and FALCPA. "
    "Emit only the JSON."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse(raw: str) -> VerdictOut | None:
    """Parse a model answer into a VerdictOut; None = parse miss."""
    if not raw:
        return None
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        return VerdictOut.model_validate(data)
    except Exception:
        return None


def to_forge_verdict(raw: str):
    """Raw model text -> forge Verdict, or None if unparseable.

    Shared by benchmark_eval and RLVR so parse-to-score is one path.
    Unrecognized type/severity strings are dropped, not a parse miss.
    """
    out = parse(raw)
    if out is None:
        return None
    from specula_forge.schema import Severity, Verdict, Violation, ViolationType
    violations = []
    for v in out.violations:
        try:
            violations.append(Violation(
                type=ViolationType(v.type),
                severity=Severity(v.severity),
                cfr=v.cfr,
                observed=v.observed,
                expected=v.expected,
                correction=v.correction,
            ))
        except ValueError:
            continue
    verdict = out.verdict if out.verdict in ("PASS", "FLAG") else "FLAG"
    return Verdict(verdict=verdict, violations=violations)
