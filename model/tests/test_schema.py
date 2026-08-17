"""Parser contract for specula_model.schema.parse."""

from specula_forge.schema import Severity, Verdict, Violation, ViolationType
from specula_model.schema import VerdictOut, parse


def test_parse_empty_is_none():
    assert parse("") is None


def test_parse_valid_verdict_json():
    out = parse('{"verdict": "PASS", "violations": []}')
    assert isinstance(out, VerdictOut)
    assert out.verdict == "PASS"
    assert out.violations == []


def test_parse_prose_wrapped_json():
    raw = 'Here is my review:\n{"verdict": "FLAG", "violations": []}\nDone.'
    out = parse(raw)
    assert isinstance(out, VerdictOut)
    assert out.verdict == "FLAG"


def test_parse_invalid_json_is_none():
    assert parse("not json") is None
    assert parse("{not valid") is None


def test_forge_flag_allergen_round_trips_through_parse():
    verdict = Verdict(
        verdict="FLAG",
        violations=[
            Violation(
                type=ViolationType.ALLERGEN,
                severity=Severity.CRITICAL,
                cfr="21 CFR 101.4 / FALCPA",
                observed="undeclared milk",
                expected="Contains: milk",
                correction="declare milk",
            )
        ],
    )
    out = parse(verdict.model_dump_json())
    assert isinstance(out, VerdictOut)
    assert out.verdict == "FLAG"
    assert len(out.violations) == 1
    assert out.violations[0].type == "ALLERGEN"
    assert out.violations[0].cfr == "21 CFR 101.4 / FALCPA"
