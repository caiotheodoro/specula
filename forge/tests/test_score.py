"""Citation exact-match scoring (caught stays type+severity)."""

from specula_forge.schema import Verdict, Violation, ViolationType
from specula_forge.score import score_predictions, summarize


def _allergen(cfr: str) -> Verdict:
    return Verdict(
        verdict="FLAG",
        violations=[
            Violation(
                type=ViolationType.ALLERGEN,
                severity="CRITICAL",
                cfr=cfr,
                observed="o",
                expected="e",
                correction="c",
            )
        ],
    )


def test_citation_mismatch_caught_but_exact_match_zero():
    expected = _allergen("21 CFR 101.4")
    predicted = _allergen("21 CFR 999.99")
    row = score_predictions(expected, predicted)
    assert row["caught"] == 1.0
    assert row["citation_hits"] == 0.0
    assert row["citation_total"] == 1.0
    assert summarize([row])["citation_exact_match"] == 0


def test_citation_match_exact_match_one():
    expected = _allergen("21 CFR 101.4")
    predicted = _allergen("21 CFR 101.4")
    row = score_predictions(expected, predicted)
    assert row["citation_hits"] == 1.0
    assert row["citation_total"] == 1.0
    assert summarize([row])["citation_exact_match"] == 1
