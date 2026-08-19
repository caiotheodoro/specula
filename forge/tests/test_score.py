"""Citation exact-match scoring (caught stays type+severity)."""

from specula_forge.schema import Verdict, Violation, ViolationType
from specula_forge.score import reward, score_predictions, summarize


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



def test_reward_perfect_flag_is_recall_plus_verdict_bonus():
    expected = _allergen("21 CFR 101.4")
    assert reward(expected, expected) == 1.0 + 0.2


def test_reward_false_positive_subtracts_fp_penalty():
    expected = _allergen("21 CFR 101.4")
    extra = Violation(
        type=ViolationType.FORMATTING,
        severity="MEDIUM",
        cfr="21 CFR 101.9",
        observed="o",
        expected="e",
        correction="c",
    )
    predicted = Verdict(
        verdict="FLAG",
        violations=list(expected.violations) + [extra],
    )
    assert reward(expected, predicted) == 1.0 - 0.3 + 0.2


def test_reward_clean_pass_is_one_or_minus_one():
    expected = Verdict(verdict="PASS", violations=[])
    assert reward(expected, Verdict(verdict="PASS", violations=[])) == 1.0
    assert reward(expected, Verdict(verdict="FLAG", violations=[])) == -1.0


def test_reward_unparseable_is_minus_one():
    assert reward(_allergen("21 CFR 101.4"), None) == -1.0



def _fmt() -> Violation:
    return Violation(
        type=ViolationType.FORMATTING,
        severity="MEDIUM",
        cfr="21 CFR 101.9",
        observed="o",
        expected="e",
        correction="c",
    )


def test_precision_is_caught_over_emitted():
    expected = _allergen("21 CFR 101.4")
    predicted = Verdict(
        verdict="FLAG",
        violations=list(expected.violations) + [_fmt()],
    )
    row = score_predictions(expected, predicted)
    summary = summarize([row])
    assert summary["precision"] == 0.5


def test_critical_and_high_recall_are_separate():
    expected = Verdict(
        verdict="FLAG",
        violations=[
            Violation(
                type=ViolationType.ALLERGEN,
                severity="CRITICAL",
                cfr="21 CFR 101.4",
                observed="o",
                expected="e",
                correction="c",
            ),
            Violation(
                type=ViolationType.MISSING_NUTRIENT,
                severity="HIGH",
                cfr="21 CFR 101.9",
                observed="o",
                expected="e",
                correction="c",
            ),
        ],
    )
    predicted = Verdict(verdict="FLAG", violations=list(expected.violations[:1]))
    row = score_predictions(expected, predicted)
    summary = summarize([row])
    assert summary["critical_recall"] == 1.0
    assert summary["high_recall"] == 0.0



def test_unparseable_flag_counts_in_recall_denominator():
    row = score_predictions(_allergen("21 CFR 101.4"), None)
    assert row["total"] == 1.0
    assert row["caught"] == 0.0
    summary = summarize([row])
    assert summary["parse_rate"] == 0.0
    assert summary["severity_weighted_recall"] == 0.0
    assert summary["n_violation_tasks"] == 1.0
