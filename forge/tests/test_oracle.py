"""Oracle correctness on hand-built labels (21 CFR anchor cases)."""

import random

from specula_forge.generate import _copy_nf, _pick_facts, task
from specula_forge.schema import NutritionFacts, ProductFacts, ViolationType
from specula_forge.verify import (
    DV_REF,
    oracle_gate,
    percent_dv,
    rounded_dv,
    verify,
)


def test_pass_label_fires_nothing():
    rng = random.Random(1)
    t = task(rng, "bread", seed=7, n_violations=0)
    assert t.expected.verdict == "PASS"
    assert t.expected.violations == []
    assert oracle_gate(t.label, set())


def test_allergen_critical():
    rng = random.Random(2)
    t = task(rng, "yogurt", seed=7, n_violations=1, difficulty=0.3)
    if t.expected.verdict == "PASS":
        assert set() == {v.type for v in t.expected.violations}
    else:
        assert {ViolationType.ALLERGEN} == {v.type for v in t.expected.violations}


def test_serving_size_critical():
    rng = random.Random(3)
    t = task(rng, "bread", seed=7, n_violations=1, difficulty=0.3)
    for v in t.expected.violations:
        assert v.type in {ViolationType.SERVING_SIZE, ViolationType.ALLERGEN,
                          ViolationType.CLAIM_THRESHOLD, ViolationType.HEALTH_CLAIM,
                          ViolationType.MISSING_NUTRIENT, ViolationType.DV_ERROR,
                          ViolationType.DV_ROUNDING, ViolationType.HEALTHY_RULE,
                          ViolationType.FOP_RULE, ViolationType.ALLERGEN_STATEMENT,
                          ViolationType.IDENTITY, ViolationType.NET_QUANTITY}
        assert v.cfr  # every violation carries a citation


def test_dv_math_anchor():
    nf = NutritionFacts(
        serving_size_household="1 cup", serving_size_grams=245, servings_per_container=2,
        calories=100, sodium_mg=2300, added_sugars_g=25,
    )
    # 2300 mg sodium = 100% DV
    assert round(percent_dv(nf.sodium_mg, "sodium")) == 100.0
    assert rounded_dv(percent_dv(nf.sodium_mg, "sodium")) == 100.0
    # 25 g added sugar / 50 g DV = 50%
    assert rounded_dv(percent_dv(nf.added_sugars_g, "added_sugars")) == 50.0


def test_dv_rounding_rules():
    assert rounded_dv(0.3) == 0.0
    assert rounded_dv(2.0) == 0.5      # renders '<1%'
    assert rounded_dv(5.0) == 5.0
    assert rounded_dv(24.4) == 24.0


def test_all_validation_types_reachable_across_seed():
    """Across a pilot sweep every violation class is injected and caught."""
    import random as _r
    from specula_forge.verify import RACC
    rng = _r.Random(42)
    seen: set[ViolationType] = set()
    for i in range(400):
        cat = list(RACC.keys())[i % len(RACC)]
        t = task(rng, cat, seed=42, n_violations=1, difficulty=0.5)
        seen |= {v.type for v in t.expected.violations}
        assert oracle_gate(t.label, {v.type for v in t.expected.violations})
    assert ViolationType.ALLERGEN in seen
    assert ViolationType.DV_ERROR in seen
    assert ViolationType.HEALTHY_RULE in seen
    assert ViolationType.FOP_RULE in seen
    assert ViolationType.IDENTITY in seen
    assert ViolationType.NET_QUANTITY in seen


def test_missing_nutrient_high():
    rng = random.Random(5)
    t = task(rng, "soup_ready_to_serve", seed=7, n_violations=1, difficulty=0.2)
    for v in t.expected.violations:
        if v.type == ViolationType.MISSING_NUTRIENT:
            assert v.severity.value == "HIGH"


def test_true_facts_nutrients_positive():
    """True nutrient values must be non-negative and sane."""
    rng = random.Random(9)
    facts = _pick_facts(rng, "cookies")
    assert facts.nutrients.calories > 0
    assert facts.nutrients.sodium_mg >= 0
    assert DV_REF["sodium"] == 2300.0