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
    from specula_forge.generate import ALL, generate_label
    from specula_forge.verify import RACC
    rng = _r.Random(42)
    seen: set[ViolationType] = set()
    for i in range(400):
        cat = list(RACC.keys())[i % len(RACC)]
        t = task(rng, cat, seed=42, n_violations=1, difficulty=0.5)
        seen |= {v.type for v in t.expected.violations}
        assert oracle_gate(t.label, {v.type for v in t.expected.violations})
    for name in ALL:
        vt = ViolationType(name)
        label = generate_label(rng, "yogurt", [vt])
        assert oracle_gate(label, {vt})
        seen.add(vt)
    assert seen == set(ViolationType)


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


# Statutory DV keys from 21 CFR 101.9 (c)(8)(iv) RDI + (c)(9) DRV, plus the
# 2,000-calorie footnote used as the calories reference. total_sugars is not
# in either table — see test_dv_ref_total_sugars_is_nonstatutory_alias.
_STATUTORY_DV_KEYS = (
    "calories",
    "total_fat",
    "sat_fat",
    "cholesterol",
    "sodium",
    "total_carb",
    "fiber",
    "added_sugars",
    "protein",
    "vit_d",
    "calcium",
    "iron",
    "potassium",
)


def test_dv_ref_matches_101_9_tables():
    """Pin DV_REF to 21 CFR 101.9 (c)(8)(iv) RDI + (c)(9) DRV (adults ≥4)."""
    for k in _STATUTORY_DV_KEYS:
        assert round(percent_dv(DV_REF[k], k)) == 100

    assert DV_REF["total_fat"] == 78.0
    assert DV_REF["sat_fat"] == 20.0
    assert DV_REF["cholesterol"] == 300.0
    assert DV_REF["sodium"] == 2300.0
    assert DV_REF["total_carb"] == 275.0
    assert DV_REF["fiber"] == 28.0
    assert DV_REF["added_sugars"] == 50.0
    assert DV_REF["protein"] == 50.0
    assert DV_REF["vit_d"] == 20.0
    assert DV_REF["calcium"] == 1300.0
    assert DV_REF["iron"] == 18.0
    assert DV_REF["potassium"] == 4700.0
    assert DV_REF["calories"] == 2000.0

    assert rounded_dv(percent_dv(39.0, "total_fat")) == 50.0
    assert rounded_dv(percent_dv(14.0, "fiber")) == 50.0
    assert rounded_dv(percent_dv(1300.0, "calcium")) == 100.0
    assert rounded_dv(percent_dv(9.0, "iron")) == 50.0


def test_dv_ref_total_sugars_is_nonstatutory_alias():
    # Recorded, not a (c)(9) row: total sugars has no statutory Daily Value.
    # Convenience alias of the added-sugars DRV; unused by _declare_dv.
    assert DV_REF["total_sugars"] == DV_REF["added_sugars"] == 50.0
