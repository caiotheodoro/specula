"""Specula verifier-as-oracle.

Deterministic rule engine over public law (21 CFR 101.9, 101.12, 101.13,
101.54-101.60, 101.72-101.83, FALCPA + sesame, 2026 "healthy"/FOP rules).
Given a label's printed fields and its true product facts, recomputes every
violation independently of the model. The generator's self-check gate demands
oracle(printed, true) == injected set exactly.

Sources to re-verify against eCFR during P1: RACC table (21 CFR 101.12(b)),
DV reference values (21 CFR 101.9(c)(8)(iv)), 2026 healthy/FOP final rules.
"""

from __future__ import annotations

from .schema import (
    Label,
    NutritionFacts,
    ProductFacts,
    Severity,
    Verdict,
    Violation,
    ViolationType,
)

# ---------------------------------------------------------------------------
# Reference data (public law). RACC values verified against eCFR 21 CFR
# 101.12(b) Table 2 on 2026-08-17; DV values are the standard 2020 update
# (21 CFR 101.9(c)(8)(iv)) — full-table re-check pending (see DECISIONS.md).
# ---------------------------------------------------------------------------
RACC: dict[str, tuple[str, float]] = {
    "beverages": ("8 fl oz (240 mL)", 240.0),          # juices, nectars, fruit drinks
    "cereal_ready_to_eat": ("1 cup (40 g)", 40.0),     # RTE cereal 20-<43 g per cup
    "cookies": ("3 cookies (30 g)", 30.0),
    "crackers": ("5 crackers (30 g)", 30.0),           # snack crackers
    "bread": ("1 slice (50 g)", 50.0),
    "snack_chips": ("1 oz (28 g)", 30.0),              # snacks, all varieties
    "yogurt": ("1 cup (170 g)", 170.0),
    "soup_ready_to_serve": ("1 cup (245 g)", 245.0),
    "vegetables_leafy": ("2 cups (85 g)", 85.0),       # fresh/frozen vegetables
    "fruits": ("1 cup (140 g)", 140.0),
    "candy": ("1 oz (30 g)", 30.0),                    # all other candies
    "frozen_dessert": ("2/3 cup (65 g)", 65.0),        # ice cream/sherbet (g estimate)
    "pizza": ("1/4 pizza (140 g)", 140.0),             # not measurable with cup
}

# Reference daily values (21 CFR 101.9(c)(8)(iv), 2020 update)
DV_REF: dict[str, float] = {
    "calories": 2000.0,
    "total_fat": 78.0,
    "sat_fat": 20.0,
    "cholesterol": 300.0,
    "sodium": 2300.0,
    "total_carb": 275.0,
    "fiber": 28.0,
    "total_sugars": 50.0,
    "added_sugars": 50.0,
    "protein": 50.0,
    "vit_d": 20.0,
    "calcium": 1300.0,
    "iron": 18.0,
    "potassium": 4700.0,
}

NUTRIENT_FIELD: dict[str, str] = {
    "calories": "calories",
    "total_fat": "total_fat_g",
    "sat_fat": "sat_fat_g",
    "cholesterol": "cholesterol_mg",
    "sodium": "sodium_mg",
    "total_carb": "total_carb_g",
    "fiber": "dietary_fiber_g",
    "total_sugars": "total_sugars_g",
    "added_sugars": "added_sugars_g",
    "protein": "protein_g",
    "vit_d": "vitamin_d_mcg",
    "calcium": "calcium_mg",
    "iron": "iron_mg",
    "potassium": "potassium_mg",
}

MANDATORY_NUTRIENTS: list[str] = [
    "calories", "total_fat", "sat_fat", "trans_fat", "cholesterol", "sodium",
    "total_carb", "fiber", "total_sugars", "added_sugars", "protein",
    "vit_d", "calcium", "iron", "potassium",
]

# Top-9 major food allergens (FALCPA + sesame, 2025)
TOP9_GROUPS: dict[str, list[str]] = {
    "milk": ["milk", "whey", "casein", "butter", "cream"],
    "eggs": ["egg", "albumin"],
    "fish": ["fish", "tuna", "salmon", "cod", "anchovy"],
    "crustacean": ["shrimp", "crab", "lobster", "crawfish", "crustacean"],
    "tree_nuts": ["almond", "walnut", "cashew", "pecan", "hazelnut",
                  "pistachio", "macadamia", "brazil nut"],
    "peanuts": ["peanut"],
    "wheat": ["wheat", "flour", "gluten", "semolina"],
    "soybeans": ["soy", "soybean", "tofu", "edamame", "miso"],
    "sesame": ["sesame"],
}

# Authorized health claims (21 CFR 101.72-101.83)
AUTHORIZED_HEALTH_CLAIMS: dict[str, str] = {
    "calcium and osteoporosis": "21 CFR 101.72",
    "sodium and hypertension": "21 CFR 101.74",
    "dietary fat and cancer": "21 CFR 101.73",
    "saturated fat and cholesterol and coronary heart disease": "21 CFR 101.75",
    "fiber-containing grain products, fruits and vegetables and cancer": "21 CFR 101.76",
    "fruits and vegetables and cancer": "21 CFR 101.78",
    "folate and neural tube defects": "21 CFR 101.79",
    "dietary sugar alcohols and dental caries": "21 CFR 101.80",
    "soluble fiber and coronary heart disease": "21 CFR 101.81",
    "whole grains and coronary heart disease": "21 CFR 101.82",
    "potassium and blood pressure and stroke": "21 CFR 101.83",
    "plant sterol esters and coronary heart disease": "21 CFR 101.83",
}


# ---------------------------------------------------------------------------
# %DV math (21 CFR 101.9(c)(6) rounding)
# ---------------------------------------------------------------------------
def percent_dv(amount: float, nutrient: str) -> float:
    """Raw %DV (unrounded)."""
    dv = DV_REF[nutrient]
    return amount / dv * 100.0


def rounded_dv(raw: float) -> float:
    """DV rounding per 21 CFR 101.9(c)(6): <0.5% -> 0; 0.5<=x<5 -> 0.5
    (rendered '<1'); >=5 -> nearest whole %."""
    if raw < 0.5:
        return 0.0
    if raw < 5.0:
        return 0.5
    return round(raw + 1e-9)


def nutrient_value(nf: NutritionFacts, nutrient: str) -> float:
    return getattr(nf, NUTRIENT_FIELD[nutrient])


# ---------------------------------------------------------------------------
# Rule checks
# ---------------------------------------------------------------------------
def _check_allergens(label: Label) -> list[Violation]:
    out: list[Violation] = []
    printed = label.printed
    true = label.true
    ing = " ".join(label.printed_ingredients).lower()
    contains = " ".join(label.printed_contains).lower()
    declared = set()
    for group, tokens in TOP9_GROUPS.items():
        hits = [t for t in tokens if t in ing]
        if hits:
            declared.add(group)
    if label.printed_contains:
        for group, tokens in TOP9_GROUPS.items():
            if any(t in contains for t in tokens):
                declared.add(group)

    for group in true.allergens_present:
        if group not in declared:
            out.append(Violation(
                type=ViolationType.ALLERGEN,
                severity=Severity.CRITICAL,
                cfr="FALCPA, 21 CFR 101.4 / 21 CFR 101.2.20",
                observed=f"allergen '{group}' not declared in ingredient list or Contains statement",
                expected=f"'{group}' must be declared",
                correction=f"declare '{group}' in ingredient list or Contains statement",
            ))
    # Inconsistent Contains vs ingredients
    if label.printed_contains:
        ing_declared = set()
        for group, tokens in TOP9_GROUPS.items():
            if any(t in ing for t in tokens):
                ing_declared.add(group)
        contains_declared = set()
        for group, tokens in TOP9_GROUPS.items():
            if any(t in contains for t in tokens):
                contains_declared.add(group)
        for group in ing_declared - contains_declared:
            out.append(Violation(
                type=ViolationType.ALLERGEN_STATEMENT,
                severity=Severity.HIGH,
                cfr="FALCPA, 21 CFR 101.4 / 21 CFR 101.2.20",
                observed=f"'{group}' listed in ingredients but missing from Contains statement",
                expected="Contains statement must match ingredient-declared allergens",
                correction=f"add '{group}' to the Contains statement",
            ))
    return out


def _check_serving_size(printed, true) -> list[Violation]:
    out: list[Violation] = []
    if true.category not in RACC:
        return out
    expected_household, expected_grams = RACC[true.category]
    if abs(printed.serving_size_grams - expected_grams) > 1e-6 or \
            printed.serving_size_household.strip().lower() != expected_household.lower():
        out.append(Violation(
            type=ViolationType.SERVING_SIZE,
            severity=Severity.CRITICAL,
            cfr="21 CFR 101.9(b) / 101.12(b)",
            observed=f"serving size declared '{printed.serving_size_household} ({printed.serving_size_grams:g} g)'",
            expected=f"'{expected_household} ({expected_grams:g} g)' for category '{true.category}'",
            correction=f"restate serving size as '{expected_household} ({expected_grams:g} g)'",
        ))
    return out


CLAIM_RULES: dict[str, str] = {
    "low fat": "total_fat",
    "low saturated fat": "sat_fat",
    "low cholesterol": "cholesterol",
    "low sodium": "sodium",
    "low calorie": "calories",
}
CLAIM_MAX: dict[str, float] = {
    "low fat": 3.0,           # g
    "low saturated fat": 1.0,  # g
    "low cholesterol": 20.0,   # mg
    "low sodium": 140.0,       # mg
    "low calorie": 40.0,       # kcal
}
SOURCE_CLAIMS: dict[str, float] = {  # "good source of X" = 10-19% DV
    "calcium": "calcium",
    "iron": "iron",
    "vitamin d": "vit_d",
    "potassium": "potassium",
    "protein": "protein",
    "fiber": "fiber",
}


def _claim_violations(claims: list[str], nf: NutritionFacts) -> list[Violation]:
    """Pure claim-check used by the oracle AND by the generator to pick
    failing claims (keeps injection and detection in lockstep)."""
    out: list[Violation] = []
    for claim in claims:
        c = claim.strip().lower()
        if c in CLAIM_MAX:
            nutrient = CLAIM_RULES[c]
            amount = nutrient_value(nf, nutrient)
            limit = CLAIM_MAX[c]
            if amount > limit:
                out.append(Violation(
                    type=ViolationType.CLAIM_THRESHOLD,
                    severity=Severity.CRITICAL,
                    cfr="21 CFR 101.13 / 101.54-101.60",
                    observed=f"'{claim}' claimed but {nutrient} = {amount:g}",
                    expected=f"{nutrient} <= {limit:g} for '{claim}'",
                    correction="remove the claim or reformulate",
                ))
            continue
        prefix = None
        for p in ("excellent source of", "high in", "good source of"):
            if c.startswith(p):
                prefix = p
                break
        if prefix:
            rest = c[len(prefix):].strip()
            nutrient = next((dv_key for key, dv_key in SOURCE_CLAIMS.items()
                             if key in rest), None)
            if nutrient:
                dv = percent_dv(nutrient_value(nf, nutrient), nutrient)
                if prefix == "good source of" and not (10.0 <= dv < 20.0):
                    out.append(Violation(
                        type=ViolationType.CLAIM_THRESHOLD,
                        severity=Severity.CRITICAL,
                        cfr="21 CFR 101.54(e)",
                        observed=f"'{claim}' claimed but {nutrient} = {dv:.1f}% DV",
                        expected="10-19% DV for 'good source'",
                        correction="remove the claim or reformulate",
                    ))
                elif prefix in ("excellent source of", "high in") and dv < 20.0:
                    out.append(Violation(
                        type=ViolationType.CLAIM_THRESHOLD,
                        severity=Severity.CRITICAL,
                        cfr="21 CFR 101.54(b)/(c)",
                        observed=f"'{claim}' claimed but {nutrient} = {dv:.1f}% DV",
                        expected=">=20% DV for 'excellent/high in'",
                        correction="remove the claim or reformulate",
                    ))
            continue
        out.append(Violation(
            type=ViolationType.CLAIM_THRESHOLD,
            severity=Severity.CRITICAL,
            cfr="21 CFR 101.13(b)",
            observed=f"unrecognized nutrient-content claim '{claim}'",
            expected="claim must meet 21 CFR 101.13 definitions",
            correction="reword or remove the claim",
        ))
    return out


def _check_claims(label: Label) -> list[Violation]:
    return _claim_violations(label.printed_claims, label.true.nutrients)


def _check_health_claims(label: Label) -> list[Violation]:
    out: list[Violation] = []
    for hc in label.printed_health_claims:
        key = hc.strip().lower()
        if key not in AUTHORIZED_HEALTH_CLAIMS:
            out.append(Violation(
                type=ViolationType.HEALTH_CLAIM,
                severity=Severity.CRITICAL,
                cfr="21 CFR 101.14(c)",
                observed=f"health claim '{hc}' not on the authorized list",
                expected="only claims in 21 CFR 101.72-101.83 are authorized",
                correction="remove or qualify the health claim",
            ))
    return out


def _check_missing_nutrients(printed: NutritionFacts) -> list[Violation]:
    out: list[Violation] = []
    for nutrient in printed.omitted_nutrients:
        if nutrient in MANDATORY_NUTRIENTS:
            out.append(Violation(
                type=ViolationType.MISSING_NUTRIENT,
                severity=Severity.HIGH,
                cfr="21 CFR 101.9(c)",
                observed=f"mandatory nutrient line '{nutrient}' omitted from Nutrition Facts",
                expected=f"'{nutrient}' must appear on the panel",
                correction="add the missing nutrient declaration",
            ))
    return out


def _check_formatting(printed: NutritionFacts) -> list[Violation]:
    out: list[Violation] = []
    if "missing_gram_measure" in printed.formatting_flags:
        out.append(Violation(
            type=ViolationType.FORMATTING,
            severity=Severity.MEDIUM,
            cfr="21 CFR 101.9(b)",
            observed="serving size line lacks the metric (gram) measure",
            expected="serving size must state the household measure and metric measure",
            correction="add the gram measure to the serving size line",
        ))
    return out


def _check_legibility(printed: NutritionFacts) -> list[Violation]:
    out: list[Violation] = []
    if "small_print_under_6pt" in printed.legibility_flags:
        out.append(Violation(
            type=ViolationType.LEGIBILITY,
            severity=Severity.LOW,
            cfr="21 CFR 101.9(h)",
            observed="label print smaller than the 6 pt minimum",
            expected="all Nutrition Facts text at or above the minimum type size",
            correction="increase type size",
        ))
    return out


def _check_dv(printed: NutritionFacts) -> list[Violation]:
    out: list[Violation] = []
    for nutrient, declared in printed.dv_declared.items():
        if nutrient not in DV_REF or nutrient not in NUTRIENT_FIELD:
            continue
        amount = nutrient_value(printed, nutrient)
        expected = rounded_dv(percent_dv(amount, nutrient))
        diff = abs(declared - expected)
        if diff >= 1.0:
            out.append(Violation(
                type=ViolationType.DV_ERROR,
                severity=Severity.HIGH,
                cfr="21 CFR 101.9(c)(6)",
                observed=f"%DV declared for {nutrient} = {declared:g}, expected {expected:g}",
                expected=f"%DV of {nutrient} must be {expected:g}",
                correction=f"restate %DV as {expected:g}",
            ))
        elif declared != expected and diff < 1.0:
            out.append(Violation(
                type=ViolationType.DV_ROUNDING,
                severity=Severity.LOW,
                cfr="21 CFR 101.9(c)(6)",
                observed=f"%DV declared for {nutrient} = {declared:g} (rounding slip, expected {expected:g})",
                expected=f"%DV of {nutrient} must be {expected:g}",
                correction=f"restate %DV as {expected:g}",
            ))
    return out


def _healthy_criteria(nf: NutritionFacts) -> list[Violation]:
    """2026 'healthy' implied-claim criteria — verified against 21 CFR
    101.65(d)(3)(iii) Table 4 (mixed product): added sugars <=10% DV,
    sodium <=15% DV, saturated fat <=10% DV."""
    out: list[Violation] = []
    added_dv = percent_dv(nf.added_sugars_g, "added_sugars")
    sat_dv = percent_dv(nf.sat_fat_g, "sat_fat")
    sodium_dv = percent_dv(nf.sodium_mg, "sodium")
    limits = {"added sugars <= 10% DV": added_dv <= 10.0,
              "saturated fat <= 10% DV": sat_dv <= 10.0,
              "sodium <= 15% DV": sodium_dv <= 15.0}
    for label, ok in limits.items():
        if not ok:
            out.append(Violation(
                type=ViolationType.HEALTHY_RULE,
                severity=Severity.HIGH,
                cfr="21 CFR 101.65(d)(3)(iii)",
                observed=f"'{label}' violated",
                expected="product must meet all 2026 healthy criteria",
                correction="remove the claim or reformulate",
            ))
    return out


def _check_healthy(label: Label) -> list[Violation]:
    if not label.printed_healthy_flag:
        return []
    return _healthy_criteria(label.true.nutrients)


def _check_fop(label: Label) -> list[Violation]:
    if label.printed_fop_symbol is None:
        return []
    if _healthy_criteria(label.true.nutrients):
        return [Violation(
            type=ViolationType.FOP_RULE,
            severity=Severity.HIGH,
            cfr="2026 front-of-package rule",
            observed="FOP symbol present but product fails underlying criteria",
            expected="FOP symbol requires compliant nutrient profile",
            correction="remove FOP symbol or reformulate",
        )]
    return []


def _check_identity_net_qty(label: Label) -> list[Violation]:
    out: list[Violation] = []
    if not label.statement_of_identity:
        out.append(Violation(
            type=ViolationType.IDENTITY,
            severity=Severity.MEDIUM,
            cfr="21 CFR 101.3",
            observed="statement of identity missing",
            expected="principal display panel must state the identity of the food",
            correction="add the statement of identity",
        ))
    if not label.net_quantity:
        out.append(Violation(
            type=ViolationType.NET_QUANTITY,
            severity=Severity.MEDIUM,
            cfr="21 CFR 101.105",
            observed="net quantity declaration missing",
            expected="net contents must be declared in the lower 30% of the PDP",
            correction="add net quantity declaration",
        ))
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def verify(label: Label) -> Verdict:
    """Oracle: recompute the full violation set for a label."""
    printed, true = label.printed, label.true
    violations: list[Violation] = []
    violations += _check_allergens(label)
    violations += _check_serving_size(printed, true)
    violations += _check_claims(label)
    violations += _check_health_claims(label)
    violations += _check_missing_nutrients(printed)
    violations += _check_dv(printed)
    violations += _check_formatting(printed)
    violations += _check_legibility(printed)
    violations += _check_healthy(label)
    violations += _check_fop(label)
    violations += _check_identity_net_qty(label)
    return Verdict(
        verdict="PASS" if not violations else "FLAG",
        violations=violations,
    )


def violation_keys(verdict: Verdict) -> set[tuple[ViolationType, Severity]]:
    """(type, severity) set for gate/score comparison (critical classes also
    match on severity per CONTRACTS §2)."""
    return {(v.type, v.severity) for v in verdict.violations}


def oracle_gate(label: Label, injected: set[ViolationType]) -> bool:
    """Self-check gate: oracle must fire exactly the injected types."""
    return {v.type for v in verify(label).violations} == injected