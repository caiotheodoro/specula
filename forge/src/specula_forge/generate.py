"""Seeded synthetic label generator with violation injection + oracle gate.

Draws true product facts, injects a known violation set, derives printed
fields that realize exactly that set, runs the oracle self-check gate
(regenerating on mismatch), renders the label to a PNG, and emits a `Task`
with a contamination signature. See CONTRACTS.md §3–§5.
"""

from __future__ import annotations

import hashlib
import io
import json
import random

from PIL import Image, ImageDraw, ImageFont

from .schema import Label, NutritionFacts, ProductFacts, Task, Verdict, ViolationType
from .verify import (
    CLAIM_LT, CLAIM_MAX, CLAIM_RULES, NUTRIENT_FIELD, RACC, TOP9_GROUPS,
    _claim_violations, rounded_dv, verify,
)

BASE_INGREDIENTS: dict[str, list[str]] = {
    "cereal_ready_to_eat": ["whole grain oats", "sugar", "corn flour", "salt", "malt flavor"],
    "cookies": ["enriched wheat flour", "sugar", "vegetable oil", "corn syrup", "salt"],
    "crackers": ["enriched wheat flour", "canola oil", "salt", "yeast", "malt"],
    "bread": ["whole wheat flour", "water", "yeast", "salt", "wheat gluten"],
    "snack_chips": ["potatoes", "vegetable oil", "salt", "corn starch"],
    "yogurt": ["cultured milk", "sugar", "strawberry puree", "pectin"],
    "beverages": ["water", "fruit juice concentrate", "sugar", "citric acid", "natural flavor"],
    "soup_ready_to_serve": ["chicken broth", "carrots", "celery", "noodles", "salt", "spices"],
    "vegetables_leafy": ["spinach", "water", "salt"],
    "fruits": ["strawberries", "sugar", "pectin"],
    "candy": ["sugar", "corn syrup", "palm oil", "chocolate liquor", "soy lecithin"],
    "frozen_dessert": ["milk", "cream", "sugar", "vanilla extract", "egg yolks"],
    "pizza": ["wheat flour", "tomato sauce", "mozzarella cheese", "water", "yeast", "salt"],
}

ALLERGEN_TOKENS: dict[str, str] = {
    "milk": "whey protein",
    "eggs": "pasteurized eggs",
    "fish": "tuna",
    "crustacean": "shrimp",
    "tree_nuts": "almonds",
    "peanuts": "peanuts",
    "wheat": "whole wheat flour",
    "soybeans": "soy lecithin",
    "sesame": "sesame seeds",
}

VALID_CLAIMS = [
    "low fat", "low sodium", "low cholesterol", "low saturated fat", "low calorie",
    "fat free", "sugar free", "sodium free", "very low sodium",
    "good source of calcium", "good source of iron", "good source of fiber",
    "high in vitamin d", "excellent source of potassium",
]

# per-100g-style nutrient templates keyed by category -> (cal, fat, sat, chol,
# sodium, carb, fiber, sugar, added, protein, vitd, ca, fe, k) at the RACC.
NUTRIENT_TEMPLATE: dict[str, dict[str, float]] = {
    "cereal_ready_to_eat": dict(cal=120, fat=1.5, sat=0.3, chol=0, sodium=180,
                                carb=26, fiber=3, sugar=7, added=5, protein=3,
                                vitd=1.0, ca=60, fe=2.5, k=120),
    "cookies": dict(cal=150, fat=7, sat=2.5, chol=0, sodium=110, carb=21,
                    fiber=1, sugar=10, added=9, protein=2, vitd=0, ca=10, fe=1, k=40),
    "crackers": dict(cal=130, fat=4, sat=0.5, chol=0, sodium=200, carb=20,
                     fiber=1, sugar=1, added=0, protein=3, vitd=0, ca=20, fe=1, k=60),
    "bread": dict(cal=90, fat=1.5, sat=0.2, chol=0, sodium=150, carb=15,
                  fiber=3, sugar=2, added=1, protein=4, vitd=0, ca=20, fe=1, k=70),
    "snack_chips": dict(cal=150, fat=10, sat=1.5, chol=0, sodium=170, carb=15,
                        fiber=1, sugar=0, added=0, protein=2, vitd=0, ca=10, fe=0.5, k=180),
    "yogurt": dict(cal=120, fat=2, sat=1.3, chol=10, sodium=70, carb=18,
                   fiber=0, sugar=15, added=8, protein=6, vitd=1.5, ca=180, fe=0, k=240),
    "beverages": dict(cal=110, fat=0, sat=0, chol=0, sodium=25, carb=27,
                      fiber=0, sugar=26, added=24, protein=0, vitd=0, ca=20, fe=0, k=100),
    "soup_ready_to_serve": dict(cal=90, fat=2.5, sat=0.8, chol=15, sodium=680,
                                carb=10, fiber=1, sugar=3, added=0, protein=5, vitd=0, ca=30, fe=1, k=200),
    "vegetables_leafy": dict(cal=25, fat=0.5, sat=0, chol=0, sodium=60, carb=4,
                             fiber=2, sugar=0, added=0, protein=2, vitd=1.5, ca=60, fe=1.5, k=340),
    "fruits": dict(cal=80, fat=0, sat=0, chol=0, sodium=0, carb=20,
                   fiber=3, sugar=15, added=8, protein=1, vitd=1.0, ca=30, fe=1, k=200),
    "candy": dict(cal=140, fat=5, sat=3, chol=5, sodium=20, carb=24,
                  fiber=1, sugar=20, added=19, protein=2, vitd=0, ca=30, fe=1, k=70),
    "frozen_dessert": dict(cal=210, fat=11, sat=7, chol=55, sodium=80, carb=25,
                           fiber=0, sugar=22, added=18, protein=4, vitd=1.0, ca=110, fe=0, k=180),
    "pizza": dict(cal=290, fat=12, sat=5, chol=25, sodium=640, carb=34,
                  fiber=2, sugar=4, added=2, protein=13, vitd=1.0, ca=200, fe=2, k=220),
}

ALL = ["ALLERGEN_STATEMENT", "SERVING_SIZE", "CLAIM_THRESHOLD", "HEALTH_CLAIM",
       "MISSING_NUTRIENT", "DV_ERROR", "DV_ROUNDING", "HEALTHY_RULE",
       "FOP_RULE", "IDENTITY", "NET_QUANTITY", "FORMATTING",
       "LEGIBILITY", "ALLERGEN"]

# Static mix from openFDA recall / warning-letter frequency (not live API).
OPENFDA_WEIGHTS: dict[str, float] = {
    "ALLERGEN": 0.22, "ALLERGEN_STATEMENT": 0.10, "CLAIM_THRESHOLD": 0.12,
    "HEALTH_CLAIM": 0.08, "SERVING_SIZE": 0.08, "MISSING_NUTRIENT": 0.06,
    "DV_ERROR": 0.06, "DV_ROUNDING": 0.03, "HEALTHY_RULE": 0.05,
    "FOP_RULE": 0.04, "IDENTITY": 0.04, "NET_QUANTITY": 0.04,
    "FORMATTING": 0.04, "LEGIBILITY": 0.04,
}


def _pick_facts(rng: random.Random, category: str) -> ProductFacts:
    _, racc_grams = RACC[category]
    t = NUTRIENT_TEMPLATE[category]
    g = racc_grams / 100.0  # scale from per-100g to per-serving

    def j(v: float, frac: float = 0.08) -> float:
        return v * (1.0 + rng.uniform(-frac, frac))

    return ProductFacts(
        category=category,
        racc_household=RACC[category][0],
        racc_grams=racc_grams,
        serving_grams=racc_grams,
        ingredients=BASE_INGREDIENTS[category][:],
        allergens_present=[],
        nutrients=NutritionFacts(
            serving_size_household=RACC[category][0],
            serving_size_grams=racc_grams,
            servings_per_container=round(rng.uniform(1, 8), 1),
            calories=round(j(t["cal"] * g, 0.05)),
            total_fat_g=round(j(t["fat"] * g), 1),
            sat_fat_g=round(j(t["sat"] * g), 1),
            cholesterol_mg=round(j(t["chol"] * g)),
            sodium_mg=round(j(t["sodium"] * g)),
            total_carb_g=round(j(t["carb"] * g), 1),
            dietary_fiber_g=round(j(t["fiber"] * g), 1),
            total_sugars_g=round(j(t["sugar"] * g), 1),
            added_sugars_g=round(j(t["added"] * g), 1),
            protein_g=round(j(t["protein"] * g), 1),
            vitamin_d_mcg=round(j(t["vitd"] * g), 1),
            calcium_mg=round(j(t["ca"] * g)),
            iron_mg=round(j(t["fe"] * g), 1),
            potassium_mg=round(j(t["k"] * g)),
        ),
    )


def _declare_dv(nf: NutritionFacts) -> dict[str, float]:
    out: dict[str, float] = {}
    for nutrient in ("total_fat", "sat_fat", "cholesterol", "sodium", "total_carb",
                     "fiber", "added_sugars", "protein", "vit_d", "calcium", "iron",
                     "potassium"):
        out[nutrient] = rounded_dv(_percent(nf, nutrient))
    return out


def _percent(nf: NutritionFacts, nutrient: str) -> float:
    from .verify import DV_REF
    return getattr(nf, NUTRIENT_FIELD[nutrient]) / DV_REF[nutrient] * 100.0


def _copy_nf(nf: NutritionFacts) -> NutritionFacts:
    return nf.model_copy(deep=True)


def _ingredient_groups(ingredients: list[str]) -> set[str]:
    """Allergen groups that appear in an ingredient list (token match)."""
    joined = " ".join(ingredients).lower()
    return {g for g, tokens in TOP9_GROUPS.items()
            if any(t in joined for t in tokens)}


def _pick_allergens(rng: random.Random, facts: ProductFacts, n: int) -> None:
    """Derive true allergens from the ingredient list; add ingredients until
    at least `n` groups are present."""
    present = _ingredient_groups(facts.ingredients)
    all_groups = list(TOP9_GROUPS.keys())
    rng.shuffle(all_groups)
    for group in all_groups:
        if len(present) >= n:
            break
        if group not in present:
            facts.ingredients.append(ALLERGEN_TOKENS[group])
            present.add(group)
    facts.allergens_present = sorted(_ingredient_groups(facts.ingredients))


def _contains_word(group: str) -> str:
    """Oracle matches Contains via TOP9 tokens, not group keys (tree_nuts)."""
    return TOP9_GROUPS[group][0]


def _sample_injections(rng: random.Random, n: int,
                       weights: dict[str, float] | None = None) -> list[ViolationType]:
    """Distinct types; never ALLERGEN + ALLERGEN_STATEMENT together."""
    w = weights or {name: 1.0 for name in ALL}
    chosen: list[ViolationType] = []
    taken: set[ViolationType] = set()
    for _ in range(n):
        pool = []
        for name in ALL:
            vt = ViolationType(name)
            if vt in taken:
                continue
            if vt is ViolationType.ALLERGEN and \
                    ViolationType.ALLERGEN_STATEMENT in taken:
                continue
            if vt is ViolationType.ALLERGEN_STATEMENT and \
                    ViolationType.ALLERGEN in taken:
                continue
            pool.append(vt)
        if not pool:
            break
        wts = [w.get(vt.value, 1.0) for vt in pool]
        pick = rng.choices(pool, weights=wts, k=1)[0]
        chosen.append(pick)
        taken.add(pick)
    return chosen


def _nudge_just_over(facts: ProductFacts, nf: NutritionFacts, claim: str) -> None:
    """Move the true/printed nutrient to just over the claim limit."""
    c = claim.strip().lower()
    if c in CLAIM_LT:
        nutrient, limit = CLAIM_LT[c]
        field = NUTRIENT_FIELD[nutrient]
        setattr(facts.nutrients, field, limit)
        setattr(nf, field, limit)
        return
    if c in CLAIM_MAX:
        nutrient = CLAIM_RULES[c]
        field = NUTRIENT_FIELD[nutrient]
        step = 1.0 if nutrient in ("calories", "cholesterol", "sodium") else 0.1
        value = CLAIM_MAX[c] + step
        setattr(facts.nutrients, field, value)
        setattr(nf, field, value)


def _failing_claims(facts: ProductFacts) -> list[str]:
    """Valid claims that would actually violate for this product (lockstep
    with the oracle via _claim_violations)."""
    return [c for c in VALID_CLAIMS if _claim_violations([c], facts.nutrients)]


def generate_label(rng: random.Random, category: str,
                   injections: list[ViolationType] | None = None,
                   difficulty: float = 0.5) -> Label:
    """Generate one label that fires exactly the given violations."""
    if injections is None:
        injections = []
    facts = _pick_facts(rng, category)
    inj = set(injections)
    need_allergen = ViolationType.ALLERGEN in inj or \
        ViolationType.ALLERGEN_STATEMENT in inj
    min_allergens = 2 if ViolationType.ALLERGEN_STATEMENT in inj else (1 if need_allergen else 0)
    n_allergen = max(rng.choice([0, 1, 2]), min_allergens)
    _pick_allergens(rng, facts, n_allergen)
    label = _realize(facts, injections, difficulty, rng)
    label.id = f"{category}-{rng.randint(0, 10**9):09d}"
    label.product_category = category
    label.product_name = category.replace("_", " ").title()
    label.difficulty = difficulty
    return label


def _realize(facts: ProductFacts, injections: list[ViolationType],
             difficulty: float, rng: random.Random) -> Label:
    nf = _copy_nf(facts.nutrients)
    ing = facts.ingredients[:]
    contains = [_contains_word(g) for g in facts.allergens_present]
    claims: list[str] = []
    health: list[str] = []
    healthy = False
    fop = None
    identity = f"{facts.category.replace('_', ' ').title()} Product"
    net_qty = "Net Wt 12 oz (340 g)"
    injected = set(injections)

    if ViolationType.ALLERGEN in injected:
        tokens_to_drop = [t for g in facts.allergens_present
                          for t in TOP9_GROUPS[g]]
        ing = [i for i in ing
               if not any(t in i.lower() for t in tokens_to_drop)]
        contains = []

    if ViolationType.ALLERGEN_STATEMENT in injected and facts.allergens_present:
        drop = facts.allergens_present[0]
        contains = [_contains_word(g) for g in facts.allergens_present if g != drop]
    if ViolationType.SERVING_SIZE in injected:
        if difficulty >= 0.6:
            nf.serving_size_household = facts.racc_household
            nf.serving_size_grams = round(
                facts.serving_grams * rng.choice([0.94, 1.06]), 1)
        else:
            nf.serving_size_household = rng.choice(["1/2 cup", "1 tbsp", "2 cups"])
            nf.serving_size_grams = round(
                facts.serving_grams * rng.choice([0.5, 1.3]), 1)

    if ViolationType.CLAIM_THRESHOLD in injected:
        failing = _failing_claims(facts)
        if failing:
            if difficulty >= 0.6:
                prefer = [c for c in failing
                          if c.lower() in CLAIM_MAX or c.lower() in CLAIM_LT]
                claim = rng.choice(prefer or failing)
                _nudge_just_over(facts, nf, claim)
            else:
                claim = rng.choice(failing)
            claims = [claim]

    if ViolationType.HEALTH_CLAIM in injected:
        health = ["cures heart disease with daily consumption"]

    if ViolationType.MISSING_NUTRIENT in injected:
        nf.omitted_nutrients = ["vit_d"]

    if ViolationType.HEALTHY_RULE in injected:
        healthy = True
        nf.added_sugars_g = max(nf.added_sugars_g, 6.0)  # > 10% DV of added sugars
        facts.nutrients.added_sugars_g = nf.added_sugars_g

    if ViolationType.FOP_RULE in injected:
        fop = "facts-on-front"
        nf.sodium_mg = max(nf.sodium_mg, 350.0)  # > 15% DV of sodium
        facts.nutrients.sodium_mg = nf.sodium_mg

    if ViolationType.IDENTITY in injected:
        identity = None

    if ViolationType.NET_QUANTITY in injected:
        net_qty = None

    if ViolationType.FORMATTING in injected:
        nf.formatting_flags = ["missing_gram_measure"]

    if ViolationType.LEGIBILITY in injected:
        nf.legibility_flags = ["small_print_under_6pt"]

    dv = _declare_dv(nf)  # recompute AFTER nutrient mutations
    if injected and ViolationType.DV_ERROR in injected and "sodium" in dv:
        dv["sodium"] = round(dv["sodium"] + 8.0, 1)
    if injected and ViolationType.DV_ROUNDING in injected and "calcium" in dv:
        dv["calcium"] = round(dv["calcium"] - 0.6, 1)

    nf.dv_declared = dv
    label = Label(
        id="", product_name="", product_category=facts.category,
        printed=nf,
        true=facts,
        printed_ingredients=ing,
        printed_contains=contains,
        printed_claims=claims,
        printed_health_claims=health,
        printed_healthy_flag=healthy,
        printed_fop_symbol=fop,
        statement_of_identity=identity,
        net_quantity=net_qty,
        difficulty=difficulty,
    )
    return label


def task(rng: random.Random, category: str, seed: int,
         n_violations: int = 1, difficulty: float = 0.5,
         max_tries: int = 12, mix: str | None = None) -> Task:
    """Generate a task whose oracle verdict fires exactly the injected set."""
    weights = OPENFDA_WEIGHTS if mix == "openfda" else None
    for _ in range(max_tries):
        inj: list[ViolationType] = []
        if n_violations > 0:
            inj = _sample_injections(rng, n_violations, weights)
        label = generate_label(rng, category, inj, difficulty)
        from .verify import oracle_gate
        if oracle_gate(label, set(inj)):
            img = render_png(label)
            sig = signature(label)
            return Task(
                task_id=f"{category}-{seed}-{rng.randint(0, 10**9):09d}",
                seed=seed,
                label=label,
                image_bytes_sha256=hashlib.sha256(img).hexdigest(),
                expected=verify(label),
                signature=sig,
            )
    raise RuntimeError(f"oracle gate failed for category={category} n_violations={n_violations}")


def signature(label: Label) -> str:
    """SHA-256 over the full ground-truth content (printed + true), excluding
    id/metadata — the "same label content" bar for contamination (CONTRACTS §5)."""
    data = {
        "printed": label.printed.model_dump(mode="json"),
        "true": label.true.model_dump(mode="json"),
        "ing": label.printed_ingredients,
        "contains": label.printed_contains,
        "claims": label.printed_claims,
        "health": label.printed_health_claims,
        "healthy": label.printed_healthy_flag,
        "fop": label.printed_fop_symbol,
        "identity": label.statement_of_identity,
        "net_qty": label.net_quantity,
    }
    body = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_NUTRIENT_LINES = (
    ("total_fat", "Total Fat"), ("sat_fat", "Saturated Fat"),
    ("cholesterol", "Cholesterol (mg)"), ("sodium", "Sodium (mg)"),
    ("total_carb", "Total Carbohydrate"), ("fiber", "Dietary Fiber"),
    ("total_sugars", "Total Sugars"), ("added_sugars", "Added Sugars"),
    ("protein", "Protein"), ("vit_d", "Vitamin D (mcg)"),
    ("calcium", "Calcium (mg)"), ("iron", "Iron (mg)"),
    ("potassium", "Potassium (mg)"),
)


def _panel_lines(label: Label) -> list[str]:
    """Plain-text panel lines (same content render_png draws)."""
    nf = label.printed
    lines: list[str] = [label.product_name or "PRODUCT"]
    if label.statement_of_identity:
        lines.append(label.statement_of_identity)
    if label.net_quantity:
        lines.append(label.net_quantity)
    if label.printed_healthy_flag:
        lines.append("HEALTHY")
    if label.printed_fop_symbol:
        lines.append("[FOP SYMBOL]")
    for c in label.printed_claims:
        lines.append(c.upper())
    for hc in label.printed_health_claims:
        lines.append(hc)
    lines.append("Nutrition Facts")
    if "missing_gram_measure" in nf.formatting_flags:
        lines.append(f"Serving Size {nf.serving_size_household}")
    else:
        lines.append(
            f"Serving Size {nf.serving_size_household} ({nf.serving_size_grams:g}g)")
    lines.append(f"Servings {nf.servings_per_container:g}")
    lines.append(f"Calories {nf.calories:g}")
    for key, name in _NUTRIENT_LINES:
        if key in nf.omitted_nutrients:
            continue
        amount = getattr(nf, NUTRIENT_FIELD[key])
        dv = nf.dv_declared.get(key)
        dv_txt = "" if dv is None else f"  {dv:g}%"
        lines.append(f"{name}: {amount:g}{dv_txt}")
    if label.printed_ingredients:
        lines.append("Ingredients: " + ", ".join(label.printed_ingredients))
    if label.printed_contains:
        lines.append("Contains: " + ", ".join(label.printed_contains))
    return lines


def render_png(label: Label) -> bytes:
    """Render a simple composite label panel to PNG bytes."""
    nf = label.printed
    W, H = 600, 900
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    if "small_print_under_6pt" in nf.legibility_flags:
        font = ImageFont.load_default(size=8)
        small = ImageFont.load_default(size=6)
    else:
        font = ImageFont.load_default(size=18)
        small = ImageFont.load_default(size=14)

    y = 20
    title = label.product_name or "PRODUCT"
    claims_u = {c.upper() for c in label.printed_claims}
    for line in _panel_lines(label):
        if line in {title, "HEALTHY", "Nutrition Facts"} or line in claims_u:
            use, step = font, 22
            fill = "green" if line == "HEALTHY" else "black"
        else:
            use, step = small, 16
            fill = "blue" if line == "[FOP SYMBOL]" else "black"
        d.text((20, y), line, fill=fill, font=use)
        y += step

    img = _augment_png(img, label)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _augment_png(img: Image.Image, label: Label) -> Image.Image:
    """OCR-ish noise / rotation / glare, scaled by difficulty. Deterministic."""
    if label.difficulty < 0.25:
        return img
    seed = int(hashlib.sha256(
        f"{label.id}:{label.difficulty:.2f}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    angle = (label.difficulty - 0.25) * 10.0
    if rng.random() < 0.5:
        angle = -angle
    out = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor="white")
    px = out.load()
    w, h = out.size
    n = int(w * h * 0.008 * label.difficulty)
    for _ in range(n):
        x, y = rng.randrange(w), rng.randrange(h)
        jitter = rng.randint(-35, 35)
        r, g, b = px[x, y]
        px[x, y] = (
            max(0, min(255, r + jitter)),
            max(0, min(255, g + jitter)),
            max(0, min(255, b + jitter)),
        )
    # corner glare
    fade = int(40 + 80 * label.difficulty)
    for i in range(min(80, w)):
        for j in range(min(80, h)):
            r, g, b = px[i, j]
            bump = int(fade * (1 - i / 80) * (1 - j / 80))
            px[i, j] = (min(255, r + bump), min(255, g + bump), min(255, b + bump))
    return out
