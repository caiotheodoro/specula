"""Generator, rendering, scoring, and contamination tests."""

import random

from specula_forge import contamination, generate
from specula_forge.schema import Verdict, Violation, ViolationType
from specula_forge.score import class_recall, score_predictions, summarize
from specula_forge.verify import oracle_gate, verify


def test_render_is_deterministic_png():
    rng = random.Random(10)
    t = generate.task(rng, "cookies", seed=7, n_violations=0)
    a = generate.render_png(t.label)
    b = generate.render_png(t.label)
    assert a == b
    assert a[:8] == b"\x89PNG\r\n\x1a\n"


def test_signature_stable_and_contamination_roc():
    rng = random.Random(11)
    train = [generate.task(rng, "bread", seed=7, n_violations=1) for _ in range(20)]
    eval_tasks = [generate.task(rng, "bread", seed=8, n_violations=1) for _ in range(20)]
    index = contamination.build_train_index(train)
    clean = contamination.probe(index, eval_tasks)
    assert clean["n_leaked"] == 0
    leaked_index = dict(index)
    leaked_index[eval_tasks[0].signature] = "LEAK"
    assert contamination.probe(leaked_index, eval_tasks)["n_leaked"] == 1


def test_scoring_catches():
    exp = Verdict(verdict="FLAG", violations=[Violation(
        type=ViolationType.ALLERGEN, severity="CRITICAL", cfr="x",
        observed="o", expected="e", correction="c")])
    pred_ok = Verdict(verdict="FLAG", violations=[Violation(
        type=ViolationType.ALLERGEN, severity="CRITICAL", cfr="x",
        observed="o", expected="e", correction="c")])
    pred_miss = Verdict(verdict="PASS", violations=[])
    r = summarize([score_predictions(exp, pred_ok), score_predictions(exp, pred_miss)])
    assert r["severity_weighted_recall"] == 0.5
    assert r["parse_rate"] == 1.0
    assert class_recall([exp], [pred_ok], ViolationType.ALLERGEN) == 1.0


def test_scoring_parse_miss():
    exp = Verdict(verdict="FLAG", violations=[Violation(
        type=ViolationType.ALLERGEN, severity="CRITICAL", cfr="x",
        observed="o", expected="e", correction="c")])
    r = score_predictions(exp, None)
    assert r["parsed"] == 0.0
    assert r["caught"] == 0.0


def test_generator_deterministic_same_seed():
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    t1 = generate.task(rng1, "pizza", seed=7, n_violations=2)
    t2 = generate.task(rng2, "pizza", seed=7, n_violations=2)
    assert t1.signature == t2.signature
    assert t1.image_bytes_sha256 == t2.image_bytes_sha256
    assert verify(t1.label) == verify(t2.label)


def test_gate_holds_on_pilot():
    rng = random.Random(17)
    for _ in range(60):
        t = generate.task(rng, "yogurt", seed=7, n_violations=2)
        assert oracle_gate(t.label, {v.type for v in t.expected.violations})


def test_all_taxonomy_types_in_injection_pool():
    assert {ViolationType(x) for x in generate.ALL} == set(ViolationType)


def test_allergen_statement_injectable_and_gated():
    rng = random.Random(44)
    label = generate.generate_label(rng, "yogurt", [ViolationType.ALLERGEN_STATEMENT])
    assert oracle_gate(label, {ViolationType.ALLERGEN_STATEMENT})


def test_allergen_and_allergen_statement_never_combined():
    rng = random.Random(21)
    n_two = 0
    banned = {ViolationType.ALLERGEN, ViolationType.ALLERGEN_STATEMENT}
    for _ in range(80):
        t = generate.task(rng, "yogurt", seed=7, n_violations=2)
        types = {v.type for v in t.expected.violations}
        assert not banned <= types
        if len(types) == 2:
            n_two += 1
    assert n_two >= 60


def test_panel_lines_include_health_claim():
    rng = random.Random(31)
    label = generate.generate_label(rng, "yogurt", [ViolationType.HEALTH_CLAIM])
    assert label.printed_health_claims
    lines = generate._panel_lines(label)
    for claim in label.printed_health_claims:
        assert any(claim in line for line in lines)


def test_formatting_omits_gram_measure_from_serving_line():
    rng = random.Random(32)
    label = generate.generate_label(rng, "bread", [ViolationType.FORMATTING])
    serving = next(line for line in generate._panel_lines(label)
                   if line.startswith("Serving Size"))
    assert f"({label.printed.serving_size_grams:g}g)" not in serving
    assert label.printed.serving_size_household in serving


def test_legibility_render_differs_from_clean():
    rng = random.Random(33)
    dirty = generate.generate_label(rng, "cookies", [ViolationType.LEGIBILITY])
    clean = dirty.model_copy(deep=True)
    clean.printed.legibility_flags = []
    assert generate.render_png(clean) != generate.render_png(dirty)


def test_forced_injection_gate_holds_across_categories():
    rng = random.Random(99)
    for cat in ("bread", "yogurt", "soup_ready_to_serve"):
        for name in generate.ALL:
            vt = ViolationType(name)
            label = generate.generate_label(rng, cat, [vt])
            assert oracle_gate(label, {vt}), f"{vt} failed gate on {cat}"


def test_high_difficulty_serving_is_near_racc():
    from specula_forge.verify import RACC
    rng = random.Random(40)
    label = generate.generate_label(
        rng, "bread", [ViolationType.SERVING_SIZE], difficulty=0.9)
    racc = RACC["bread"][1]
    ratio = label.printed.serving_size_grams / racc
    assert 0.9 <= ratio <= 1.15
    assert abs(label.printed.serving_size_grams - racc) > 1e-6


def test_low_difficulty_serving_is_far_from_racc():
    from specula_forge.verify import RACC
    rng = random.Random(41)
    label = generate.generate_label(
        rng, "bread", [ViolationType.SERVING_SIZE], difficulty=0.1)
    racc = RACC["bread"][1]
    ratio = label.printed.serving_size_grams / racc
    assert ratio <= 0.6 or ratio >= 1.25


def test_high_difficulty_claim_is_just_over_limit():
    from specula_forge.verify import CLAIM_MAX, CLAIM_RULES, nutrient_value
    rng = random.Random(42)
    for _ in range(20):
        label = generate.generate_label(
            rng, "cookies", [ViolationType.CLAIM_THRESHOLD], difficulty=0.9)
        if not label.printed_claims:
            continue
        claim = label.printed_claims[0].lower()
        if claim in CLAIM_MAX:
            amount = nutrient_value(label.true.nutrients, CLAIM_RULES[claim])
            limit = CLAIM_MAX[claim]
            assert limit < amount <= limit + 2.0
            return
        from specula_forge.verify import CLAIM_LT
        if claim in CLAIM_LT:
            nutrient, limit = CLAIM_LT[claim]
            amount = nutrient_value(label.true.nutrients, nutrient)
            assert amount == limit
            return
    raise AssertionError("no numeric-threshold claim realized")


def test_render_aug_deterministic_and_changes_pixels_not_signature():
    rng = random.Random(43)
    quiet = generate.generate_label(rng, "bread", [], difficulty=0.1)
    noisy = quiet.model_copy(deep=True)
    noisy.difficulty = 0.9
    a = generate.render_png(noisy)
    b = generate.render_png(noisy)
    assert a == b
    assert generate.render_png(quiet) != a
    assert generate.signature(quiet) == generate.signature(noisy)


def test_openfda_weights_cover_all_classes():
    assert set(generate.OPENFDA_WEIGHTS) == set(generate.ALL)
    assert abs(sum(generate.OPENFDA_WEIGHTS.values()) - 1.0) < 1e-6
