"""Generator, rendering, scoring, and contamination tests."""

import random

from labelforge_forge import contamination, generate
from labelforge_forge.schema import Verdict, Violation, ViolationType
from labelforge_forge.score import class_recall, score_predictions, summarize
from labelforge_forge.verify import oracle_gate, verify


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