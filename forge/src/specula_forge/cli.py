"""CLI: pilot benchmark generation, split build, contamination probe."""

from __future__ import annotations

import json
import os
import random

import click

from . import contamination
from .generate import task
from .schema import Task, ViolationType


def _dump(tasks: list[Task], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(t.model_dump_json() + "\n")


def _load(path: str) -> list[Task]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Task.model_validate_json(line))
    return out


@click.group()
def cli():
    pass


@cli.command()
@click.option("--seed", default=7)
@click.option("--n", default=400)
@click.option("--out", default="data/pilot.jsonl")
@click.option("--mix", default="uniform", type=click.Choice(["uniform", "openfda"]))
def pilot(seed: int, n: int, out: str, mix: str) -> None:
    """Generate a pilot benchmark (all categories, mixed violation counts)."""
    from .verify import RACC
    rng = random.Random(seed)
    categories = list(RACC.keys())
    tasks: list[Task] = []
    mix_arg = None if mix == "uniform" else mix
    for i in range(n):
        cat = categories[i % len(categories)]
        n_viol = rng.choice([0, 1, 1, 2])
        diff = round(rng.random(), 2)
        tasks.append(task(rng, cat, seed, n_violations=n_viol,
                          difficulty=diff, mix=mix_arg))
    _dump(tasks, out)
    n_flag = sum(1 for t in tasks if t.expected.verdict == "FLAG")
    click.echo(f"wrote {n} tasks to {out} ({n_flag} FLAG)")


@cli.command()
@click.option("--pilot", default="data/pilot.jsonl")
@click.option("--out-train", default="data/train.jsonl")
@click.option("--out-val", default="data/val.jsonl")
def split(pilot: str, out_train: str, out_val: str) -> None:
    """Stratified train/val split on (difficulty decile x violation class)."""
    tasks = _load(pilot)
    bins: dict[tuple, list[Task]] = {}
    for t in tasks:
        decile = min(9, int(t.label.difficulty * 10))
        cls = "PASS" if t.expected.verdict == "PASS" else \
            ",".join(sorted({v.type.value for v in t.expected.violations}))
        bins.setdefault((decile, cls), []).append(t)
    train, val = [], []
    rng = random.Random(7)
    for key, group in bins.items():
        group = sorted(group, key=lambda t: t.task_id)
        rng.shuffle(group)
        n_val = max(1, round(len(group) * 0.2))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    _dump(sorted(train, key=lambda t: t.task_id), out_train)
    _dump(sorted(val, key=lambda t: t.task_id), out_val)
    overlap = contamination.split_overlap(train, val)
    click.echo(f"train={len(train)} val={len(val)} overlap={overlap}")
    if overlap > 0:
        raise SystemExit("split overlap > 0: refusing to write")


@cli.command()
@click.option("--train", default="data/train.jsonl")
@click.option("--eval-file", default="data/benchmark.jsonl")
def leakprobe(train: str, eval_file: str) -> None:
    """Verify the contamination monitor fires on intentional leaks."""
    train_tasks = _load(train)
    eval_tasks = _load(eval_file)
    index = contamination.build_train_index(train_tasks)
    clean = contamination.probe(index, eval_tasks)
    leaked_index = dict(index)
    for i in range(0, min(10, len(eval_tasks))):
        leaked_index[eval_tasks[i].signature] = "LEAK"
    leaked = contamination.probe(leaked_index, eval_tasks)
    click.echo(f"clean: fired={clean['n_leaked']}/{clean['n_eval']}")
    click.echo(f"leaked(10): fired={leaked['n_leaked']}/{leaked['n_eval']}")
    if clean["n_leaked"] != 0 or leaked["n_leaked"] < 10:
        raise SystemExit("contamination monitor failed the ROC check")


if __name__ == "__main__":
    cli()