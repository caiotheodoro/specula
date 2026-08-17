PY ?= uv run

.PHONY: sync validate study bench serve

sync:
	cd forge && uv sync --no-editable && chflags -R nohidden .venv
	cd model && uv sync --no-editable && chflags -R nohidden .venv

validate:
	cd forge && uv run pytest -q
	cd model && uv run pytest -q

study:
	cd forge && $(PY) python -m specula_forge.cli pilot --seed 7 --n 400

bench:
	cd model && $(PY) python -m specula_model.benchmark_eval --adapter-path adapters/champion --tasks-file data/benchmark.jsonl

serve:
	cd model && $(PY) python -m specula_model.serve --adapter-path adapters/champion