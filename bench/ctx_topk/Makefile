# Makefile — bancada JANELA×TRECHOS + painel de eixos da v4 (task poc-ctx-topk).
# Ambiente: classifier/.venv (sklearn 1.9.1). Servidores GGUF na RTX 5070 (ver README).
# Comentários PT-BR.

PY := ../../classifier/.venv/bin/python
SERIAL ?= 192.168.0.6:41073

.PHONY: all grid judge analyze device consolidate clean

all: grid judge analyze device consolidate

# PART 1 — grade de qualidade (8 células: topK 2..5 × full/trim)
grid:
	./run_grid.sh

judge:
	$(PY) build_judge_input.py
	$(PY) ../judge.py --input data/harness_for_judge.json \
	  --output-eval data/judge_evaluations.json --output-csv data/judge_summary.csv --no-calibration

analyze:
	$(PY) analyze.py

# PART 2 — custo real no S24+ (requer ADB no PATH e aparelho liberado)
device:
	$(PY) gen_device_prompts.py
	PATH=$$HOME/android-sdk/platform-tools:$$PATH $(PY) device_bench.py --serial $(SERIAL) --all

# PART 3 — tabela mestre qualidade × custo + recomendação
consolidate:
	$(PY) consolidate.py

clean:
	rm -f data/responses_*.json data/harness_for_judge.json data/judge_*.json data/judge_summary.csv
	rm -rf data/device_prompts
