# Baseline F0 — LFM3-1.7B bf16 no benchmark_mobile.
# Geração roda na Spark (ver README); aqui ficam gabarito/score/analyze (locais).

PY := ../../classifier/.venv/bin/python
CELLS := closed_book oraculo ranking distrator

.PHONY: gabarito score analyze all pull

gabarito:
	$(PY) build_gabarito_f0.py --workers 16

pull:
	scp "walcyrios@spark-b431:~/cemig-poc/lfm3_f0/data/gen_*.jsonl" data/

score:
	for c in $(CELLS); do $(PY) score_f0.py --cell $$c --workers 16; done

analyze:
	python3 analyze_f0.py

all: gabarito score analyze
