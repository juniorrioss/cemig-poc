# Makefile — pipeline do classificador de NR (POC CEMIG, Opção B).
# Interpretador: venv dedicado classifier/.venv (uv, py3.12, scikit-learn 1.9).

PY := .venv/bin/python
VLLM := http://10.100.0.111:8005/v1
LLAMA := http://127.0.0.1:8090/v1

.PHONY: all labels classic prompt compare hybrid export test-parity clean help

help:
	@echo "Alvos: labels classic prompt compare hybrid export all"
	@echo "  labels   FASE A: gera dataset rotulado via vLLM (~2000 falas)"
	@echo "  classic  FASE B: shootout de classificadores clássicos (sklearn)"
	@echo "  prompt   FASE B: prompt-classify no LFM2.5 (requer llama-server em $(LLAMA))"
	@echo "  compare  FASE B: tabela comparativa única + escolha do vencedor"
	@echo "  hybrid   FASE C: híbrido 2 estágios + calibração de boost nas 151 reais"
	@echo "  export   FASE C: exporta o vencedor p/ Kotlin puro (nr_classifier.bin)"

# FASE A — dataset rotulado (idempotente por dedupe + checkpoint incremental)
labels:
	$(PY) gen_labels.py --target 2000 --workers 8 --base-url $(VLLM) --out data/labels.jsonl

# FASE B — clássicos (pula gradientboost, já desqualificado por top1=28.7% e fit 587s)
classic:
	$(PY) train_classic.py --skip-slow

# FASE B — prompt-classify (ceticismo do capitão sobre 'só o número')
prompt:
	$(PY) prompt_classify.py --base-url $(LLAMA)

# FASE B — tabela comparativa única
compare:
	$(PY) compare.py

# FASE C — híbrido 2 estágios (métrica oficial: 151 reais de qa_pairs_v2.jsonl)
hybrid:
	$(PY) hybrid.py --subset qa_v2

# FASE C — export Kotlin puro (paridade validada vs sklearn)
export:
	$(PY) export_kotlin.py
	cp models/nr_classifier.bin ../android/app/src/main/assets/nr_classifier.bin
	cp models/nr_classifier.bin ../android/app/src/test/resources/nr_classifier.bin

all: classic compare hybrid export

clean:
	rm -f data/classic_results.json data/prompt_results.json data/hybrid_results.json \
		data/compare_table.json data/kotlin_export.json models/*.pkl models/*.bin models/*.json
