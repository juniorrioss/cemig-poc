# Makefile — reprocessamento do corpus com Document AI Form Parser (docai/)
# Etapas 1-2 rodam na DGX Spark (batch DocAI); 3-6 rodam localmente.
# Interpretador local: classifier/.venv (tem requests; FTS5 disponível no sqlite do sistema).

REPO_ROOT := $(shell cd .. && pwd)
PY ?= $(REPO_ROOT)/classifier/.venv/bin/python
export PYTHONPATH := $(REPO_ROOT):$(PYTHONPATH)

PARSED ?= docai/data_all
ALL_NRS := nr-03,nr-04,nr-05,nr-07,nr-09,nr-10,nr-12,nr-13,nr-15,nr-16,nr-18,nr-19,nr-20,nr-22,nr-28,nr-29,nr-30,nr-31,nr-32,nr-33,nr-34,nr-36,nr-37

.PHONY: help verify compare compare-all replace ceiling e2e results

help:
	@echo "Alvos (rode a partir da raiz do repo com: make -C docai <alvo>):"
	@echo "  verify      - verificacao celula-a-celula (etapa 5)"
	@echo "  compare     - reindex aditivo 5-piloto + recall nas 151 (etapa 6)"
	@echo "  compare-all - reindex aditivo 23 NRs + recall nas 151"
	@echo "  ceiling     - REPLACE cirurgico + teto do tokenizer (caso 13,8 kV)"
	@echo "  e2e         - aprovacao honesta E2E NR-10 (precisa llama-server :8080)"
	@echo "  NOTA        - etapas 1-2 (batch DocAI) rodam na Spark, ver README.md"

verify:
	cd $(REPO_ROOT) && $(PY) docai/03_verify_cells.py --parsed-dir $(PARSED) \
	  --out docai/data/verify_report_all.json --nrs $(ALL_NRS)

compare:
	cd $(REPO_ROOT) && $(PY) docai/04_reindex_compare.py --parsed-dir docai/data \
	  --new-db docai/data/index_docai.db --nrs nr-10,nr-15,nr-28,nr-04,nr-32 \
	  --out docai/data/compare_report.json

compare-all:
	cd $(REPO_ROOT) && $(PY) docai/04_reindex_compare.py --parsed-dir $(PARSED) \
	  --new-db docai/data/index_docai_all.db --nrs $(ALL_NRS) \
	  --out docai/data/compare_report_all.json

ceiling:
	cd $(REPO_ROOT) && $(PY) docai/06_replace_ceiling.py

e2e:
	cd $(REPO_ROOT) && $(PY) docai/05_e2e_approval.py

results: verify compare compare-all ceiling
	@echo "Resultados em docai/results/*.json"
