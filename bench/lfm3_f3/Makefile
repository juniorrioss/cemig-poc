# bench/lfm3_f3 — F3: robustez de uso de contexto (dataset ancorado no retriever real)
# Interpretador local: classifier/.venv. Encoder denso: llama-server :8399 (GGUF).

PY := ../../classifier/.venv/bin/python

.PHONY: funnel gates retrieve assign targets render dataset score-all export analyze

# 1a. NRs extras (6,12,17,22,35) pelo funil da F2 (saida em data/synth/)
funnel:
	$(PY) synth/run_funnel_f3.py --workers 16

# gates por estágio (validate_stage da F2; GATE: exit 1)
gates:
	for nr in 6 12 17 22 35; do \
	  python3 ../lfm3_f2/synth/validate_stage.py data/synth/units_nr$$nr.jsonl --kind units && \
	  python3 ../lfm3_f2/synth/validate_stage.py data/synth/qa_passed_nr$$nr.jsonl ; done

# 1b. retriever real v4 sobre o corpus completo (top-8 + rótulos)
retrieve:
	$(PY) retrieve_f3.py --pool f2
	$(PY) retrieve_f3.py --pool f3

# 1c. atribuição de casos/k/posições (determinística)
assign:
	$(PY) build_dataset.py --assign

# 1d. alvos do 27B (gold+partial)
targets:
	$(PY) synth/gen_targets.py --workers 12

render:
	$(PY) build_dataset.py --render
	$(PY) validate_dataset.py

dataset: assign targets render

# 3. julgamento (depois de puxar gen_*.jsonl da Spark p/ data/gen/)
score-all:
	for tag in $(TAGS); do for cell in closed_book oraculo ranking distrator; do \
	  $(PY) eval/score_f3.py --tag $$tag --cell $$cell --workers 16 ; done ; done

export:
	$(PY) eval/export_parquet_f3.py --all

analyze:
	$(PY) analyze_f3.py
