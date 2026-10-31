# bench/lfm3_f3b — sweep one-factor-at-a-time sobre a receita F3.
# Interpretador local: classifier/.venv. Juiz 27B: 10.100.0.111:8005.

PY := ../../classifier/.venv/bin/python

.PHONY: general variants funnel-2x retrieve-2x targets-2x score-all probe-judge analyze

# eixo 6: pares gerais PT-BR (27B, anticontaminação vs probe congelado)
general:
	$(PY) gen_general.py --n 1600 --workers 16

# eixos 1 e 6: datasets variantes a partir do F3 congelado
variants:
	$(PY) build_variants.py --variant recusa35
	$(PY) build_variants.py --variant anchor25

# eixo 2: funil NRs novas (27B) + retriever real + assign/targets/render
funnel-2x:
	$(PY) synth/run_funnel_f3b.py --workers 12

retrieve-2x:
	$(PY) ../lfm3_f3/retrieve_f3.py --pool f3b \
	  --out data/retrieval/topk_f3b.jsonl

targets-2x:
	$(PY) gen_targets_2x.py --workers 12

# julgamento das células do sweep (depois de puxar gen_*.jsonl p/ data/gen/)
score-all:
	for tag in $(TAGS); do for cell in ranking distrator; do \
	  $(PY) eval/score_f3b.py --tag $$tag --cell $$cell --workers 16 ; done ; done

# probe de capacidade (judge local; gen roda na Spark/H100)
probe-judge:
	$(PY) ../lfm3_f2/probe/run_probe.py judge --tag $(TAG)

analyze:
	$(PY) analyze_f3b.py
