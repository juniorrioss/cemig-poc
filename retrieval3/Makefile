# Retrieval v3 — pipeline mobile-first (POC CEMIG)
# Ambientes: CLF (classifier/.venv, sklearn 1.9.1) p/ busca/fusão;
#            TRAIN (.venv-train, torch cu128) p/ encode denso/rerank.
CLF   := ../classifier/.venv/bin/python
TRAIN := ../.venv-train/bin/python
GEMMA := google/embeddinggemma-300m

.PHONY: all baseline devset expand dense rerank consolidate clean

all: baseline expand dense rerank consolidate

# Etapa 0 — baseline (main atual)
baseline:
	$(CLF) eval_baseline.py --subset qa_v2

# dev-set sintético p/ calibração (nunca holdout) — via vLLM
devset:
	$(CLF) gen_devset.py --n 200 --seed 13 --out data/dev_set.jsonl

# Etapa 1 — expansão de documento (vLLM) + índice + gate
expand:
	$(CLF) expand_docs.py --out data/expansions.jsonl --workers 8
	$(CLF) build_expanded_index.py
	$(CLF) eval_expansion.py

# Etapa 2 — RRF multi-query (diagnóstico)
rrf:
	$(CLF) eval_rrf.py --db indices/index_hf_36nr_exp.db --json-out results/rrf_exp.json

# Etapa 3 — busca densa (build GPU -> dump -> fusão)
dense:
	./run_dense_model.sh gemma768_exp     $(GEMMA) 0 none Retrieval-query Retrieval-document --exp
	./run_dense_model.sh gemma768_exponly $(GEMMA) 0 none Retrieval-query Retrieval-document --exp-only
	$(CLF) eval_fusion_final.py \
	  --dense-text results/dense_rank_gemma768_exp.json \
	  --dense-exp  results/dense_rank_gemma768_exponly.json

# Etapa 4 — rerankers (teto cloud + candidatos mobile)
rerank:
	$(CLF) dump_fusion3.py \
	  --dense-text results/dense_rank_gemma768_exp.json \
	  --dense-exp  results/dense_rank_gemma768_exponly.json \
	  --rrf-k 30 --topn 10 --out results/fusion3_cand.json
	$(CLF) rerank_llm.py --cand results/fusion3_cand.json \
	  --out results/rerank_llm_3way.json --label 27B-3way --workers 8

# Etapa 5 — consolidação final
consolidate:
	$(CLF) consolidate.py

clean:
	rm -f indices/dense_*.db results/dense_rank_*.json results/fusion*_cand*.json
