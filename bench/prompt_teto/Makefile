# Makefile — teste de teto & otimização de prompt (bench/prompt_teto)
# Régua única: bench/regua. Juiz/teto: vLLM 27B (REGUA_JUDGE_URL).
# Síntese 2.6B/1.2B: llama-server na RTX 5070. Tudo paralelo, n=151.

PY        := ../../classifier/.venv/bin/python
JUDGE_URL ?= http://10.100.0.111:8005/v1
LFM26_URL ?= http://127.0.0.1:8410
LFM12_URL ?= http://127.0.0.1:8411
EMB_URL   ?= http://127.0.0.1:8399
VARIANTS  := baseline anti_evasao acao_primeiro numeros combinado_6f combinado_4f

export REGUA_JUDGE_URL=$(JUDGE_URL)

.PHONY: all retrieval part1 part2 oracle app consolidate clean

all: part1 part2 oracle app consolidate

# Reconstrói índice FTS5 v4 + cache denso v4-exp + chunks das 151 (retrieval embarcado).
retrieval:
	$(PY) ../../retrieval4/build_index_v4.py --mode v4 \
	   --out ../../retrieval4/indices/index_hf_36nr_expv4.db
	$(PY) ../../retrieval4/dense_v4.py --which exp \
	   --out ../../retrieval4/results/densev4_exp_rank.json --url $(EMB_URL)
	$(PY) retrieval_teto.py --build --top-k 2

# PARTE 1: 6 variantes de prompt no 2.6B (retrieved).
part1:
	@for v in $(VARIANTS); do \
	  $(PY) harness.py --model lfm2.6b --variant $$v --context retrieved \
	     --url $(LFM26_URL) --out data/resp_lfm2.6b_$${v}_retrieved.json --max-tokens 400 --workers 8 ; \
	  $(PY) score.py --resp data/resp_lfm2.6b_$${v}_retrieved.json \
	     --out data/score_lfm2.6b_$${v}_retrieved.json --workers 16 ; \
	done

# PARTE 2: teto do 27B (baseline/numeros x retrieved/oracle).
part2:
	@for cfg in "baseline retrieved" "numeros retrieved" "baseline oracle" "numeros oracle"; do \
	  set -- $$cfg ; \
	  $(PY) harness.py --model judge27b --variant $$1 --context $$2 --url $(JUDGE_URL) \
	     --out data/resp_judge27b_$${1}_$${2}.json --max-tokens 400 --workers 16 ; \
	  $(PY) score.py --resp data/resp_judge27b_$${1}_$${2}.json \
	     --out data/score_judge27b_$${1}_$${2}.json --workers 16 ; \
	done

# ORÁCULO do 2.6B (separa retrieval de geração).
oracle:
	@for v in baseline numeros; do \
	  $(PY) harness.py --model lfm2.6b --variant $$v --context oracle --url $(LFM26_URL) \
	     --out data/resp_lfm2.6b_$${v}_oracle.json --max-tokens 400 --workers 8 ; \
	  $(PY) score.py --resp data/resp_lfm2.6b_$${v}_oracle.json \
	     --out data/score_lfm2.6b_$${v}_oracle.json --workers 16 ; \
	done

# 1.2B EMBARCADO (estilo atual do app) — precisa do llama-server 1.2B em $(LFM12_URL).
# --label lfm1.2b corrige a metadata (o flag --model é só o caminho de sampling LFM).
app:
	@for v in baseline numeros; do \
	  $(PY) harness.py --model lfm2.6b --label lfm1.2b --variant $$v --context retrieved --url $(LFM12_URL) \
	     --out data/resp_lfm1.2b_$${v}_retrieved.json --max-tokens 400 --workers 8 ; \
	  $(PY) score.py --resp data/resp_lfm1.2b_$${v}_retrieved.json \
	     --out data/score_lfm1.2b_$${v}_retrieved.json --workers 16 ; \
	done

consolidate:
	$(PY) consolidate.py

clean:
	rm -f data/*.ckpt.json data/*.tmp
