# tools_oraculo — oráculo + recusa + híbrido + origem da alucinação do tool-calling do 1.2B.
# Reusa checkpoints do tools_v1 (NÃO retreina), régua bench/regua, índice v4, pack de recusa.
# Ambiente: classifier/.venv (jinja2, requests, sklearn). LLM na DGX Spark (llama-server).
# Juiz 27B via REGUA_JUDGE_URL (default http://10.100.0.111:8005/v1, roteável deste host).
PY=../classifier/.venv/bin/python
SPARK=walcyrios@spark-b431
URL?=http://100.79.169.101:8500
CKPT?=tools_1_2b_r128
JUDGE?=http://10.100.0.111:8005/v1

.PHONY: help serve-q4 serve-bf16 oracle refusal hybrid pure score refusal-metrics analyze samples all clean

help:
	@echo "Alvos: serve-q4|serve-bf16 (sobe llama-server na Spark), oracle|refusal|hybrid|pure (geração),"
	@echo "       score (régua), refusal-metrics, analyze, samples, all. Ver README para a ordem."

## sobe o checkpoint na Spark — CRÍTICO: --parallel 4 (4096 tok/slot; oráculo estoura 2048)
serve-q4:
	ssh $(SPARK) "bash ~/cemig-poc/scripts/start_srv_net.sh ~/cemig-poc/models/lfm2.5-1.2b-$(CKPT)-Q4_0.gguf 8500 16384 4"
serve-bf16:
	ssh $(SPARK) "bash ~/cemig-poc/scripts/start_srv_net.sh ~/cemig-poc/models/lfm2.5-1.2b-$(CKPT)-bf16.gguf 8500 16384 4"

## geração (defina QUANT=q4|bf16 e RUN=1..3). oracle/hybrid/pure = 151; refusal = pack sft_v2.
QUANT?=q4
RUN?=1
oracle:
	$(PY) harness_oraculo.py --url $(URL) --label $(CKPT)_oracle_$(QUANT)_run$(RUN) --mode oracle \
	  --pack 151 --out data/gen_$(CKPT)_oracle_$(QUANT)_run$(RUN).json --workers 6
hybrid:
	$(PY) harness_oraculo.py --url $(URL) --label $(CKPT)_hibrido_$(QUANT)_run$(RUN) --mode hibrido \
	  --pack 151 --out data/gen_$(CKPT)_hibrido_$(QUANT)_run$(RUN).json --workers 6
pure:
	$(PY) harness_oraculo.py --url $(URL) --label $(CKPT)_pure_$(QUANT)_run$(RUN) --mode pure \
	  --pack 151 --out data/gen_$(CKPT)_pure_$(QUANT)_run$(RUN).json --workers 6
refusal:
	$(PY) harness_oraculo.py --url $(URL) --label $(CKPT)_refusal_$(QUANT) --mode oracle \
	  --pack refusal --out data/gen_$(CKPT)_refusal_$(QUANT).json --workers 6

## score: pontua um arquivo de geração pela régua honesta (reuso sft_v2/score_v2.py). GEN=<arquivo>.
GEN?=data/gen_$(CKPT)_oracle_$(QUANT)_run$(RUN).json
score:
	cd ../sft_v2 && REGUA_JUDGE_URL=$(JUDGE) $(PY) score_v2.py \
	  --resp ../tools_oraculo/$(GEN) --out ../tools_oraculo/$(subst gen_,score_,$(GEN)) --workers 8

refusal-metrics:
	$(PY) ../sft_v3/refusal_metrics.py \
	  --scores data/score_$(CKPT)_refusal_bf16.json:$(CKPT)_bf16 --out data/refusal_confusion_bf16.json
	$(PY) ../sft_v3/refusal_metrics.py \
	  --scores data/score_$(CKPT)_refusal_q4.json:$(CKPT)_q4 --out data/refusal_confusion_q4.json

analyze:
	$(PY) analyze.py
samples:
	$(PY) extract_samples.py

## consolida tudo (assume gerações+scores já produzidos; ver README p/ a bateria completa).
all: refusal-metrics analyze samples

clean:
	rm -f data/gen_*.json data/score_*.json
