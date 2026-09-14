# tools_2_6b — repetir no 2.6B a bateria de tool-calling do 1.2B (comparabilidade maçã-com-maçã).
# Reusa: dataset tools_v1 (MESMOS 4.236 diálogos), motor sft_v2/train_v2, régua bench/regua,
# harness tools_v1/tools_oraculo (via patch do renderer 2.6B thinking-OFF). Juiz vLLM 27B.
PY=../classifier/.venv/bin/python
SPARK=walcyrios@spark-b431
JUDGE?=http://10.100.0.111:8005/v1
LABEL?=tools_2_6b_r64
QUANT?=q4

.PHONY: help verify passo0 truncation train evals score consolidate all clean

help:
	@echo "verify (PASSO 0 render, local) | passo0 (módulos+trunc na Spark) | train (Spark, detached)"
	@echo "evals GGUF=<spark> (bateria) | score (régua) | consolidate | all"

## PASSO 0 — render byte-identity + round-trip + 1 exemplo/família (local, classifier/.venv)
verify:
	$(PY) verify_2_6b.py

## PASSO 0 — confirmação de módulos (Spark) + truncamento com tokenizer do 2.6B (Spark)
passo0:
	bash sync_to_spark.sh
	ssh $(SPARK) "cd ~/cemig-poc && ~/jupyterlab/.venv/bin/python tools_2_6b/verify_modules_2_6b.py --base ~/cemig-poc/base_hf --lora-r 64 --out ~/cemig-poc/tools_2_6b/data/passo0_modules.json"
	ssh $(SPARK) "cd ~/cemig-poc && ~/jupyterlab/.venv/bin/python tools_2_6b/measure_truncation_2_6b.py --data ~/cemig-poc/tools_v1/data/train_full.jsonl --base ~/cemig-poc/base_hf --max-length 2048 3072 --out ~/cemig-poc/tools_2_6b/data/truncation.json"
	scp $(SPARK):~/cemig-poc/tools_2_6b/data/{passo0_modules.json,truncation.json} data/

## PASSO 1 — treino (curva r32 + varredura r32/64/128), 1 época, detached na Spark
train:
	bash sync_to_spark.sh
	ssh $(SPARK) "cd ~/cemig-poc && WANDB_MODE=online setsid nohup bash tools_2_6b/run_train_tools_2_6b.sh > logs/train_tools_2_6b.log 2>&1 < /dev/null & echo LANCADO"

## PASSO 2 — bateria completa de um candidato (defina GGUF=<caminho na Spark>, QUANT, RUNS)
evals:
	bash run_all_evals_2_6b.sh $(GGUF) $(LABEL) $(QUANT) 8500 3

score:
	REGUA_JUDGE_URL=$(JUDGE) bash score_gens_2_6b.sh $(LABEL) $(QUANT) 3

consolidate:
	$(PY) consolidate_2_6b.py --label $(LABEL)
	$(PY) extract_samples_2_6b.py || true

all: verify passo0 train
	@echo "treino detached; rode 'make evals GGUF=...' quando os GGUF estiverem prontos, depois score+consolidate."

clean:
	rm -f data/gen_*.json data/score_*.json
