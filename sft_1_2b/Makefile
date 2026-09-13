# Makefile — SFT do LFM2.5-1.2B (aplicar a receita do v3 do 2.6B ao modelo embarcado).
# Juiz vLLM 27B via REGUA_JUDGE_URL (default 10.100.0.111:8005). PY = classifier/.venv.
PY ?= ../classifier/.venv/bin/python

.PHONY: train bases evals panel consolidate all

# PARTE 1 (Spark): treino varredura de rank r16/r32/r64/r128 — rode manualmente na Spark.
train:
	@echo "Rode na Spark: setsid nohup bash ~/cemig-poc/slm_scripts_1_2b/run_train_1_2b.sh > ~/cemig-poc/logs/train_1_2b.log 2>&1 &"

# PARTE 2 (Spark): GGUFs do base NAO-treinado (bf16 + Q4 sem QAD) p/ comparar com o QAD.
bases:
	@echo "Rode na Spark: bash ~/cemig-poc/slm_scripts_1_2b/make_base_ggufs.sh"

# PARTE 2 (local): avalia bases + varredura em bf16 e Q4 (4 suites cada).
evals:
	./run_all_evals_1_2b.sh

# PARTE 2 (local): painel geral+comportamental do modelo escolhido vs base (forgetting).
#   reusa o run_general_panel do v3 (aponta o gguf 1.2B escolhido).
panel:
	./run_general_panel_1_2b.sh

consolidate:
	$(PY) consolidate_1_2b.py

all: evals consolidate
