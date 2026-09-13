# Makefile — SFT v3 (recusa <= 25% + r64/r128 + calibração + embarque S24+).
# Juiz vLLM 27B via REGUA_JUDGE_URL (default 10.100.0.111:8005). PY = classifier/.venv.
PY ?= ../classifier/.venv/bin/python

.PHONY: data train evals quant panel consolidate all

# PARTE 1 (local): rebalanceia + completa oráculo + corpus de calibração.
data:
	$(PY) rebalance_data.py --target-oracle 2300 --target-recusa 300 --target-parcial 300 --target-distrator 150
	$(PY) topup_oracle.py --need 700
	$(PY) build_calibration.py --n 400

# PARTE 2 (Spark): treino r64/r128 — rode manualmente na Spark (ver README).
train:
	@echo "Rode na Spark: setsid nohup bash ~/cemig-poc/slm_scripts_v3/run_train_v3.sh > ~/cemig-poc/logs/train_v3.log 2>&1 &"

# PARTE 3-4 (local): avalia variantes + quantizações.
evals:
	./run_all_evals_v3.sh

quant:
	./run_quant_evals.sh

# PARTE 5 (local): painel geral+comportamental do modelo escolhido vs base.
panel:
	./run_general_panel.sh /home/walcyrios/cemig-poc/models/lfm2.5-2.6b-sft_v3_r64-Q4_0.gguf v3_r64_q4

consolidate:
	$(PY) consolidate_v3.py

all: evals quant consolidate
