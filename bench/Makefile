# ==============================================================================
# bench/Makefile — Automação de Benchmark de SLMs Tier 1 (CEMIG POC)
# ==============================================================================

SHELL := /bin/bash
VENV_PYTHON := $(HOME)/.venvs/cemig-bench/bin/python
MODELS_DIR ?= $(HOME)/.models-poc
LLAMA_DIR ?= $(HOME)/llama.cpp

.PHONY: all setup perf bench judge clean

all: setup perf bench judge

setup:
	@echo "=== [1/4] Executando Setup do Ambiente e Modelos ==="
	./bench/setup.sh

perf:
	@echo "=== [2/4] Executando Benchmark de Performance Desktop (llama-bench) ==="
	$(VENV_PYTHON) bench/perf.py

bench:
	@echo "=== [3/4] Executando Harness de Avaliação (Modos Clássico e Tool-Calling) ==="
	$(VENV_PYTHON) bench/harness.py --models all --mode both

judge:
	@echo "=== [4/4] Executando Avaliação por LLM Juiz e Gerando Planilha Consolidada ==="
	$(VENV_PYTHON) bench/judge.py --input bench/results/harness_results.json --cli-tool claude

clean:
	@echo "Limpando resultados temporários..."
	rm -rf bench/results/*.tmp /tmp/test_harness*.json
