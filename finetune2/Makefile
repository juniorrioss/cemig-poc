# Makefile — pipeline finetune2 (treino de síntese do LFM2.5-2.6B: SFT + DPO + painel OOD).
# Interpretadores: gen/juiz em classifier/.venv (sklearn+requests); treino em .venv-train (torch+trl).
# Detalhes e higiene: finetune2/README.md e run_all.sh.

PY_GEN   := ../classifier/.venv/bin/python
PY_TRAIN := ../.venv-train/bin/python

.PHONY: all data sft-data dpo-questions dpo-data ood train eval consolidate clean

all: ## pipeline completo (data -> train -> eval)
	./run_all.sh all

data: ## gera todos os datasets (SFT + DPO questions + DPO pairs + OOD congelado)
	./run_all.sh data

sft-data: ## FASE A: dataset SFT por inversão de chunks (~1500 pares)
	$(PY_GEN) gen_sft.py --n 1500

dpo-questions: ## pool CLEAN de perguntas p/ DPO (~800, NÃO holdout)
	$(PY_GEN) gen_dpo_questions.py --n 800

dpo-data: ## FASE B: pares DPO (2.6B local :8401 + juiz/editor 27B) — exige servidor base no ar
	$(PY_GEN) gen_dpo.py --target 400

ood: ## conjunto FIXO de capacidade geral + comportamental (CONGELADO; não regenera)
	$(PY_GEN) gen_ood.py

train: ## FASE A (SFT) + FASE B (DPO) na 5070 + export GGUF Q4
	HF_HUB_OFFLINE=1 $(PY_TRAIN) train.py --phase both

eval: ## painel 4 camadas base/sft/dpo (sobe servidores GGUF) + consolidação
	./run_all.sh eval

consolidate: ## só a consolidação/veredito de regressão a partir dos panel_*.json
	$(PY_GEN) eval_panel.py --consolidate --variants base,sft,dpo

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n",$$1,$$2}'
