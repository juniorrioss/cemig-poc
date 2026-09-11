# Makefile — síntese v2 (shootout de sintetizadores + engenharia da síntese).
# RTX 5070 p/ inferência; juiz vLLM 27B do capitão. Comentários PT-BR; código em inglês.

PY := ../../classifier/.venv/bin/python

.PHONY: all shootout ablation judge analyze dspark clean

all: shootout ablation judge analyze ## pipeline completo (sem dspark, que exige 2 servidores)

shootout: ## 4 candidatos × baseline+full (llama-server CUDA)
	./run_shootout.sh

ablation: ## few-shot / citação / ordenação isoladas no vencedor 2.6B
	./run_shootout.sh --conditions "fewshot citation ordering" lfm2.6b_noth

judge: ## empacota e roda o juiz vLLM 27B (reusa cache)
	$(PY) build_judge_input.py
	$(PY) ../judge.py \
	  --input data/harness_for_judge.json \
	  --output-eval data/judge_evaluations.json \
	  --output-csv data/judge_summary.csv --no-calibration

analyze: ## tabela mestre + eixo-gargalo + histogramas + projeção mobile
	$(PY) analyze.py

dspark: ## speedup de decode do DSpark (rode os 2 servidores antes; ver README)
	$(PY) dspark_bench.py --url-nodraft http://127.0.0.1:8395 --url-draft http://127.0.0.1:8396

clean: ## remove respostas/juízo (mantém sanity)
	rm -f data/responses_*.json data/harness_for_judge.json data/judge_*.json data/analysis.json
