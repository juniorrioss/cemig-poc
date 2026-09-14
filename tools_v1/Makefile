# tools_v1 — bancada de TOOL-CALLING nativo do LFM2.5 (1.2B primeiro).
# Ambientes: geração/validação/eval no classifier/.venv (sklearn+jinja); render de
# treino/paridade no .venv-train (transformers). Treino na DGX Spark.
PY=../classifier/.venv/bin/python
PYT=../.venv-train/bin/python

.PHONY: verify data prep truncation evalsets clean

## PASSO 0 — render idêntico ao contrato + máscara assistant-only (roda no .venv-train)
verify:
	$(PYT) verify_format.py | tee data/passo0_render.txt
	$(PY) test_render_parity.py --emit jinja > /tmp/j.txt
	$(PYT) test_render_parity.py --emit hf > /tmp/h.txt
	diff /tmp/j.txt /tmp/h.txt && echo "PARIDADE render byte-a-byte OK"

## PASSO 1+2 — gera os 4500 diálogos (5 famílias) com validação mecânica
data:
	$(PY) gen_dialogs.py --total 4500 --workers 24 \
	  --out data/dialogs.jsonl --discards data/lexical_discards.json

## prep de treino — tools no system + degraus aninhados da curva de saturação
prep:
	$(PY) prep_train.py --in data/dialogs.jsonl --steps 750,1500,3000,4500

## mede truncamento a max_length (roda no .venv-train)
truncation:
	$(PYT) measure_truncation.py --data data/train_full.jsonl --max-length 2048
	$(PYT) measure_truncation.py --data data/train_full.jsonl --max-length 3072

## PASSO 4 — monta as suites de avaliação (fora do treino + holdout só p/ medir)
evalsets:
	$(PY) build_eval_sets.py

clean:
	rm -f data/eval_*.json data/e2e_*.json
