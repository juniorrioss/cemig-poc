# bench/lfm3_f4 — tool-calling multiturno sobre o campeão data2x-ep4.
# Interpretador local: classifier/.venv (geração/scoring) e .venv-f4
# (torch+transformers 5.x: merge, harness local na 5070).
# Juiz/gerador 27B: 10.100.0.111:8005. Encoder denso: llama-server :8399.

PY := ../../classifier/.venv/bin/python
PYT := ../../.venv-f4/bin/python
TAG ?= f4-r32-ep2

.PHONY: verify merge assemble gen27b build validate harness score-tools parquet

# PASSO 0: formato nativo + paridade baked + máscara + round-trip
verify:
	$(PY) verify_format.py

# merge do LoRA campeão F3b no instruct (publica md5)
merge:
	$(PYT) merge_base.py --adapter /tmp/f4_base/checkpoint-1396 \
	  --out /tmp/f4_base/lfm3-1.7b-data2x-ep4-merged

# dataset em 3 estágios
assemble:
	$(PY) gen_dialogs_f4.py --stage assemble
gen27b:
	$(PY) gen_dialogs_f4.py --stage gen27b --workers 24
build:
	$(PY) gen_dialogs_f4.py --stage build

validate:
	$(PY) validate_dataset_f4.py

# harness de diálogo (simulador de tool = retriever v4 REAL)
harness:
	$(PYT) eval/run_dialog_harness.py \
	  --model /tmp/f4_base/lfm3-1.7b-data2x-ep4-merged \
	  --adapter $(ADAPTER) --tag $(TAG)

score-tools:
	$(PY) eval/score_tools.py --tag $(TAG)

parquet:
	$(PY) eval/export_parquet_f4.py --tools $(TAG)
