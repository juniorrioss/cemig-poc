# Retrieval v4 — classificador reforçado + expansão rica, com travas anti-mascaramento.
# Ambiente único: classifier/.venv (sklearn 1.9.1). Encoder denso via llama-server GGUF (:8399).
PY    := ../classifier/.venv/bin/python
EMB   := http://127.0.0.1:8399

.PHONY: all frentea frenteb asr embark clean help

help:
	@echo "Alvos: testsets frentea expand frenteb asr embark all"
	@echo "  testsets  trava 3: gera samegen (Qwen 27B) + diffgen (Gemma-3-1B)"
	@echo "  frentea   Frente A: 3 medições (otimista/honesta/decisiva) + gate sweep"
	@echo "  expand    Frente B: expansão v4 (vLLM 27B) + índice FTS5 v4"
	@echo "  frenteb   Frente B: dense v4 + componentes na fusão + calibração"
	@echo "  asr       trava 4: TTS 151 -> Whisper Base no S24+ -> recall fiel"
	@echo "  embark    exporta bins DVEC1 + paridade (requer :8399)"

testsets:
	$(PY) gen_testsets.py --which samegen --per-nr 20 --workers 8
	$(PY) gen_testsets.py --which diffgen --per-nr 20 --workers 4

frentea:
	$(PY) sweep_frentea.py
	$(PY) sweep_gate.py

expand:
	$(PY) expand_v4.py --workers 10
	$(PY) build_index_v4.py --mode v4

frenteb:
	$(PY) dense_v4.py --which text --out results/densev4_text_rank.json
	$(PY) dense_v4.py --which exp  --out results/densev4_exp_rank.json
	$(PY) eval_expansion_v4.py
	$(PY) sweep_frenteb.py
	$(PY) calibrate_v4.py

asr:
	$(PY) asr_loop.py --all
	$(PY) eval_asr_fusion_true.py

embark:
	$(PY) export_bins_v4.py --which text --out export/dense_text.bin
	$(PY) export_bins_v4.py --which exp  --out export/dense_exponly.bin
	$(PY) verify_bins.py

all: frentea frenteb asr embark

clean:
	rm -f results/densev4_dev_*_rank.json
