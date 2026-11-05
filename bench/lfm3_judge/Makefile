# bench/lfm3_judge — decision model (pplx-decider / vela3-flash) × LLM-as-judge (Qwen27b).
# Interpretador local: classifier/.venv (pyarrow, requests). Servidores: serve/decider_server.py
# na H100 (ver README §Reprodução). Juiz Qwen: REGUA_JUDGE_URL (default 10.100.0.111:8005).
PY := ../../classifier/.venv/bin/python
PPLX := http://10.100.0.111:8006
VELA := http://10.100.0.111:8007

.PHONY: sets pplx vela qwen analyze disagreements pack

sets:
	$(PY) build_sets.py --nv 150 --nb 300
pplx:
	$(PY) run_decider.py --model pplx --url $(PPLX) --sets V R B --rep 1
	$(PY) run_decider.py --model pplx --url $(PPLX) --sets V R --rep 2
	$(PY) run_decider.py --model pplx_lean --url $(PPLX) --sets R --rep 1 --lean
vela:
	$(PY) run_decider.py --model vela --backend vela --url $(VELA) --sets V R B --rep 1
	$(PY) run_decider.py --model vela --backend vela --url $(VELA) --sets V --rep 2
	$(PY) run_decider.py --model vela_lean --backend vela --url $(VELA) --sets R --rep 1 --lean
qwen:
	$(PY) run_qwen.py --sets V R B --rep 1 --workers 12
	$(PY) run_qwen.py --sets V R --rep 2 --workers 12
	$(PY) run_qwen.py --sets R --rep 1 --workers 12 --temperature 0.7 --tag qwen_t07
	$(PY) run_qwen.py --sets R --rep 2 --workers 12 --temperature 0.7 --tag qwen_t07
	$(PY) run_qwen.py --sets R --rep 3 --workers 48 --tag qwen_w48
	$(PY) run_qwen.py --sets R --rep 4 --workers 128 --tag qwen_w128
analyze:
	$(PY) analyze.py
disagreements:
	$(PY) disagreements.py --model pplx --set R --n 15
	$(PY) disagreements.py --model pplx --set B --n 8
# versiona os .jsonl crus como .gz (os crus ficam gitignored)
pack:
	for f in data/sets/*.jsonl data/raw/*.jsonl; do gzip -9 -k -f -n $$f; done
