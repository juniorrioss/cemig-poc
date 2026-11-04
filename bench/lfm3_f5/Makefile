# bench/lfm3_f5 — alavancas de retrieval no v4 + compressão de contexto (F5).
# Interpretador local: classifier/.venv (sklearn 1.9.1 + tokenizers). Encoder: llama-server :8399.
# Juiz 27B: REGUA_JUDGE_URL (default http://10.100.0.111:8005/v1).

PY := ../../classifier/.venv/bin/python
TAG ?= f4-r32-ep2
F1 := v4_anchor,v4app_smoke,v4app,noclf,noclf_ovr,thr0.6,thr0.7,thr0.8,thr0.9,f2,f1.5,f1,thr0.7_f2,thr0.8_f1.5,hardoff_f2
F2 := v4app+nb1,thr0.7_f2+nb1
F4 := v4app+man0.5,v4app+man1,v4app+man1.25,v4app+man1.5,v4app+man1.75,v4app+man2,v4app+man3,v4app+man1+intent,v4app+man2+intent,v4app+man1.5+margin,v4app+man2+margin,v4app+man2+margin0.02,v4app+man3+margin0.02,thr0.7_f2+man0.5,thr0.7_f2+man1

.PHONY: align manuals units retrieval proxy e2e-prompts score consolidate

align:
	$(PY) align.py
manuals:
	$(PY) manuals.py
units:
	$(PY) compress.py
retrieval:
	$(PY) eval_retrieval.py --variants $(F1)
	$(PY) eval_retrieval.py --variants $(F2)
	$(PY) eval_retrieval.py --variants $(F4)
proxy:
	$(PY) eval_compress.py --mode hybrid --lam 0.3 --prior 0.05 --tag _h03p05
e2e-prompts:
	$(PY) e2e_build.py
score:
	$(PY) score_e2e.py --tag $(TAG) --workers 12
consolidate:
	$(PY) consolidate.py
