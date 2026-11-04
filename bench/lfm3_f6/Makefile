# bench/lfm3_f6 — manuais nos índices: ponto de operação + fim-a-fim (F6).
# Interpretador local: classifier/.venv (sklearn 1.9.1 + tokenizers). Encoder: llama-server :8399.
# Juiz 27B: REGUA_JUDGE_URL (default http://10.100.0.111:8005/v1).
# Pré-requisito: caches da F5 (make -C ../lfm3_f5 align manuals units).

PY := ../../classifier/.venv/bin/python
# Grade do Passo 1 (gramática da F5: base[+man{w}][+margin{m}])
GRID := v4app,v4app+man1.25,v4app+man1.3,v4app+man1.35,v4app+man1.4,v4app+man1.5,v4app+man1.75,v4app+man2,v4app+man1.4+margin,v4app+man1.5+margin,v4app+man1.75+margin,v4app+man2+margin,thr0.7_f2,thr0.7_f2+man1.25,thr0.7_f2+man1.3,thr0.7_f2+man1.4,thr0.7_f2+man1.45,thr0.7_f2+man1.5,thr0.7_f2+man1.6,thr0.7_f2+man1.7,thr0.7_f2+man1.8,thr0.7_f2+man2,thr0.7_f2+man1.75+margin,thr0.7_f2+man1.8+margin,thr0.7_f2+man1.9+margin,thr0.7_f2+man2+margin

.PHONY: retrieval units prompts score consolidate export

retrieval:
	$(PY) eval_retrieval_f6.py --variants $(GRID)
units:
	$(PY) compress_f6.py
prompts:
	$(PY) e2e_build_f6.py --parity --which recorte,151
# Geração na Spark (setsid nohup, batch 8 — ver RESULTS.md §Reprodução), depois:
score:
	$(PY) score_f6.py --suite recorte --tag f6-recorte
	$(PY) score_f6.py --suite recorte --tag f6-recorte-rep2
	$(PY) score_f6.py --suite recorte --tag f6-recorte-rep3
	$(PY) score_f6.py --suite 151 --tag f6-151
	$(PY) score_f6.py --suite 151 --tag f6-151-rep2
consolidate:
	$(PY) consolidate_f6.py
export:
	$(PY) export_parquet_f6.py
