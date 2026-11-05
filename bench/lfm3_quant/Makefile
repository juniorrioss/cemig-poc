# bench/lfm3_quant — retenção de qualidade por formato de quantização (f4-r32-ep2, recorte 344 op_d50).
# Interpretador local: classifier/.venv (requests + pyarrow). Juiz 27B: REGUA_JUDGE_URL
# (default http://10.100.0.111:8005/v1). Geração na Spark (ver RESULTS.md §Reprodução).

PY := ../../classifier/.venv/bin/python
TAGS := gguf_bf16-r1 gguf_bf16-r2 gguf_q8_0-r1 gguf_q8_0-r2 gguf_q4_0-r1 gguf_q4_0-r2 gguf_q4_k_m-r1 gguf_q4_k_m-r2 \
        gguf_q4_0_rep1.0-r1 litert_int4_gpu-r1 litert_int4_gpu-r2 litert_int4_cpu-r1 \
        litert_int4_cpu-r2 litert_int8_gpu-r1 litert_int8_gpu-r2 litert_int8_cpu-r1 litert_int8_cpu-r2

.PHONY: prompts score consolidate export

prompts:  # célula op_d50 da F6, congelada
	$(PY) -c "import gzip,json; P=[json.loads(l) for l in gzip.open('../lfm3_f6/data/e2e/prompts_recorte.jsonl.gz')]; open('data/prompts_op344.jsonl','w').writelines(json.dumps(p,ensure_ascii=False)+'\n' for p in P if p['cell']=='op_d50')"
score:
	$(PY) score_quant.py --tag $(TAGS)
consolidate:
	$(PY) consolidate_quant.py
export:
	$(PY) export_parquet_quant.py
