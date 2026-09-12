# Makefile — régua honesta (bench/regua)
# Rode com o juiz vLLM 27B acessível via REGUA_JUDGE_URL.
# Na DGX Spark use o túnel reverso: REGUA_JUDGE_URL=http://127.0.0.1:18005/v1

PY ?= python3
WORKERS ?= 16
THRESHOLD ?= 0.5

.PHONY: all gabarito rejudge analyze test clean

all: gabarito rejudge analyze

# Passo 1 — gabarito de fatos (juiz 27B, paralelo)
gabarito:
	$(PY) build_gabarito.py --workers $(WORKERS)

# Passo 3 — rejulgamento retroativo de todas as respostas salvas
rejudge:
	$(PY) rejudge.py --workers $(WORKERS) --threshold $(THRESHOLD)

# Consolidação e respostas ao capitão
analyze:
	$(PY) analyze.py

# Passo 2 — validação offline do detector de tautologia (não precisa do juiz)
test:
	$(PY) test_tautology.py

clean:
	rm -f data/*.ckpt.json data/*.tmp
