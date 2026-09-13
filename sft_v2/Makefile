# Makefile — SFT v2 (4 familias + varredura de rank). Comentarios PT-BR; codigo em ingles.
#
# Geracao de dados e pontuacao rodam LOCAIS (classifier/.venv; juiz 27B roteavel).
# O TREINO e o export GGUF rodam na DGX Spark (ver README.md: run_train_v2.sh).
#
# Juiz via REGUA_JUDGE_URL (default 10.100.0.111:8005).

PY := ../classifier/.venv/bin/python
JUDGE ?= http://10.100.0.111:8005/v1

.PHONY: data eval-packs packs151 consolidate clean-smoke

# PARTE 1 — fabrica os 4 familias (o coracao do v2). ~3200 pares, filtrados por regua.
data:
	$(PY) gen_data.py --target-oracle 1600 --target-recusa 640 \
	  --target-parcial 640 --target-distrator 320 --workers 24 \
	  --out data/train_v2.jsonl --discards data/train_lexical_discards.json

# Packs de VALIDACAO (~150) e de TESTE DE RECUSA (fora do treino/holdout).
eval-packs:
	$(PY) build_eval_packs.py --workers 16

# Copia os packs das 151 (oraculo + retrieved v4) do v1 (auto-contido).
packs151:
	cp ../slm_oraculo/data/oracle_pack.json    data/oracle_pack_151.json
	cp ../slm_oraculo/data/retrieved_pack.json data/retrieved_pack_151.json

# PARTE 3 — tabela mestre + veredito (apos rodar eval_checkpoint_v2.sh por variante).
consolidate:
	$(PY) consolidate_v2.py

clean-smoke:
	rm -f data/_smoke* data/_tmp*
