# Makefile do pipeline de corpus — CEMIG POC Assistente por Voz Offline
# Executa extração de PDFs, chunking com cabeçalhos contextuais,
# criação do índice SQLite FTS5 (index.db) e avaliação de retrieval BM25.

REPO_ROOT := $(shell cd .. && pwd)
export PYTHONPATH := $(REPO_ROOT):$(PYTHONPATH)
PYTHON ?= $(shell if [ -f $(REPO_ROOT)/.venv/bin/python ]; then echo $(REPO_ROOT)/.venv/bin/python; elif [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

DATA_DIR ?= /home/rios/projetos/cemig-mobile-llm/firstmate/data/NRs
EXTRACT_DIR ?= $(REPO_ROOT)/corpus/data/extracted
CHUNKS_FILE ?= $(REPO_ROOT)/corpus/data/chunks.json
INDEX_DB ?= $(REPO_ROOT)/corpus/index.db
QA_FILE ?= $(REPO_ROOT)/corpus/qa_pairs.jsonl
NRS ?= nr-10,nr-06,nr-35,nr-12,nr-18

PARQUET_FILE ?= /home/rios/projetos/cemig-mobile-llm/firstmate/data/nrs_hf/nrs.parquet
EXTRACT_HF_DIR ?= $(REPO_ROOT)/corpus/data/extracted_hf
CHUNKS_5NR ?= $(REPO_ROOT)/corpus/data/chunks_hf_5nr.json
CHUNKS_5NR_MANUAL ?= $(REPO_ROOT)/corpus/data/chunks_hf_5nr_manual.json
CHUNKS_36NR ?= $(REPO_ROOT)/corpus/data/chunks_hf_36nr.json
INDEX_5NR ?= $(REPO_ROOT)/corpus/index_hf_5nr.db
INDEX_5NR_MANUAL ?= $(REPO_ROOT)/corpus/index_hf_5nr_manual.db
INDEX_36NR ?= $(REPO_ROOT)/corpus/index_hf_36nr.db
QA_V2_FILE ?= $(REPO_ROOT)/corpus/qa_pairs_v2.jsonl
NRS_36 := nr-01,nr-03,nr-04,nr-05,nr-06,nr-07,nr-08,nr-09,nr-10,nr-11,nr-12,nr-13,nr-14,nr-15,nr-16,nr-17,nr-18,nr-19,nr-20,nr-21,nr-22,nr-23,nr-24,nr-25,nr-26,nr-28,nr-29,nr-30,nr-31,nr-32,nr-33,nr-34,nr-35,nr-36,nr-37,nr-38

.PHONY: all pipeline extract chunk index eval synth test clean help v2-all v2-ingest v2-chunk v2-index v2-eval v2-synth

help:
	@echo "Alvos disponíveis:"
	@echo "  make all       - Executa todo o pipeline v1 (extract, chunk, index, eval)"
	@echo "  make extract   - Extrai texto estruturado dos PDFs das 5 NRs principais (v1)"
	@echo "  make chunk     - Gera chunks de ~200-400 tokens alinhados às seções (v1)"
	@echo "  make index     - Constrói o banco index.db com SQLite FTS5 (BM25) (v1)"
	@echo "  make eval      - Executa avaliação de Recall@1/3/5 e MRR contra qa_pairs.jsonl (v1)"
	@echo "  make synth     - Gera perguntas sintéticas de operários com o LLM CLI ('claude -p') (v1)"
	@echo "  make test      - Executa a suíte de testes unitários do pipeline (v1 e v2)"
	@echo "  make v2-all    - Executa todo o pipeline v2 (ingest, chunk, index, eval)"
	@echo "  make v2-ingest - Ingestão do dataset parquet limpo (36 NRs + manuais)"
	@echo "  make v2-chunk  - Gera os 3 conjuntos de chunks (5nr, 5nr+manual, 36nr)"
	@echo "  make v2-index  - Constrói os 3 índices FTS5 (index_hf_5nr, index_hf_5nr_manual, index_hf_36nr)"
	@echo "  make v2-eval   - Executa as 4 análises comparativas do v2 (eval_v2.py)"
	@echo "  make v2-synth  - Gera perguntas sintéticas adicionais cobrindo NR-01, NR-33, NR-16, NR-26"
	@echo "  make clean     - Remove artefatos intermediários gerados"

all: extract chunk index eval

extract:
	@echo "==> [1/4] Extraindo PDFs das NRs..."
	$(PYTHON) -m corpus.extract --pdf-dir $(DATA_DIR) --output-dir $(EXTRACT_DIR) --nrs $(NRS)

chunk:
	@echo "==> [2/4] Gerando chunks alinhados às seções..."
	$(PYTHON) -m corpus.chunk --extracted-dir $(EXTRACT_DIR) --output-file $(CHUNKS_FILE) --nrs $(NRS)

index:
	@echo "==> [3/4] Construindo índice SQLite FTS5 (index.db)..."
	$(PYTHON) -m corpus.build_index --chunks-file $(CHUNKS_FILE) --db $(INDEX_DB) --rebuild

eval:
	@echo "==> [4/4] Avaliando retrieval BM25..."
	$(PYTHON) -m corpus.eval_retrieval --db $(INDEX_DB) --qa-file $(QA_FILE)

synth:
	@echo "==> Gerando pares sintéticos P&R com claude -p..."
	$(PYTHON) -m corpus.synth_qa --db $(INDEX_DB) --output $(QA_FILE) --force

test:
	@echo "==> Executando testes unitários..."
	$(PYTHON) -m unittest corpus.test_pipeline

v2-all: v2-ingest v2-chunk v2-index v2-eval

v2-ingest:
	@echo "==> [v2 1/4] Ingerindo Parquet limpo (36 NRs + Manuais)..."
	$(PYTHON) -m corpus.ingest_hf --parquet-path $(PARQUET_FILE) --output-dir $(EXTRACT_HF_DIR)

v2-chunk:
	@echo "==> [v2 2/4] Gerando chunks para os 3 cenários..."
	$(PYTHON) -m corpus.chunk --extracted-dir $(EXTRACT_HF_DIR) --output-file $(CHUNKS_5NR) --nrs $(NRS)
	$(PYTHON) -m corpus.chunk --extracted-dir $(EXTRACT_HF_DIR) --output-file $(CHUNKS_5NR_MANUAL) --nrs $(NRS),nr-06-manual,nr-10-manual,nr-12-manual,nr-35-manual
	$(PYTHON) -m corpus.chunk --extracted-dir $(EXTRACT_HF_DIR) --output-file $(CHUNKS_36NR) --nrs $(NRS_36)

v2-index:
	@echo "==> [v2 3/4] Construindo os 3 índices SQLite FTS5..."
	$(PYTHON) -m corpus.build_index --chunks-file $(CHUNKS_5NR) --db $(INDEX_5NR) --rebuild
	$(PYTHON) -m corpus.build_index --chunks-file $(CHUNKS_5NR_MANUAL) --db $(INDEX_5NR_MANUAL) --rebuild
	$(PYTHON) -m corpus.build_index --chunks-file $(CHUNKS_36NR) --db $(INDEX_36NR) --rebuild

v2-eval:
	@echo "==> [v2 4/4] Executando bateria comparativa v2..."
	$(PYTHON) -m corpus.eval_v2 --v1-db $(INDEX_DB) --hf-5nr-db $(INDEX_5NR) --hf-5nr-manual-db $(INDEX_5NR_MANUAL) --hf-36nr-db $(INDEX_36NR) --qa-v1 $(QA_FILE) --qa-v2 $(QA_V2_FILE)

v2-synth:
	@echo "==> Gerando perguntas sintéticas v2 (NR-01, NR-33, NR-16, NR-26)..."
	$(PYTHON) -m corpus.synth_qa --mode v2 --base-qa $(QA_FILE) --output $(QA_V2_FILE) --db $(INDEX_36NR)

clean:
	rm -rf $(EXTRACT_DIR) $(CHUNKS_FILE) __pycache__
