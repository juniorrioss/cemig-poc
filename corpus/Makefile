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

.PHONY: all pipeline extract chunk index eval synth test clean help

help:
	@echo "Alvos disponíveis:"
	@echo "  make all       - Executa todo o pipeline (extract, chunk, index, eval)"
	@echo "  make extract   - Extrai texto estruturado dos PDFs das 5 NRs principais"
	@echo "  make chunk     - Gera chunks de ~200-400 tokens alinhados às seções"
	@echo "  make index     - Constrói o banco index.db com SQLite FTS5 (BM25)"
	@echo "  make eval      - Executa avaliação de Recall@1/3/5 e MRR contra qa_pairs.jsonl"
	@echo "  make synth     - Gera perguntas sintéticas de operários com o LLM CLI ('claude -p')"
	@echo "  make test      - Executa a suíte de testes unitários do pipeline"
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

clean:
	rm -rf $(EXTRACT_DIR) $(CHUNKS_FILE) __pycache__
