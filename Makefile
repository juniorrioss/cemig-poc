# Makefile raiz — CEMIG POC Assistente por Voz Offline

PYTHON ?= $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: all pipeline extract chunk index eval synth test clean help

help:
	@echo "CEMIG POC Assistente Offline - Comandos Principais:"
	@echo "  make all       - Executa todo o pipeline do corpus (extract -> chunk -> index -> eval)"
	@echo "  make extract   - Extrai texto das NRs principais a partir dos PDFs"
	@echo "  make chunk     - Segmenta NRs em chunks de ~200-400 tokens"
	@echo "  make index     - Constrói o banco SQLite FTS5 (corpus/index.db)"
	@echo "  make eval      - Executa avaliação de Recall@1/3/5 do BM25"
	@echo "  make synth     - Gera dataset sintético P&R de operários com o LLM CLI"
	@echo "  make test      - Roda testes unitários do pipeline"
	@echo "  make clean     - Limpa artefatos temporários"

all:
	@$(MAKE) -C corpus all

extract:
	@$(MAKE) -C corpus extract

chunk:
	@$(MAKE) -C corpus chunk

index:
	@$(MAKE) -C corpus index

eval:
	@$(MAKE) -C corpus eval

synth:
	@$(MAKE) -C corpus synth

test:
	@$(MAKE) -C corpus test

clean:
	@$(MAKE) -C corpus clean
