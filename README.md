# cemig-poc — Assistente por voz 100% offline (POC)

Q&A sobre Normas Regulamentadoras (domínio elétrico) rodando inteiramente no dispositivo Android (Galaxy S21/S24+), sem internet na execução.

Pipeline: Voz (PT-BR) → ASR → SLM com tool-calling → `retriever(query)` BM25/SQLite-FTS5 → resposta com citação de fonte obrigatória.

## Estrutura
- `corpus/` — pipeline desktop (Python): PDF das NRs → chunks → index.db (FTS5) → avaliação (recall@k, P&R sintético)
- `bench/` — benchmark de SLMs (llama.cpp GGUF, protocolo único de qualidade PT + tool-calling)
- `asr/` — bateria de experimentos ASR (nativo Android, whisper.cpp, Nemotron/sherpa-onnx)
- `android/` — app Kotlin/Compose (tema CEIA, motores fake → reais)

Plano completo e decisões: artefato Lavish do firstmate (`.lavish/poc-slm-mobile.html`).

Requisitos-chave: latência total ≤ 10 s no S21 · fonte obrigatória · só texto · RAG disparado via tool pelo SLM (nem toda pergunta consulta; o modelo pode reformular a query).
