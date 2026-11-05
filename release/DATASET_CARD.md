---
license: other
license_name: mixed-public-documents
language:
- pt
tags:
- rag
- normas-regulamentadoras
- seguranca-do-trabalho
- retrieval
pretty_name: CEMIG NR retrieval assets (36 NRs + 9 manuais)
---

# cemig-nr-retrieval-assets

Artefatos de **busca** do melhor pipeline da POC CEMIG/CEIA (RAG offline de Normas
Regulamentadoras em PT-BR). São consumidos pelo `inference.py` do modelo
[`juniorrios/cemig-lfm3-1.7b-nr`](https://huggingface.co/juniorrios/cemig-lfm3-1.7b-nr). Os
parâmetros do ponto de operação ficam em `config.json`.

| arquivo | conteúdo |
|---|---|
| `index.db` | SQLite FTS5 (`unicode61 remove_diacritics 2`) com 2.202 trechos das **36 NRs**. Colunas doc/section/title/page/text + `expansion`, com perguntas coloquiais, sinônimos e variantes de erro de ASR geradas por LLM a partir de cada trecho (prompt genérico, sem ver perguntas de avaliação) |
| `manuals.db` | SQLite FTS5 próprio com 1.092 trechos de **9 manuais/guias oficiais do MTE** (NR-01, 06, 10, 12, 22, 32, 33, 35, 36), ids a partir de 1.000.000 |
| `dense_text.bin`, `dense_exponly.bin` | vetores EmbeddingGemma-300M QAT Q4_0 (768-d, L2) dos trechos de norma, texto+expansão e só expansão. Formato DVEC1: `b"DVEC1"` + dim i32 + count i32 + count×(id i32, f32×dim), little-endian |
| `manuals_dense.npz` | vetores dos trechos de manual (`ids`, `vecs`) |
| `units.json`, `units_win3.f16.npy` | índice do compressor: unidades (item/alínea/frase) de cada trecho (`[ini, fim, item, começa_item]` sobre o corpo após `" — "`) e o vetor fp16 da janela de 3 unidades. Ordem: normas por id, depois manuais |
| `nr_classifier.bin` | classificador de NR (TF-IDF char_wb 2-5 + word 1-2, LogReg C=8, 37 classes), formato binário NRC2, lido com numpy |
| `classic_winner.pkl` | o mesmo classificador em pickle (scikit-learn 1.9.1, precisa de `text_utils.preprocess` no path). É só referência: o `inference.py` usa o `.bin`. **Pickle executa código ao carregar; só abra se confiar na fonte** |
| `config.json` | ponto de operação: gate 0,7/2×, manuais w=1,75 com margem 0, RRF k=10, compressão 50%, geração |
| `MANIFEST.json` | md5 e tamanho de cada arquivo |

## Origem e licença dos dados

- Texto: **Normas Regulamentadoras** e **manuais/guias técnicos do Ministério do Trabalho e
  Emprego**, documentos públicos do governo federal brasileiro, extraídos dos PDFs/Markdown
  oficiais. Não há documento interno de empresa nem dado de cliente: as bases contêm só as NRs
  e os manuais listados acima.
- As expansões e o classificador vêm de dados **sintéticos** gerados pela equipe a partir do
  texto público. Nenhum benchmark privado foi usado.
- Os vetores foram calculados com o EmbeddingGemma-300M (Google; Gemma Terms of Use). A
  consulta precisa do mesmo GGUF (`ggml-org/embeddinggemma-300M-qat-q4_0-GGUF`) com
  `pooling mean` e o prompt `task: search result | query: `.
- Limitações: algumas tabelas estão corrompidas pela extração de PDF (ex.: Anexo II da NR-10), e
  o corpus está congelado na versão de 2026.
