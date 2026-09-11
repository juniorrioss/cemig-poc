# M3 — Diagnóstico de Retrieval e Chunking (Fase 1)

Investigação do sintoma "quase toda pergunta retorna 'Não sei com base nas normas
consultadas'" no app real do S24+, ordenada pela firstmate: **retrieval primeiro**.

## Metodologia honesta (anti-overfitting)

Todas as métricas abaixo usam **as 151 perguntas coloquiais reais** de
`corpus/qa_pairs_v2.jsonl` — conjunto NÃO-contaminado, escrito à mão, distinto de
qualquer dado sintético. A busca replica **fielmente o caminho do app**
(`corpus/eval_fino.py::app_fts_query`): sanitização idêntica ao `Fts5Retriever.kt`
(remoção de acentos, stopwords de domínio, radical de 6 chars, prefixo `*`, junção
`OR`) e pesos BM25 de produção `bm25(1.5, 3.0, 2.0, 1.0)`.

Isto corrige a distorção do relatório v2 anterior, cujo "85.4% Recall@5" usava
**match exato + filtro de norma-ouro** — dois artifícios que o app NÃO executa.

Rodar: `python3 -m corpus.eval_fino --db <db> --qa corpus/qa_pairs_v2.jsonl [--no-rewrite]`.

## Causa-raiz do chunking

`corpus/ingest_hf.py` já extrai **itens finos** (ex.: NR-10 = 184 itens, mediana 34
tokens). Mas `corpus/chunk.py::chunk_items()` **remescla** os itens em blocos grossos
de ~380 tokens por seção maior, colapsando a NR-10 para 23 chunks. Foi adicionada uma
política de granularidade de item (`chunk_items_fine`, flag `--fine`), gerando
`index_hf_36nr_fino.db` (4.429 chunks; NR-10 = 60 chunks; 8.70 MB) e uma variante média
`index_hf_36nr_med.db` (3.115 chunks).

## Resultado 1 — Chunking fino/médio PIORAM o recall

Contraintuitivamente ao brief, fragmentar dilui o sinal BM25 (o item certo compete
com dezenas de itens curtos irmãos). Com **termos curados + filtro de norma-ouro**
(teto):

| Índice | chunks | R@2 | R@5 |
|---|---:|---:|---:|
| `index_hf_36nr.db` (grosso, atual) | 2.202 | **77.5%** | **91.4%** |
| `index_hf_36nr_med.db` (médio) | 3.115 | 60.3% | 74.8% |
| `index_hf_36nr_fino.db` (fino) | 4.429 | 51.0% | 67.5% |

**Decisão: manter o índice grosso `index_hf_36nr.db`.** O código do chunker fino fica
disponível (`--fine`) mas não é embarcado. A desbalanceamento entre NRs (NR-15=307 vs
NR-10=23) NÃO é o gargalo sob busca com filtro de norma.

## Resultado 2 — O gargalo real é o REWRITER + o FILTRO DE NORMA

Isolando cada fator no índice grosso (caminho do app):

| Fonte da query | Filtro de norma | R@2 | R@5 |
|---|---|---:|---:|
| Pergunta bruta (voz) | não | 13.9% | 20.5% |
| Rewrite zero-shot (LFM2.5-1.2B, prompt de produção) | não | 14.6% | 21.2% |
| Pergunta bruta | **oracle (norma-ouro)** | **44.4%** | **65.6%** |
| Termos curados | não | 51.0% | 68.2% |
| Termos curados | **oracle (norma-ouro)** | **77.5%** | **91.4%** |

O salto de 13.9% → 44.4% (só ligando o filtro de norma) e de 44.4% → 77.5% (termos
curados) mostra que **o valor está em (a) descobrir a NR certa e (b) converter a fala
em termos técnicos**. Ambos são tarefa do rewriter (Turno 1), não do chunking.

- Estratégias de mitigação de dominância testadas e **rejeitadas** por número:
  boost de campo (5,5,2,1), índice 2 camadas (5 críticas), auto-detecção de norma pelo
  top-k da busca ampla — todas ≤ pergunta bruta, porque a fala coloquial não faz o
  chunk certo emergir para votar a norma.

## Resultado 3 — Critério de saída da Fase 1 (R@2 ≥ 45% zero-shot) NÃO atingível por retrieval

O teto zero-shot medido é **14.6% R@2** (o rewriter genérico não melhora a fala). Nenhum
ajuste de chunking/boost/filtro-automático chega a 45% sem um rewriter que **infira a
norma e produza termos técnicos**. Teto documentado; segue-se para Fase 2 (rewriter) e
Fase 3 (app), conforme o brief autoriza.

## Artefatos
- `corpus/eval_fino.py` — harness Fase 1 (rewrite zero-shot cacheado + estratégias).
- `corpus/chunk.py::chunk_items_fine` — chunker de granularidade de item (`--fine`).
- `corpus/data/eval_*_*.json`, `corpus/data/rewrites_*.json` — métricas e caches.
- Índices: `index_hf_36nr_fino.db`, `index_hf_36nr_med.db` (diagnóstico; não embarcados).
