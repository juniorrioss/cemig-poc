# Retrieval v3 — Fechando o abismo de vocabulário (POC CEMIG)

> **Intenção do capitão:** "o retriever ainda é um gargalo mto grande… 29% é muito
> baixo… precisamos ter números melhores." Requisito fundacional da POC: **100%
> offline na execução** (sem depender de internet na hora do uso em campo). Restrições:
> **holdout intocável** e **seleção de embeddings mobile-first** (proibido bge/e5 como
> default; proibido tocar nos `vector_db` da UFGCEMIGONA).

## TL;DR — resultado

**Entrega v3 aceita: pipeline 100% on-device com R@2 52.3% / R@5 63.6% nas 151** —
salto de **+22.5 p.p. R@2 (1.75×)** sobre o main. "Números melhores" atendido dentro
do requisito offline; ver `## Decisão` abaixo.

| Pipeline | R@1 | **R@2** | R@5 | MRR | Caminho do produto? |
|---|---|---|---|---|---|
| E0 — main atual (híbrido classificador+BM25) | 22.5% | **29.8%** | 41.1% | 0.293 | on-device (produção anterior) |
| E1 — + expansão de documento | 29.1% | **37.7%** | 52.3% | 0.374 | on-device |
| E3 — + denso EmbeddingGemma (fusão 2 sinais) | 39.7% | **49.0%** | 62.9% | 0.483 | on-device |
| **E3 — fusão 3 sinais (VENCEDOR — ENTREGA v3)** | 37.1% | **52.3%** | 63.6% | 0.478 | **on-device (produto de campo)** |
| E4 — + reranker 27B listwise | 38.4% | 56.3% | 68.2% | 0.509 | ✗ exige nuvem — **NÃO é o produto** (só modo conectado opcional) |

## Decisão (capitão, 2026-09-11)

**Caminho A — on-device puro — é a entrega v3.** O requisito fundacional da POC é
execução 100% offline em campo; um reranker na nuvem contradiz a razão de existir do
produto. Os **52.3% R@2 / 63.6% R@5 mobile-only** são a entrega aceita (a marca de 55%
era projeção interna, não requisito). O **reranker 27B na nuvem fica registrado apenas
como opção futura para cenário conectado opcional** (ex.: modo escritório), **nunca**
como caminho do produto de campo.

Métrica oficial: 151 perguntas reais de `corpus/qa_pairs_v2.jsonl`; o holdout (151 + 20
smoke) **nunca** foi usado para ajustar nada (calibração só no dev-set sintético
`data/dev_set.jsonl`). Caminho de busca idêntico ao app (`app_fts_query`, pesos
1.5/3/2/1, `check_hit`), reusando `corpus/eval_*`.

## Etapa a etapa

### Etapa 1 — Expansão de documento (offline) — GANHO GRANDE
Para cada um dos 2202 chunks, o vLLM do capitão (Qwen3.8-27B-FP8) gerou **4-6 perguntas
coloquiais de operário** que o chunk responde + **sinônimos leigos** dos termos técnicos
(ex.: `talabarte = cinto/corda; desenergização = desligar a energia`). Indexado num 5º
campo FTS5 `expansion` (peso calibrado = 1.0 do texto no dev-set).
- **Anti-contaminação:** prompt genérico por chunk; **nunca** viu as 151.
- **Resultado:** BM25 gated 29.8% → **37.7% R@2**, 41.1% → **52.3% R@5**.
- Artefatos: `expand_docs.py`, `data/expansions.jsonl` (2200/2202, procedência no header),
  `build_expanded_index.py`, `indices/index_hf_36nr_exp.db` (14.08 MB).

### Etapa 2 — RRF multi-query BM25 (runtime, custo ~0) — SEM GANHO
Fundir por RRF a fala bruta + fala-com-boost + keywords TF-IDF do classificador **não
supera** o híbrido gated (o filtro-duro confiante já é forte). Mantido como diagnóstico.
- Artefatos: `rrf.py`, `eval_rrf.py`.

### Etapa 3 — Busca densa mobile-first — GANHO GRANDE
Seleção própria com critério mobile explícito **antes** de testar (ver
`MODEL_SELECTION.md`). Vencedor: **EmbeddingGemma-300M** (Google, mobile-first,
Matryoshka, prompts query/document nativos). Dois índices densos complementares:
- **texto+expansão** (query↔passagem): dense-puro 34.4% R@2.
- **expansão-only** (query↔query coloquial): dense-puro 37.1% R@2, R@5 57.6%.

Fundidos por RRF com o BM25-gated-expandido:
- BM25-exp + Gemma-texto: **49.0% R@2**.
- **BM25-exp + Gemma-texto + Gemma-exp-only (3 sinais, k=30): 52.3% R@2 / 63.6% R@5.**

Modelos triados e o porquê da escolha estão em `MODEL_SELECTION.md`. Controle exigido pela
ordem: **e5/MiniLM ficaram muito atrás** dense-puro (e5-small 10.6%, MiniLM-paraphrase
4.0% — simétrico, impróprio p/ retrieval), **provando** que os mobile-first do capitão são
superiores; bge/e5 **não** entram como default.

### Etapa 4 — Reranker — NÃO faz parte do produto de campo
Sobre os top-10 da fusão 3-sinais, buscando ultrapassar a marca interna de 55%:
- **27B listwise: 56.3% R@2** — mas **exige nuvem**, o que contraria o requisito offline
  fundacional. Registrado **apenas como opção futura p/ cenário conectado opcional**
  (ex.: modo escritório com Wi-Fi), **nunca** o caminho do produto de campo.
- Cross-encoders mobile (mMiniLM, bge-reranker, jina-v2) **regridem** (gap de domínio).
- **LFM2.5-1.2B embarcado** como reranker (listwise ou pointwise sim/não) **piora**
  (43-46% R@2): modelo pequeno não ranqueia listwise de forma confiável.
- Conclusão: **não há reranker on-device que ajude hoje**; a fusão 3-sinais (Etapa 3) é
  o ponto final do produto de campo.
- Artefatos: `rerank.py` (cross-encoder), `rerank_llm.py` (listwise), `rerank_pointwise.py`.

## Pipeline vencedor (mobile) — especificação p/ integração

> **NÃO integrado ao app nesta task** (a consolidação do app corre em paralelo).
> Entregue como biblioteca + relatório.

```
consulta (fala bruta do operário)
  ├─ classificador de NR (Kotlin puro, já no app) → top-2 NR + probs
  ├─ s1: BM25 gated-expandido  (index_hf_36nr_exp.db, pesos 1.5/3/2/1/1)  ~5-15 ms
  ├─ sT: denso EmbeddingGemma-300M sobre texto+expansão (sqlite-vec)      ~encode+ANN
  └─ sE: denso EmbeddingGemma-300M sobre expansão-only  (sqlite-vec)      (1 encode serve p/ sT e sE)
        → RRF(k=30, pesos iguais) → top-5
```

### Custo mobile do componente novo (denso)
| Item | Valor |
|---|---|
| Modelo | EmbeddingGemma-300M (≤350M, mobile-first Google) |
| Dim | 768 (Matryoshka: truncável p/ 256 → índice ~menor, perda pequena) |
| Índice denso (texto+exp) | 17.39 MB (sqlite-vec) |
| Índice denso (exp-only) | 14.05 MB (sqlite-vec) |
| Índice BM25 expandido | 14.08 MB (FTS5) |
| Encode 1 query (RTX 5070) | ~37 ms |
| Encode 1 query projetado S24+ (int8, ~30 tok) | ~150-250 ms (Google reporta <200 ms CPU / <50 ms EdgeTPU); **dentro do teto de 300 ms** |
| Motor vetorial | **sqlite-vec** (compila arm64, mesma família do `libsqliteX` já embarcado) |

Somando ao orçamento atual (SLM ~700 MB + ASR ~197 MB), o denso acrescenta ~300 MB de
modelo int8 + ~31 MB de índices vetoriais + ~14 MB do índice BM25 expandido. **Requer
validação de RAM no S21** (6 GB); no S24+/≥8 GB é confortável. A busca vetorial é 1 encode
+ ANN top-K (~dezenas de ms).

## Reprodução

Ambientes (ver AGENTS.md): busca/fusão/classificador em **`classifier/.venv`** (sklearn
1.9.1); encode denso/rerank em **`.venv-train`** (torch cu128, sentence-transformers,
sqlite-vec). O classificador foi picklado com sklearn 1.9.1 → o encode denso escreve
rankings em cache (`dump_dense.py`) e a fusão avalia no venv do classificador.

```bash
make -C retrieval3 all         # E0..E5 a partir dos artefatos já gerados
# ou passo a passo:
make -C retrieval3 baseline    # E0
make -C retrieval3 expand      # E1: gera expansão (vLLM) + índice + gate  [~25 min, VPN]
make -C retrieval3 dense       # E3: build índices densos (GPU) + dump + fusão
make -C retrieval3 rerank      # E4: teto 27B + candidatos mobile
make -C retrieval3 consolidate # E5: tabela final -> results/consolidation.json
```

## Higiene e procedência
- Toda geração via vLLM/LLM é **não-interativa, com timeout e retry**; expansão e dev-set
  têm **checkpoint incremental** (a VPN oscila) e **cabeçalho de procedência** (modelo,
  endpoint, prompt-hash, timestamp) na 1ª linha do JSONL.
- **Holdout intocável:** o dev-set sintético (`gen_devset.py`, seed 13, 200 perguntas)
  exclui da amostragem os chunk-ouro do holdout; toda calibração de k/pesos usa só o dev.
  O dev é mais fácil que o holdout (perguntas menos idiomáticas) → **decisões finais
  ancoradas no gate do holdout**, não no dev.

## Arquivos
| Arquivo | Papel |
|---|---|
| `eval_common.py` | harness honesto (holdout, check_hit, app_fts_query, métricas) |
| `bm25.py` | BM25/FTS5 + classificador + híbrido gated (réplica do app) |
| `expand_docs.py` / `build_expanded_index.py` | Etapa 1 |
| `rrf.py` / `eval_rrf.py` | Etapa 2 (RRF multi-query) |
| `MODEL_SELECTION.md` | Etapa 3 — triagem mobile-first (critério + veredito) |
| `dense.py` / `dump_dense.py` / `run_dense_model.sh` | Etapa 3 — encode + índice sqlite-vec |
| `eval_dense.py` / `eval_fusion_final.py` | Etapa 3 — denso puro + fusão |
| `rerank*.py` / `dump_fusion*.py` | Etapa 4 — rerankers e candidatos |
| `consolidate.py` | Etapa 5 — tabela final + pipeline vencedor |
| `results/*.json` | métricas de cada etapa (procedência preservada) |

Procedência: task **poc-retrieval-v3**, branch `fm/poc-retrieval-v3`.
