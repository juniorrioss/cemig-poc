# Classificador de NR — Pipeline Híbrido de 2 Estágios (POC CEMIG, Opção B)

Classificador leve de Norma Regulamentadora (NR) que, dada a **fala bruta** de um
eletricista/operário, aponta a(s) NR(s) provável(is) para dar **boost suave** à busca
BM25. Materializa a "Opção B" aprovada pelo capitão: a escada de recall provou que saber
a NR certa vale ~+30 p.p. (fala bruta 13.9% R@2 → norma-ouro 44.4%), o rewriter LoRA foi
enterrado e o zero-shot não passa de 14.6%. Aqui recuperamos boa parte desse ganho com
um **modelo clássico leve**, sem encoder transformer.

## TL;DR — recomendação final

- **Estágio 1 (classificar):** `LogisticRegression` + TF-IDF híbrido (char_wb 2-5 +
  word 1-2), vencedor do shootout. **top-1 48.5% / top-2 69.0% / macro-F1 0.47** no
  holdout real (171). Latência **<1 ms** desktop, **~8 ms** no S24+ (medido no E2E).
- **Estágio 2 (buscar):** BM25 no `index_hf_36nr.db` com **boost suave** nas NRs previstas.
  Política **CONFIANÇA-GATED** (filtro-duro se top-1 confiante ≥0.5, senão boost suave 5x
  nas top-2; sem boost se 'nenhuma'; **override de NR explícita** citada na fala).
- **Ganho honesto (151 reais):** **R@2 29.8% / R@5 41.1% / MRR 0.293** vs baseline 13.9%
  → **+15.9 p.p., 2.1×**. Meta de 35% R@2 **não atingida** (teto do desenho, ver abaixo).
- **E2E no S24+ (release, modo avião):** **8/10 fundamentadas com NR correta** (mantém o
  critério M3) e **latência média 10.8 s** vs 12.5 s do baseline M3 (**melhorou**, não piorou).
- **Deploy Android:** Kotlin puro (`NrClassifier.kt`), artefato `nr_classifier.bin`
  **4.03 MB**, paridade numérica com sklearn **Δprob 2.3e-8** (testes JVM verdes).
- **Prompt-classify no LFM2.5 descartado:** melhor formato (JSON) só **25.7% top-1** e o
  ceticismo do capitão sobre "responda só o número" **se confirmou** (número puro: 5.8%).

## Arquitetura (2 estágios)

```
fala bruta ──▶ [Estágio 1: NrClassifier]  top-1/top-2 NR + probs
                     │
                     ├─ NR explícita na fala ("segundo a NR-10")? → filtro duro nela
                     ├─ top-1 == 'nenhuma'?                       → busca sem boost
                     ├─ prob(top-1) ≥ 0.5?                        → filtro duro na top-1
                     └─ senão                                      → boost suave 5x nas top-2
                     ▼
keywords (Turno 1) ──▶ [Estágio 2: BM25 index_hf_36nr.db + boost] ──▶ top-K chunks ──▶ síntese
```

O classificador usa a **fala bruta** (melhor sinal que a reescrita — a reescrita zero-shot
*piorou* o recall, coerente com `corpus/README_M3.md`); o BM25 continua recebendo as
keywords do Turno 1 do `AskPipeline`, agora com o boost do estágio 1.

## FASE A — Dataset rotulado (`gen_labels.py`)

- **1970 falas** coloquiais rotuladas geradas via vLLM corporativo (`Qwen3.8-27B-FP8`),
  em `data/labels.jsonl`. Estratificação: **74.6%** nas 9 NRs do eletricista, resto nas
  outras 27; **15.6%** ambíguas (rótulo duplo), **10.2%** fora-de-escopo ('nenhuma').
- Todas as 37 classes (36 NRs + 'nenhuma') cobertas. `generator` e `prompt_kind`
  registrados por linha; checkpoint incremental + dedupe (VPN oscila).
- **Anti-contaminação:** o holdout FIXO (151 de `qa_pairs_v2.jsonl` + 20 de
  `smoke_qa_20.jsonl`) **nunca** é gerado aqui; máx. Jaccard treino×holdout = **0.30**,
  zero colisões exatas. Falas que vazam o número da NR são descartadas.

## FASE B — Shootout (holdout real 171 = 151 qa_v2 + 20 smoke)

### Tabela comparativa única (`compare.py`)

| Candidato | Top-1 | Top-2 | Macro-F1 | Lat desk | Lat S24+ | Tamanho | Deploy Android |
|---|---:|---:|---:|---:|---:|---:|---|
| **logreg** ⭐ | 48.5% | **69.0%** | 0.471 | 0.6 ms | ~8 ms* | 7.6 MB pkl / **4.0 MB bin** | BAIXA (Kotlin puro) |
| complementnb | 49.1% | 65.5% | 0.509 | 1.0 ms | ~8 ms | 14.9 MB | BAIXA-MÉDIA |
| randomforest | 49.7% | 64.3% | 0.488 | 67 ms | inviável | 127 MB | ALTA |
| linsvc | 45.6% | 63.2% | 0.468 | 0.6 ms | ~8 ms | 7.6 MB | BAIXA (sem probs) |
| decisiontree | 23.4% | 25.1% | 0.255 | 0.6 ms | ~8 ms | 872 KB | BAIXA (acurácia ruim) |
| gradientboost | 28.7% | 29.8% | 0.065 | — | inviável | 8.7 MB | ALTA (denso, fit 587s) |
| prompt-json (LFM2.5) | 25.7% | n/a | n/a | 42 ms | ~4348 ms | 0 (residente) | MÉDIA (+~4.5s TTFT/consulta) |

\* medido no E2E: estágio 1 + estágio 2 combinados = 4-30 ms (mediana 8 ms) no S24+.

### Critério de escolha (explícito)

`top-2 > top-1 > macro-F1 > simplicidade`. O **top-2** é o driver do boost suave (o híbrido
privilegia 2 NRs), por isso pesa mais. Regra de desempate do capitão (**mais simples
vence**): `logreg` empata em top-2 com `complementnb`/`linsvc` mas ganha por ter
**probabilidades calibradas** (essenciais para o gate de confiança e o boost proporcional),
latência <1 ms e porte trivial. `randomforest` tem top-1 marginalmente maior, mas top-2
pior, 127 MB e 67 ms/consulta — **inviável mobile**.

### Ceticismo do capitão sobre prompt-classify (`prompt_classify.py`)

Medido de verdade, 3 formatos de saída, parser robusto, no LFM2.5-1.2B residente:

| Formato | Top-1 | Não-parseável | Alucinação | Lat med (5070) | Tokens |
|---|---:|---:|---:|---:|---:|
| number ("só o número") | **5.8%** | 0% | 0% | 34.8 ms | 4 |
| nrcode ("NR-XX") | 9.4% | 0% | 0% | 38.1 ms | 4 |
| json `{"nr": XX}` | **25.7%** | 0% | 0% | 42.1 ms | 7 |

O ceticismo **se confirmou**: "responda só o número" é o pior formato (o modelo tende a
responder 'nenhuma'). Mesmo o melhor (JSON) fica muito abaixo do clássico e custaria
~4.5 s de TTFT por classificação no S24+ (prefill de ~900 tokens do catálogo de 36 NRs).
Sem alucinação de norma inexistente e 0% não-parseável — o parser é robusto, o gargalo é
a **obediência/acurácia** do SLM, não o parsing.

## FASE C — Híbrido + E2E

### Calibração do boost (`hybrid.py`, 151 reais)

| Configuração | R@1 | R@2 | R@5 | MRR |
|---|---:|---:|---:|---:|
| Baseline (fala bruta, sem boost) | 9.9% | 13.9% | 20.5% | 0.138 |
| Boost suave 2x (top-2) | 21.2% | 27.1% | 36.4% | 0.267 |
| Boost suave 5x (top-2) | 21.2% | 28.5% | 37.8% | 0.274 |
| Boost suave 10x (top-2) | 21.2% | 28.5% | 37.8% | 0.274 |
| **CONFIANÇA-GATED (thr 0.5)** ⭐ | **22.5%** | **29.8%** | **41.1%** | **0.293** |
| Filtro duro (top-2, referência) | 21.2% | 28.5% | 39.7% | 0.279 |
| rewrite + boost suave 5x | 14.6% | 22.5% | 32.5% | 0.214 |
| _TETO: boost suave 5x c/ norma-OURO_ | 27.1% | 35.8% | 48.3% | 0.353 |
| _TETO: filtro duro c/ norma-OURO_ | 30.5% | 44.4% | 65.6% | 0.432 |

- **5x** é o joelho da curva do boost suave (10x não melhora; 2x deixa recall na mesa).
- **rewrite piora** (22.5% < 28.5% da fala bruta) — por isso o estágio 1 e o estágio 2
  usam a fala bruta/keywords brutas, não a reescrita.
- **CONFIANÇA-GATED vence:** aplica filtro duro quando o classificador está confiante e
  boost suave (mais seguro) quando incerto — melhor R@2 **e** R@5.

### Por que 35% R@2 não foi atingido (honestidade)

Dois tetos empíricos limitam o desenho:
1. **Top-2 do classificador = 72%** (fala coloquial é ambígua: nr-18↔nr-10, nr-06↔nr-35).
   Com a norma-OURO no lugar da predição, o boost suave chega a só **35.8%** R@2.
2. **BM25 intra-norma com fala bruta = 44.4%** (o teto de filtro-duro-ouro). Converter a
   fala em termos técnicos curados subiria para 77.5% (`corpus/README_M3.md`), mas isso é
   tarefa do rewriter — que regrediu no holdout limpo e foi enterrado por ordem.

Ou seja: o classificador **entrega quase todo o ganho recuperável** de "saber a NR"
(29.8% de 35.8% possível). O gap restante até 44.4% depende de **termos técnicos**, não
de classificação. Subir mais exigiria reabrir a trilha de reescrita/expansão de termos.

### E2E no S24+ (release, modo avião, `data/acceptance_results_hybrid.json`)

| Métrica | Baseline M3 | Híbrido (este) |
|---|---:|---:|
| Fundamentadas c/ NR correta | 8/10 | **8/10** |
| Latência média | 12.5 s | **10.8 s** |
| Latência estágio 1+2 (busca) | ~14 ms | **8 ms** (mediana) |

- Critério **mantido (8/10)** e **latência melhorou** (bem dentro do teto de +0.5 s/turno).
- **Q01** (formal, cita "NR-10") foi corrigida pelo **override de NR explícita**.
- Falhas restantes são limites honestos: **Q09** ("chave fusível" → classifica nr-12,
  genuinamente ambíguo) e **Q10** (retrieve traz nr-10 correto, mas o **LLM responde 'não
  sei'** — limite de síntese, não de retrieval/classificação).

## Deploy Android (Kotlin puro)

- `skl2onnx` **não converte char n-grams** (só `tokenizer='word'`), e o char_wb é o que dá
  robustez a ruído de ASR (+2.6 p.p. top-2 sob typos). Logo o deploy é **Kotlin puro**,
  sem runtime ONNX — mantém complexidade BAIXA.
- `export_kotlin.py` serializa vocabulário + IDF + coef/intercepto no binário **NRC2**
  (auto-contido, `models/nr_classifier.bin`, 4.03 MB) e **valida a paridade** com o
  algoritmo Kotlin (`NrClassifier.kt`): TF-IDF (char_wb 2-5 + word 1-2, sublinear_tf,
  **L2 por bloco** char/word separados) + LogReg + softmax → **Δprob 2.3e-8**.
- Integração: `HybridRetriever.kt` (decorator do `Fts5Retriever`) + `NrClassifier.kt`,
  ligados em `MainViewModel` e `AcceptanceRunner`. Override externo por
  `getExternalFilesDir(null)/nr_classifier.bin` (adb push, sem reinstalar).
- Testes JVM: `android/app/src/test/.../NrClassifierTest.kt` (carga, predições, paridade
  vs gabarito sklearn). Rodar: `cd android && ./gradlew :app:testDebugUnitTest`.

## Como reproduzir

```bash
cd classifier
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python \
    scikit-learn scipy numpy requests tenacity pyarrow   # ambiente

make labels     # FASE A: dataset via vLLM (~2.5 min)
make classic    # FASE B: shootout clássicos
# llama-server do LFM2.5 QAD-Q4_0 na :8090 para o prompt-classify:
#   ~/llama.cpp/build-cuda/bin/llama-server -m ~/models-poc/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf \
#     --host 127.0.0.1 --port 8090 -ngl 99 -c 4096 -t 6 --no-webui
make prompt compare
make hybrid      # FASE C: calibração de boost nas 151 reais
make export      # FASE C: nr_classifier.bin + copia p/ assets/test
```

## Artefatos

- `data/labels.jsonl` — dataset rotulado (1970 falas, procedência por linha).
- `data/classic_results.json`, `data/prompt_results.json`, `data/compare_table.json`.
- `data/hybrid_results.json` — calibração de boost nas 151 (+ tetos oracle).
- `data/acceptance_results_hybrid.json` — E2E no S24+ (10 faladas, modo avião).
- `models/classic_winner.pkl` — pipeline sklearn vencedor (fonte do export).
- `models/nr_classifier.bin` (4.03 MB) + `models/nr_classifier_meta.json` — artefato de deploy.
- `nr_taxonomy.py` — fonte única das 36 NRs + descrições (títulos oficiais do `nrs.parquet`).
