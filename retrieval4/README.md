# Retrieval v4 — Classificador reforçado + expansão mais rica, com travas anti-mascaramento

> **Intenção do capitão:** melhorar retriever e classificador é "sempre mais garantido"
> que forçar treino do modelo — cada ponto de recall MULTIPLICA todo o pipeline. **A dúvida
> central que este relatório responde com número:** *como garantir que não vamos forçar um
> treino tão específico que mascara o real desempenho?* Resposta curta: **medimos a
> mascarada** (otimista − honesta) e só embarcamos o que sobe a **régua decisiva** (R@2/R@5
> final nas 151). O resultado foi um veredito assimétrico e honesto: a **Frente A
> (classificador) é uma mascarada e foi DESCARTADA**; a **Frente B (expansão) é real,
> generaliza, é robusta a ASR e foi EMBARCADA**.

## TL;DR — resultado

| Métrica (151 reais) | v3 (main) | **v4 (embarcado)** | Δ |
|---|---|---|---|
| Fusão 3-sinais R@2 (texto limpo, bins DVEC1) | 51.0% | **51.7%** | +0.7 p.p. |
| Fusão 3-sinais R@5 (texto limpo, bins DVEC1) | 63.6% | **65.6%** | +2.0 p.p. |
| **Fusão R@2 sob ASR real (Whisper Base, campo)** | **41.1%** | **49.0%** | **+7.9 p.p.** |
| Fusão R@5 sob ASR real | 57.0% | **62.9%** | +5.9 p.p. |
| E2E S24+ NR correta (roteiro 10) | 8/10 | **9/10** | +1 |
| E2E S24+ latência média | 6.6 s | **6.34 s** | igual (≤10s 10/10) |
| RAM PSS S24+ | 2.40 GB | **2.42 GB** | igual |

O maior ganho **não** aparece no texto limpo (onde a v3 já era forte): aparece **na fala
real transcrita por ASR** — exatamente o cenário de campo. A expansão v4 inclui variantes
de erro de transcrição e verbalizações coloquiais que **fecham o abismo entre a fala do
operário e o texto normativo mesmo quando o Whisper erra**.

## As 5 travas anti-mascaramento (o coração deste brief)

1. **Holdout intocável** — as 151 (`corpus/qa_pairs_v2.jsonl`) + 20 smoke nunca passaram
   por nenhum gerador de treino nem calibraram hiperparâmetro. Toda geração é genérica por
   chunk/NR. Filtro extra: pares-de-graça derivados de chunks-ouro do holdout são removidos
   (`v4_common.holdout_gold_chunk_ids`, 769 pares descartados).
2. **Split por NR** — reservamos **nr-33/16/26 inteiras** fora do treino do classificador
   (mesmas do finetune2) e reportamos a acurácia nelas isolada (`--nr-split`).
3. **Paráfrase adversarial** — dois conjuntos de teste: `test_samegen` (Qwen 27B, mesmo
   estilo do treino = OTIMISTA) e `test_diffgen` (**Gemma-3-1B**, família diferente + prompt
   radicalmente distinto + temperatura 1.15 + personas = HONESTO). **A distância entre eles
   é o "tamanho da mascarada".**
4. **ASR-in-the-loop** — as 151 foram sintetizadas por TTS (edge-tts pt-BR) e transcritas
   pelo **Whisper Base Q5_1 rodando no próprio S24+** (o engine do app), e o recall foi
   medido **sobre a transcrição**, não o texto limpo. É o teste mais próximo do campo.
5. **Régua que decide** — nenhum ganho de acurácia do classificador conta se o R@2/R@5
   final não subir. Reportamos SEMPRE as três medições lado a lado.

## Frente A — Classificador: 3 medições lado a lado (OTIMISTA | HONESTA | DECISIVA)

Pares-de-graça: as expansões da v3 (`retrieval3/data/expansions.jsonl`) dão **11.982**
pares pergunta-coloquial→NR sem custo (após remover 769 do holdout). Treinamos SÓ o logreg
vencedor (não refizemos o shootout — `classifier/README.md`), variando os dados.

| receita | OTIM. same t2 | HON. diff t2 | HON. holdout t2 | held-NR t2 | **mascarada** | **DEC R@2** | **DEC R@5** |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1_baseline (embarcado) | 69.3% | 34.6% | 69.0% | 80.0% | **+34.7** | **51.0%** | 63.6% |
| v1+free_all | 81.3% | 34.3% | 71.3% | 82.9% | +47.0 | 45.0% | 60.3% |
| v1+free_cap100 | 77.7% | 35.0% | 74.8% | 82.9% | +42.7 | 47.7% | 62.9% |
| v1+free_cap60_w0.3 | 75.3% | 34.6% | 77.2% | 77.1% | +40.7 | 49.7% | 62.9% |
| v1+free_eleconly_cap150_w0.5 | 69.7% | 33.6% | **79.0%** | 80.0% | +36.2 | 49.0% | 64.2% |

**Leitura (o número que responde à dúvida do capitão):**
- Toda receita com pares-de-graça **sobe a acurácia do holdout** do classificador (top-2
  69→79%) **e ao mesmo tempo DERRUBA o recall decisivo** (R@2 51→45-50%). É o sintoma
  clássico de mascarada: o classificador aprende o **vocabulário do gerador** (as expansões
  são derivadas do texto normativo e desbalanceadas — nr-15/28/12 dominam), fica
  "over-confident" na NR errada e o **filtro-duro do gate mata o chunk certo**
  (diagnóstico em `diag_gate.py`: qa-128/129 viram hard-filter na NR errada com p≈0.75).
- A **mascarada de +35 a +47 p.p.** (otimista − honesta) é a medida direta do vício. O
  `test_diffgen` (Gemma, deliberadamente vago/truncado) é mais difícil que o holdout real,
  então é um **piso pessimista**; o ponto não é o valor absoluto e sim que a mascarada
  **cresce** quando adicionamos os pares do gerador.
- **Calibração de gate/top-k** (sem dados novos, `sweep_gate.py`): top-3 boost sobe R@5
  +1.3 p.p. com R@2 igual; ganho marginal, não justifica trocar o gate top-2 validado.

**Veredito Frente A: DESCARTADA.** Isto é resultado, não fracasso — é a prova empírica,
com o gerador-diferente e a régua decisiva, de que "melhorar o classificador com mais
dados do mesmo gerador" mascara o desempenho real. O classificador **v1 baseline segue
embarcado sem mudança** (`nr_classifier.bin` 4 MB intacto).

## Frente B — Expansão mais rica: a única frente que sobe a régua decisiva

`expand_v4.py` aprofunda a maior alavanca da POC (não redescobre): por chunk, gera
**6-8 verbalizações** extras (pergunta indireta, ordem no rádio, desabafo, truncada),
**4-6 variantes com erro típico de ASR** (troca fonética, palavra colada, número por
extenso) e **sinônimos regionais adicionais**. Total sobre 2197 chunks: **17.504
verbalizações + 13.242 variantes de ASR + 19.660 sinônimos**. Campo FTS5 `expansion`
(mesmos 5 campos do app). Anti-contaminação: prompt genérico por chunk, nunca vê o holdout.

**Incremento isolado (BM25 gated, holdout 151):** R@2 37.7% → **39.7%** (+2.0 p.p.),
MRR +0.019 (`eval_expansion_v4.py`).

**Componentes na fusão (`sweep_frenteb.py`) — descoberta importante:** a expansão v4 no
**denso texto** (query↔passagem) *piora* (o ASR-noise polui o vetor de passagem), mas no
**denso expansão-only** (query↔query coloquial) *ajuda*. O combo vencedor é misto:

| combo | R@1 | R@2 | R@5 | MRR |
|---|---:|---:|---:|---:|
| BASE v3 (bm v3 + dense v3) | 37.7% | 51.0% | 63.6% | 0.4788 |
| **bm v4 + dense v3-text + v4-exp (VENCEDOR)** | **41.1%** | **51.7%** | **65.6%** | **0.4944** |

Calibrado no **dev-set sintético** (nunca holdout): **RRF k=10, pesos (BM25=2, texto=1,
exp=2)** (`calibrate_v4.py`). Gate no holdout confirma R@2 51.0% / R@5 65.6%.

## Régua decisiva sob ASR real (Trava 4) — onde o v4 brilha

Fala → TTS → **Whisper Base no S24+** → retrieval completo sobre a transcrição
(`eval_asr_fusion_true.py`, query densa reencodada na transcrição):

| cenário | R@2 | R@5 | MRR |
|---|---:|---:|---:|
| v3 texto limpo | 49.0% | 62.3% | 0.4805 |
| v3 **sob ASR** | 41.1% | 57.0% | 0.4065 |
| v4 texto limpo | 51.0% | 64.2% | 0.4951 |
| v4 **sob ASR** | **49.0%** | **62.9%** | 0.4691 |

**O v3 desaba 7.9 p.p. sob ASR; o v4 cai só 2.0 p.p.** — as variantes de erro de
transcrição na expansão tornam o retrieval robusto ao ruído do Whisper. **Este é o ganho
que mais importa para o campo** e não apareceria em nenhuma avaliação de texto limpo.

## Embarque no app (integrado + validado E2E)

- **Índice FTS5:** `index_hf_36nr_expv4.db` (19.1 MB) substitui `assets/index.db`;
  `ModelFileManager.ASSET_VERSION` 3→4 invalida a cópia antiga.
- **Denso:** `dense_text.bin` = **v3 (inalterado)**; `dense_exponly.bin` = **v4**
  (expansão v4-only reencodada com o MESMO encoder GGUF on-device, DVEC1 6.77 MB).
  Paridade dos `.bin` confirmada (`verify_bins.py`): R@2 51.7% / R@5 65.6%.
- **Fusão:** `HybridRetriever` agora usa **k=10 e pesos (BM25=2, texto=1, exp=2)** (antes
  k=30 pesos iguais). Defaults no construtor; `MainViewModel`/`AcceptanceRunner` herdam.
- **E2E S24+** (release, modo avião, `android/acceptance_results_v4.json`): **9/10 com NR
  correta no top-chunk** (v3 era 8/10; Q07 nr-06 e Q01 explícita agora acertam),
  **latência média 6.34 s (máx 7.05 s, 10/10 ≤10 s)**, dense encode 28 ms mediana, **RAM
  PSS 2.42 GB** (= v3). Única falha: **Q09 "chave fusível"** (ambiguidade genuína nr-12/18
  vs nr-10 — limite honesto já conhecido da v3, não regressão).

## Distância otimista-vs-honesta = "tamanho da mascarada" (resumo p/ o capitão)

- **Classificador com pares-de-graça:** mascarada **+35 a +47 p.p.** e recall decisivo
  **CAI** → o ganho de acurácia era ilusório. Descartado.
- **Expansão v4:** não há mascarada — o ganho aparece no holdout intocável, **generaliza**
  (held-NR não usadas em treino do classificador não foram tocadas nesta frente) e
  **cresce sob ASR** (o teste mais próximo do campo). Embarcado.
- A distinção que o capitão pediu se confirmou empiricamente: aprender
  "cinturão/trava-quedas → NR-35" via **expansão de documento** é conhecimento de domínio
  que generaliza; aprender **o gerador** (via pares sintéticos do mesmo modelo) é overfit
  que a régua decisiva expõe.

## Reprodução

Ambiente: `classifier/.venv` (sklearn 1.9.1) para tudo (busca/fusão/treino); o encoder
denso usa o **llama-server GGUF em modo embedding** (mesmo do app), não `.venv-train`.

```bash
# 0. servidor de embedding GGUF (encoder on-device) na :8399
~/llama.cpp/build-cuda/bin/llama-server -m ~/models-poc/embed/embeddinggemma-300M-qat-Q4_0.gguf \
  --host 127.0.0.1 --port 8399 -ngl 99 --embedding --pooling mean -b 2048 -ub 2048 -c 2048 --no-webui

cd retrieval4
PY=../classifier/.venv/bin/python
# FRENTE A (mascarada)
$PY gen_testsets.py --which samegen --per-nr 20 --workers 8   # Qwen 27B (:8005)
$PY gen_testsets.py --which diffgen --per-nr 20 --workers 4   # Gemma-3-1B (:8091)
$PY sweep_frentea.py            # 3 medições lado a lado + régua decisiva
$PY sweep_gate.py               # calibração de gate/top-k (dev) + gate holdout
# FRENTE B (expansão)
$PY expand_v4.py --workers 10                                   # expansão rica (vLLM 27B)
$PY build_index_v4.py --mode v4                                 # índice FTS5 v4
$PY dense_v4.py --which text --out results/densev4_text_rank.json
$PY dense_v4.py --which exp  --out results/densev4_exp_rank.json
$PY eval_expansion_v4.py        # incremento BM25 isolado
$PY sweep_frenteb.py            # componentes da v4 na fusão
$PY calibrate_v4.py             # RRF k/pesos no dev, gate no holdout
# TRAVA 4 (ASR)
$PY asr_loop.py --all           # TTS 151 -> Whisper Base no S24+ -> transcrição
$PY eval_asr_fusion_true.py     # recall fiel sobre a transcrição
# EMBARQUE
$PY export_bins_v4.py --which text --out export/dense_text.bin
$PY export_bins_v4.py --which exp  --out export/dense_exponly.bin
$PY verify_bins.py              # paridade dos .bin DVEC1
```

## Arquivos

| Arquivo | Papel |
|---|---|
| `v4_common.py` | pares-de-graça + travas 1 (anti-contaminação) e 2 (split por NR) |
| `gen_testsets.py` | trava 3: conjuntos samegen (Qwen) e diffgen (Gemma) |
| `train_v4.py` | treina só o logreg com receitas de dados; 3 medições |
| `sweep_frentea.py` | Frente A: tabela otimista/honesta/decisiva por receita |
| `sweep_gate.py` | calibração de gate/threshold/top-k (dev → gate holdout) |
| `diag_gate.py` | diagnóstico do gate (por que free piora) |
| `expand_v4.py` / `build_index_v4.py` | Frente B: expansão rica + índice FTS5 v4 |
| `dense_v4.py` / `export_bins_v4.py` | reencode denso GGUF + export DVEC1 |
| `eval_expansion_v4.py` / `sweep_frenteb.py` / `calibrate_v4.py` | régua decisiva Frente B |
| `asr_loop.py` / `eval_asr_recall.py` / `eval_asr_fusion_true.py` | trava 4 (ASR) |
| `eval_recall_v4.py` / `verify_bins.py` | recall final + paridade dos bins |
| `results/consolidation.json` | tabela mestre das 3 medições + embarque |
| `data/expansions_v4.jsonl` | expansão v4 (procedência no cabeçalho) |
| `data/test_samegen.jsonl` / `data/test_diffgen.jsonl` | conjuntos da trava 3 |
| `data/asr_transcriptions.jsonl` | 151 transcrições Whisper Base (S24+) |

Procedência: task **poc-retrieval-v4**, branch `fm/poc-retrieval-v4`. Gerações via vLLM
27B (`10.100.0.111:8005`) e Gemma-3-1B local; encoder denso via llama-server GGUF.
Comentários em PT-BR, código em inglês.
