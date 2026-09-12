# ctx_topk — JANELA × NÚMERO DE TRECHOS + PAINEL DE EIXOS da v4

Bancada de duas ordens do capitão sobre o pipeline **v4 embarcado** (retrieval RRF 3-sinais
+ **LFM2.5-1.2B QAD-Q4_0**, o sintetizador que está no app):

- **ORDEM 1 — janela × trechos.** O app envia só **2 trechos** (`AskPipeline.topK=2`), mas a
  v4 mede **R@2 51,7% vs R@5 65,6%** — ~14 p.p. de resposta certa são DESCARTADOS por não
  caberem na janela. Chunks têm ~370 tok; 5 chunks ≈ 1850 tok de contexto, estourando o
  orçamento atual. A pergunta que decide: **adiantou enxergar mais trechos?** Medido de
  verdade — qualidade no juiz 27B (151 reais) e **custo real no S24+** (não estimado).
- **ORDEM 2 — painel de eixos da v4.** Para a versão embarcada, o mesmo diagnóstico por eixo
  do `sintese2`: **média, desvio, mediana, HISTOGRAMA 1-5 por eixo** (acerto_factual,
  fidelidade, qualidade_pt, citacao_fonte) e o **EIXO-GARGALO explícito** — não só gate binário.

## TL;DR — o veredito

> **Enxergar mais trechos NÃO adianta com o LM 1.2B. Manter topK=2 (full) é o resultado legítimo.**

- A **visão sobe** exatamente como o capitão previu: chunk-ouro visível vai de **51,7% (k2)
  → 59,6% (k3) → 64,2% (k4) → 65,6% (k5)** — +13,9 p.p. de R@5 entram na janela.
- Mas a **qualidade da resposta fica FLAT**: global ~1,8, **gate-pass ~1-2% em TODAS as 8
  células** (Δgate máx **+0,7 p.p.**, dentro do ruído). O ganho teórico de R@5 **quase não se
  converte** em resposta aprovada: `Δaprovadas / Δchunks_visíveis ≈ 0` (ver `analysis.json`).
- **EIXO-GARGALO GLOBAL = CITAÇÃO** (μ **1,23**), seguido de acerto (1,55) e fidelidade
  (1,56); **PT-BR NÃO é gargalo** (μ 3,64). Diferente da v3 (onde a citação baixa era efeito
  do retrieval), aqui **o chunk-ouro está visível e mesmo assim o 1.2B não cita o item certo**
  — o teto é o **LM**, não a visão. (Ex.: qa-001 tem o 10.2.8 no top-5 e a resposta cita
  "10.5" genérico; R01 no aparelho responde "10.5.1 e 10.5.2" sem o prefixo "NR-".)
- **Custo real no S24+ é proibitivo para 5 trechos full:** o prompt triplica (1419→3066 tok)
  e o **TTFT triplica: 7,9 s → 21,8 s** (llama-bench sob carga) / wall real **13,8 s → 44,1 s**.
  O capitão valoriza TTFT acima do total: **+13,9 s de TTFT por zero ganho de qualidade é
  regressão pura de experiência.**
- **TRIM** (recorte determinístico) resolve o CUSTO (cabe 5 trechos em ~1340 tok, como o k2)
  **mas não a QUALIDADE** (gate igual/menor) e **piora levemente a citação** e o acerto de
  norma no aparelho (**13/20 vs 18/20** do full) — cortar o miolo do chunk remove o item que
  faltava citar. Não compensa.

**Decisão:** **MANTER `topK=2` full como default.** O default do app **não muda** (sem
evidência que pague o custo — exatamente o critério do brief). A alavanca de qualidade
continua sendo **(a) o LM** (citação/síntese: 2.6B thinking-OFF ou fine-tune, ver
`finetune2/` e `android/README_ENGINE_UPGRADE.md`) e **(b) o retrieval intra-norma** — não o
número de trechos.

---

## PART 1 — Grade de configurações (bancada 5070)

Grade completa **topK ∈ {2,3,4,5} × trecho ∈ {full, trim}** = 8 células de qualidade, nas
**151 reais**, com o pipeline v4 completo e o **1.2B QAD embarcado**. Juiz vLLM 27B (8
workers), 4 eixos + gate.

### Por que `n_ctx` não é uma célula de qualidade
Com sampling determinístico (temp 0.1, seed fixa), a **resposta depende só do conteúdo do
prompt** (topK, trim), não de `n_ctx` — que apenas dimensiona o KV cache. Validado: mesmo
prompt → saída idêntica em `n_ctx` 2048 vs 4096. Por isso `n_ctx` entra como eixo de
**VIABILIDADE + LATÊNCIA** (PART 2), não de qualidade. **Combinações que não cabem não foram
rodadas como qualidade** — são medidas de custo/viabilidade no aparelho.

### Tabela — 8 células × eixos × gate × visão × conversão

| célula | Fac | Fid | PT | Cit | Global | Gate% | R_hit% (visão) | conv% | gargalo | prompt_tok μ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|--:|
| **k2_full** (default) | 1.60 | 1.62 | 3.70 | 1.25 | **1.85** | **1.3** | 51.7 | 2.6 | cit | 1387 |
| k2_trim | 1.55 | 1.53 | 3.69 | 1.15 | 1.78 | 0.7 | 51.7 | 1.3 | cit | 1262 |
| k3_full | 1.54 | 1.55 | 3.67 | 1.23 | 1.80 | 2.0 | 59.6 | 3.3 | cit | 1949 |
| k3_trim | 1.49 | 1.48 | 3.64 | 1.33 | 1.78 | 2.0 | 59.6 | 3.3 | cit | 1324 |
| k4_full | 1.64 | 1.74 | 3.62 | 1.21 | **1.88** | 1.3 | 64.2 | 1.0 | cit | 2491 |
| k4_trim | 1.47 | 1.49 | 3.62 | 1.26 | 1.76 | 1.3 | 64.2 | 2.1 | cit | 1359 |
| k5_full | 1.60 | 1.61 | 3.56 | 1.11 | 1.80 | 1.3 | **65.6** | 2.0 | cit | 3032 |
| k5_trim | 1.51 | 1.46 | 3.58 | 1.28 | 1.76 | 2.0 | 65.6 | 2.0 | cit | 1365 |

**Leitura (a pergunta que decide):** a coluna **R_hit%** (visão) sobe +13,9 p.p.; **Gate%** e
**Global** ficam parados. `analysis.json → tradeoff_vs_k2full` mostra `convert_ratio` ~0 (a
visão extra não vira aprovação). O gargalo é **cit** em todas.

### Painel por eixo (ORDEM 2) — exemplo do default k2_full
```
fac: μ=1.60 σ=0.74 med=1 hist={1:79, 2:59, 3:8,  4:5,  5:0}
fid: μ=1.62 σ=0.91 med=1 hist={1:95, 2:27, 3:21, 4:8,  5:0}
pt : μ=3.70 σ=0.55 med=4 hist={1:0,  2:0,  3:52, 4:92, 5:7}
cit: μ=1.25 σ=0.79 med=1 hist={1:134,2:4,  3:9,  4:0,  5:4}
```
Histogramas completos das 8 células + quase-pass por eixo em `data/analysis.json` e no stdout
de `analyze.py`. **PT-BR é o único eixo saudável** (moda 4); os três eixos técnicos colapsam
em 1 — assinatura de um LM pequeno que não ancora item normativo, independente de quantos
trechos vê.

---

## PART 2 — Custo real no S24+ (medido, não estimado)

Levadas ao aparelho as **3 configs de custo** (k2_full = default; k5_full = máx trechos, pior
custo; k5_trim = máx trechos com recorte barato). Duas frentes:
- **llama-bench** (`-t 6 -r 3`) no **tamanho de prompt representativo** de cada config
  (mediana medida nas 151) → prefill/decode/TTFT + RAM pico (VmHWM, polling 50 ms);
- **llama-cli** em **20 prompts reais** (roteiro-10 + **10 extra fora do holdout**,
  `data/device_questions.jsonl`) → prefill/gen/wall reais + **acerto de norma** (a NR citada
  bate a esperada).

| config | prompt_tok | prefill (bench) | **TTFT S24+** | decode | RAM pico | wall real (med) | **acerto norma** |
|---|--:|--:|--:|--:|--:|--:|:--:|
| **k2_full** (default) | 1419 | 180 t/s | **7,9 s** | 22,4 t/s | 1420 MiB | 13,8 s | **18/20** |
| k5_full | 3066 | 141 t/s | **21,8 s** | 21,9 t/s | 1440 MiB | 44,1 s | 18/20 |
| k5_trim | 1338 | 140 t/s | 9,6 s | 18,7 t/s | 1420 MiB | 8,6 s | **13/20** |

> Nota: o prefill do llama-bench cai sob carga térmica (alta variância σ≈25 t/s); os prompts
> reais com aparelho frio deram prefill mediano **164 t/s (k2) / 103 (k5_full) / 231 (k5_trim)**.
> O sinal é robusto: **k5_full ~3× o TTFT do k2_full**; k5_trim volta ao patamar do k2 em
> latência, mas ao custo de **5 acertos de norma a menos** (o TRIM remove o item citável).

### Viabilidade de `n_ctx` (importante p/ embarque)
O app carrega **`ctxSize=2048`** hoje. Com chunks v4 (~370 tok), **só cabem em 2048**:
`k2_full/k2_trim` e **todos os trim**. **`k3_full` estoura em 125/151, `k4_full` em 150/151,
`k5_full` em 151/151** (máx 3508 tok). Ou seja, **qualquer full > 2 trechos exige subir
`n_ctx` para 4096** — que custa **+26 MiB** de KV (medido: 1433→1459 MiB só do LLM; some
encoder ~1 GB + Whisper) além do TTFT. Detalhes em `results/device_cost.json → nctx_*`.

---

## PART 3 — Recomendação (tabela mestre em `consolidate.py`)

**Regras de decisão explícitas** (não hard-coded): embarcar topK maior só se
`Δgate_pass ≥ 3,0 p.p.` **E** o TTFT no aparelho não regredir além de `+2,0 s`.

- **Nenhuma** célula atinge +3,0 p.p. de gate (máx **+0,7**, ruído).
- Subir p/ 5 trechos full custa **+13,9 s de TTFT** — regressão de experiência sem ganho.
- TRIM neutraliza o custo de latência mas **piora citação/acerto de norma** — não compensa.

### → MANTER `topK=2` (full). Default do app inalterado.
Não há evidência que pague o custo. Isto é um resultado legítimo pedido pelo brief
("manter 2 trechos é um resultado legítimo"). **O app não foi alterado.** As alavancas reais
de qualidade seguem sendo o **LM** (citação/síntese) e o **retrieval intra-norma**, não o
número de trechos.

---

## Arquivos e reprodução

Ambiente: `classifier/.venv` (sklearn 1.9.1). Encoder denso GGUF via llama-server (:8399);
sintetizador 1.2B via llama-server CUDA (:8410, `--parallel 4`, `-c 16384`).

```bash
# 0. Servidores (RTX 5070)
~/llama.cpp/build-cuda/bin/llama-server -m ~/models-poc/embed/embeddinggemma-300M-qat-Q4_0.gguf \
  --host 127.0.0.1 --port 8399 -ngl 99 --embedding --pooling mean -b 2048 -ub 2048 -c 2048 --no-webui
~/llama.cpp/build-cuda/bin/llama-server -m ~/models-poc/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf \
  --host 127.0.0.1 --port 8410 -ngl 99 -c 16384 --parallel 4 -b 2048 -ub 512 --no-webui

PY=../../classifier/.venv/bin/python
cd bench/ctx_topk
# PART 1 — grade de qualidade (8 células) + juiz + painel de eixos
./run_grid.sh                                   # 8 responses_*.json
$PY build_judge_input.py
$PY ../judge.py --input data/harness_for_judge.json \
   --output-eval data/judge_evaluations.json --output-csv data/judge_summary.csv --no-calibration
$PY analyze.py                                  # painel de eixos + tradeoff -> data/analysis.json
# PART 2 — custo real no S24+
$PY gen_device_prompts.py                       # prompts reais das 3 configs de custo
PATH=~/android-sdk/platform-tools:$PATH $PY device_bench.py --serial 192.168.0.6:41073 --all
# PART 3 — tabela mestre + recomendação
$PY consolidate.py                              # results/consolidation.json
```

| Arquivo | Papel |
|---|---|
| `retrieval_v4.py` | espelho do retrieval v4 EMBARCADO (RRF k=10 w=2,1,2). **Paridade provada** com os `.bin` DVEC1 puxados do S24+: R@2 51.7 / R@5 65.6 / MRR 0.4944 |
| `trim.py` | recorte determinístico de chunk (sem LLM): janela em torno do melhor casamento, preserva cabeçalho/citação |
| `pipeline.py` | gera respostas das 8 células (checkpoint incremental, síntese paralela) |
| `run_grid.sh` | orquestra as 8 células no llama-server (guard de porta) |
| `build_judge_input.py` / `analyze.py` | empacota p/ o juiz e produz o **painel de eixos** (ORDEM 2) + tradeoff (ORDEM 1) |
| `gen_device_prompts.py` / `device_bench.py` | PART 2: prompts reais + medição no S24+ (llama-bench + llama-cli, RAM VmHWM, acerto de norma) |
| `consolidate.py` | PART 3: tabela mestre qualidade×custo + recomendação por regras |
| `data/device_questions.jsonl` | roteiro-10 + 10 extra **fora do holdout** (procedência marcada) |
| `data/dense_*.bin` | DVEC1 v4 puxados do S24+ (índices densos embarcados) — paridade |
| `data/responses_*.json` / `data/analysis.json` | respostas por célula + análise |
| `results/device_cost.json` / `results/consolidation.json` | custo no device + tabela mestre |

## Governança

O LLM juiz é **auxiliar**. A leitura humana das 10 primeiras questões do vencedor é
**mandatória** antes de qualquer homologação/embarque (disclaimer herdado de `bench/judge.py`).
Holdout (151) **intocável**: nenhuma calibração usou o holdout; TRIM e grade são determinísticos.
Procedência: task **poc-ctx-topk**, branch `fm/poc-ctx-topk`. Comentários PT-BR, código inglês.
