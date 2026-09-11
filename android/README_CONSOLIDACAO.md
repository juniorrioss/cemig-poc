# Consolidação do app — antes/depois (POC CEMIG)

Consolidação aprovada pelo capitão sobre o APK do `main`. Duas frentes: (1) limpar o
pipeline do app tirando o Turno 1 de rewrite via LLM e mostrar o pipeline real no modo
debug; (2) decidir por número se o sintetizador sobe do 1.2B para o 2.6B conciso.

## TL;DR

- **Turno 1 de rewrite REMOVIDO.** O fluxo agora é `ASR → classificador+gate (HybridRetriever) →
  BM25 com a FALA BRUTA → síntese`. A bancada do classifier já provara que keywords do rewrite
  PIORAM a busca vs fala bruta (22.5% vs 28.5% R@2); o rewrite ainda custava ~1–1.5 s por
  pergunta. O capitão viu "turno 1 rewrite" no debug e estranhou corretamente.
- **Modo debug reescrito para o pipeline real**, com a **decisão do gate VISÍVEL**
  (hard/soft/explicit/none/reuso, NR top-1/top-2 e probabilidades).
- **2.6B: aprovado na GPU, barrado no engine do aparelho.** O experimento de concisão aprovou o
  2.6B thinking-OFF **por número** (gate-pass 9.9% ≈ 3× o 1.2B, 178 tok médios). Mas no aparelho o
  **engine nativo (commit 434ddbb) força thinking ON** no 2.6B (700 tok de `<think>`, 44–54 s por
  pergunta — inviável para voz). Aplicada a **regra de fallback do brief: o 1.2B fica**, agora com
  o **system prompt conciso** herdado do experimento.
- **E2E S24+ final (1.2B): 10/10 dentro do teto de 10 s (média 5,97 s)**, todas citando a NR
  correta. Antes (M3, com rewrite): média **12,52 s**.

---

## PARTE 1 — Limpeza do pipeline (`android/`)

### Antes (main)
```
ASR → [Turno 1: LLM reescreve fala em 3-6 keywords (~1-1.5s)] → BM25(keywords) → síntese
```
- `AskPipeline` chamava `generateComplete` no T1 (REWRITE_SYSTEM_PROMPT), limpava keywords e
  decidia reuso por Jaccard **entre keywords**.
- Modo debug mostrava "2. Turno 1 (Rewrite)" com as keywords e o tempo do T1 — o pipeline antigo.

### Depois (esta consolidação)
```
ASR → classificador+gate (HybridRetriever, FALA BRUTA) → BM25(FALA BRUTA) top-2 → síntese
```
- Removidos `REWRITE_SYSTEM_PROMPT`, `formatRewriteUserPrompt`, `cleanKeywords` e a chamada
  `generateComplete` do T1. `FakeLlm.reformulateKeywords`/`customRewriteOutput` também saíram.
- **Reuso por Jaccard mantido, mas agora compara falas BRUTAS consecutivas** (antes comparava
  keywords do rewrite, que não existem mais). Justificativa: em multiturno, perguntas de
  continuação repetem a fala anterior; o reuso evita nova consulta BM25 quando Jaccard > 0.7.
- `PipelineStage.REWRITING`/`SEARCHING` → **`CLASSIFYING`** (classificação + busca unificadas).
- `TurnEvent.KeywordsExtracted` → **`TurnEvent.Classified`** (top-1/top-2, probs, gate, boost).
- `TurnMetrics`: removidos `rewriteMs`/`keywords`/`keywordsReused`; adicionados
  `nrTop1`/`nrTop1Prob`/`nrTop2`/`gateMode`/`boostNrs`/`chunksReused`.

### Modo debug (Modo Engenharia) — pipeline real
`DebugTurnCard` agora mostra, na ordem do pipeline:
1. **ASR** (Whisper) + tempo + transcrição;
2. **Classificador NR + Gate** — o gate acionado é o destaque: `FILTRO-DURO (prob ≥ 0.5)`,
   `BOOST-SUAVE 5x (top-2)`, `NR EXPLÍCITA NA FALA`, `SEM BOOST (fora de escopo)` ou
   `REUSO JACCARD > 0.7`; com `top1 (prob) · top2 · boost=[...]`;
3. **Busca BM25 top-2 (fala bruta)** — chunks (doc+seção+score) + tempo;
4. **Síntese (LFM2.5)** — contexto tok, TTFT, decode (tok @ tok/s);
5. **Timeline** proporcional (ASR / Classif+BM25 / TTFT / Decode) — sem a faixa roxa de rewrite.

### Testes JVM
`AskPipelineTest` reescrito: multiturno com histórico bruto na síntese; reuso/sem-reuso por
Jaccard entre **falas brutas**; poda de orçamento do T2; sanitização FTS5; telemetria
(`gate_mode` no lugar de `rewrite_ms`). `./gradlew :app:testDebugUnitTest` → verde.

---

## PARTE 2 — Experimento de concisão do 2.6B (`bench/judge151/`)

Objetivo: domar a verbosidade do `LFM2.5-2.6B-Q4_0` thinking-OFF (baseline: 498 tok, markdown com
títulos) para respostas de voz (≤4 frases, sem markdown, citação inline).

### Fase DEV (20 smoke, fora do holdout) — `concision_dev.py`
Métricas objetivas de estilo (sem juiz) nas 4 variantes de `concision_prompts.py`:

| Variante | tok médio | ≤4 frases | sem markdown | cita inline |
| :-- | --: | --: | --: | --: |
| base (produção antiga) | 351.1 | 0.0% | 0.0% | 60.0% |
| **v1_rigido (vencedora)** | **144.0** | **55.0%** | **100.0%** | **75.0%** |
| v2_exemplo | 168.8 | 40.0% | 100.0% | 80.0% |
| v3_radio | 188.7 | 35.0% | 100.0% | 70.0% |

### Fase FINAL (151 reais + juiz vLLM 27B) — `pipeline.py --prompt-key v1_rigido`
Retrieval idêntico ao app (paridade R@2 = 29.1% = 44/151, determinístico):

| Config | Global | Fact | Fidel | Fonte | Gate-pass | tok médio |
| :-- | --: | --: | --: | --: | --: | --: |
| lfm1.2b (embarcado, base) | 1.72 | 1.44 | 1.38 | 1.25 | **3.3%** | — |
| lfm2.6b thinking-OFF (base, verboso) | 2.12 | 1.89 | 1.73 | 1.64 | 7.9% | ~498 |
| **lfm2.6b thinking-OFF (v1_rigido)** | 2.09 | 1.88 | 1.70 | 1.63 | **9.9%** | **178.3** |

**Critério do capitão atendido na GPU:** gate-pass 9.9% ≥ ~2× do 1.2B (≥7%, na verdade 3×) **e**
tokens médios 178 ≤ 200. A concisão manteve a qualidade do 2.6B verboso e **ainda subiu o gate**
(9.9% vs 7.9%) cortando 2.8× os tokens.

### Validação de RAM no aparelho (dumpsys / VmHWM) — pré-selagem
| Métrica (S24+, SM-S926B) | 2.6B-Q4_0 | 1.2B-QAD-Q4_0 |
| :-- | --: | --: |
| `llama-bench` pp800 VmHWM | 3170 MiB (3.10 GiB) | — |
| App completo TOTAL PSS (dumpsys) | ~3.37 GB | ~1.77 GB |
| App completo TOTAL RSS (dumpsys) | ~3.48 GB | ~1.88 GB |
| Prefill pp800 | 108.8 t/s | 243.9 t/s |
| Decode | ~10 t/s | ~48 t/s |

RAM do 2.6B cabe no S24+ (11.47 GB), mas confirma inviabilidade no S21 (6 GB) coexistindo com Whisper.

### ⛔ Porquê o 2.6B NÃO foi embarcado (achado E2E decisivo)
Embarquei o 2.6B, instalei o APK (~1.7 GB) e rodei o roteiro no S24+. **No aparelho o 2.6B roda
com thinking ON:** respostas de 700 tokens cheias de `<think>`, **44–54 s por pergunta**.

Causa raiz: o engine nativo do app (`libllama_engine.so`, commit `434ddbb`) sintetiza via
**C-API `llama_chat_apply_template` com `add_assistant=true`**, que injeta `<|im_start|>assistant\n<think>`
e **não tem equivalente a `--reasoning-budget 0`** (o desligamento do thinking usado no experimento
GPU era via `llama-server --reasoning-budget 0`, um recurso do path jinja do servidor, ausente no
JNI). Testado no aparelho: `--no-jinja` + template exato do app mantém o thinking; `/no_think` no
system **não** suprime; fechar o bloco `<think></think>` manualmente **não** suprime.

Conclusão: **a config vencedora do experimento (2.6B thinking-OFF conciso) não é reproduzível no
engine atual do app.** Pela regra do brief ("se não couber… o 1.2B fica"), **mantido o 1.2B QAD**
(modelo Instruct, não-reasoning; 48 t/s, sem thinking) **com o system prompt conciso v1_rigido**
— a lição de concisão do experimento aproveita ao 1.2B também.

> Trilha futura (fora desta task): habilitar thinking-OFF no JNI (fechar reasoning no template
> nativo ou subir um engine com controle de reasoning) para reabrir o 2.6B no S24+/≥8 GB.

---

## PARTE 3 — E2E final no S24+ (1.2B, release, 100% offline/local)

`android/acceptance_results_consolidation.json` — 10 faladas + 2 multiturno (Q02, Q05).

| Q | total | fits ≤10s | gate | NR top-1 | chunks |
| :-- | --: | :--: | :-- | :-- | :-- |
| Q01 | 6.21s | ✅ | explicit | nr-10 | 10.4, 10.5 |
| Q02* | 6.36s | ✅ | soft | nr-10 | 10.2.8, AnexoIII |
| Q03 | 6.00s | ✅ | soft | nr-10 | AnexoIII, nr-12 |
| Q04 | 4.85s | ✅ | explicit | nr-35 | Glossário |
| Q05* | 5.03s | ✅ | hard | nr-35 | 35.6.2, Glossário |
| Q06 | 5.67s | ✅ | hard | nr-10 | 10.7, Glossário |
| Q07 | 5.59s | ✅ | hard | nr-06 | AnexoI |
| Q08 | 6.60s | ✅ | hard | nr-10 | 10.3, AnexoIII |
| Q09 | 6.33s | ✅ | soft | nr-12 | AnexoXII |
| Q10 | 7.05s | ✅ | hard | nr-10 | 10.7, 10.6 |

`*` = continuação multiturno. **Média 5,97 s; 10/10 dentro do teto de 10 s.** Todas citam a NR
correta (10/10). Deslizes de número de item ainda aparecem (Q05/Q07 copiam o padrão "10.5.1"),
limite honesto do 1.2B — mantém o critério M3 de ~8/10 fundamentadas com a norma certa.

### Antes/depois de latência (S24+)
| | Antes (M3, com rewrite) | Depois (consolidação, sem rewrite) |
| :-- | --: | --: |
| Média total por pergunta | **12,52 s** | **5,97 s** |
| Turno 1 rewrite | 1,0–1,7 s | **0 s (removido)** |
| Perguntas ≤10 s | ~parcial (5/10 acima) | **10/10** |

O ganho supera o rewrite isolado porque as respostas ficaram concisas (system prompt v1_rigido):
decode de ~34–77 tok em vez de gerações longas.

## Como reproduzir
```bash
# PARTE 1 — testes JVM
export JAVA_HOME="$HOME/android-sdk/jdk-17" ANDROID_HOME="$HOME/android-sdk"
(cd android && ./gradlew :app:testDebugUnitTest)

# PARTE 2 — experimento de concisão (RTX 5070 + juiz vLLM)
#   sobe llama-server 2.6B thinking-OFF na porta 8397 (--reasoning-budget 0)
python3 bench/judge151/concision_dev.py --url http://127.0.0.1:8397          # fase DEV (20 smoke)
python3 bench/judge151/pipeline.py --config-key lfm2.6b_noth_v1 \
  --model-file LFM2.5-2.6B-Q4_0.gguf --reasoning off --url http://127.0.0.1:8397 \
  --prompt-key v1_rigido --out bench/judge151/data/responses_lfm2.6b_noth_v1.json --max-tokens 512
python3 bench/judge151/build_judge_input.py --configs lfm1.2b lfm2.6b_noth lfm2.6b_noth_v1 \
  --out bench/judge151/data/harness_concision.json
python3 bench/judge.py --input bench/judge151/data/harness_concision.json \
  --output-eval bench/judge151/data/judge_concision.json \
  --output-csv bench/judge151/data/judge_concision.csv --cli-tool vllm --no-calibration

# PARTE 3 — E2E no aparelho (APK 1.2B ~880 MB)
(cd android && ./gradlew assembleRelease)
adb -s 192.168.0.6:41073 install -r -d android/app/build/outputs/apk/release/app-release.apk
adb -s 192.168.0.6:41073 shell pm grant br.org.ceia.cemigpoc android.permission.RECORD_AUDIO
adb -s 192.168.0.6:41073 shell am start -n br.org.ceia.cemigpoc/.MainActivity --ez run_acceptance true
adb -s 192.168.0.6:41073 pull \
  /sdcard/Android/data/br.org.ceia.cemigpoc/files/acceptance_results.json
```
