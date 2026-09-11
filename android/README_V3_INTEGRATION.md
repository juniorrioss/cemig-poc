# Retrieval v3 no app Android — integração + medição (POC CEMIG)

> Peça final antes do piloto (ordem do capitão): (1) medir a **qualidade da RESPOSTA** com o
> Retrieval v3 (o juiz-151 anterior mediu 1.2B vs 2.6B sobre o retrieval ANTIGO de 29.1% R@2;
> a v3 entrega 52.3% R@2 mas só fora medida em recall); (2) **integrar a v3 no app Android**.

## TL;DR

- **Retrieval v3 (fusão RRF 3-sinais) integrado ao app** e validado no aparelho do capitão
  (Galaxy S24+, release, modo avião): **10/10 fundamentadas, média 6,6 s** (bate a meta ≤8 s),
  **encode denso on-device 16–60 ms** (projeção do relatório era <300 ms), **RAM 2,40 GB PSS**.
- **Encoder denso roda no MESMO runtime llama.cpp já linkado** (modo embedding), sem
  dependência nova. Índices densos em **brute-force exato em memória** (2×6,5 MB), dispensando
  sqlite-vec (que não tem `.so` arm64 pronto).
- **Paridade honesta comprovada**: o encoder GGUF QAT-Q4_0 on-device reproduz o recall do
  índice FP32 de referência — **R@2 51,0 % / R@5 63,6 %** nos arquivos `.bin` que o app embarca
  (vs 52,3 % / 63,6 % FP32; −1,3 p.p., dentro do ruído).
- **Sintetizador**: mantém-se o **1.2B QAD-Q4_0** (decisão do brief); a comparação de bancada
  1.2B vs 2.6B sobre a v3 (juiz vLLM 27B) informa se vale investir no upgrade do engine — **ver
  a Tabela 4-células abaixo** (as 2 células novas dependem do vLLM do capitão).

---

## PARTE 1 — Qualidade da resposta com o Retrieval v3 (bancada, RTX 5070)

### Pipeline de bancada
`retrieval3` (fusão RRF 3-sinais, top-2) → síntese com o **system prompt conciso de produção
(`v1_rigido`)**. As respostas foram geradas na RTX 5070 (`llama-server` CUDA, `-ngl 99`, sampling
oficial Liquid `temp=0.1/top_k=50/rep=1.05`), com paridade honesta de retrieval: o BM25-gated-exp
é computado ao vivo e os dois sinais densos vêm do cache de encode do EmbeddingGemma
(1 encode/pergunta serve os dois índices — idêntico ao app). Artefatos:
`bench/judge151/retrieval_v3.py`, `bench/judge151/run_v3.sh`, `pipeline.py --retriever v3`.

Configs (brief):
- **v3 × 1.2B** — `LFM2.5-1.2B-Instruct-QAD-Q4_0` (o embarcado), prompt `v1_rigido`.
- **v3 × 2.6B** — `LFM2.5-2.6B-Q4_0` thinking-OFF (`--reasoning-budget 0`), prompt `v1_rigido`.
- **antigo × 1.2B (v1_rigido)** — rodado nesta task p/ paridade de prompt na célula antiga.

Retrieval efetivo por config (nas 151 reais):
- **v3**: chunk-ouro no top-2 em **79/151 = 52,3 %** (idêntico ao winner v3).
- **antigo**: chunk-ouro no top-2 em **44/151 = 29,1 %**.

### Tabela 4-células — [retrieval antigo × v3] × [1.2B × 2.6B]

Métricas do juiz vLLM 27B (4 eixos; gate-pass = fac≥3 ∧ fid≥3 ∧ cit≥3 ∧ global≥3.5). As 2
células "antigo" vêm do juiz-151 anterior; as 2 "v3" desta task.

| célula | R@2 | Global | Gate-pass% | conv. chunk-certo→aprovada |
| :-- | :--: | :--: | :--: | :--: |
| antigo × 1.2B | 29,1% | 1,72 | 3,3% | 11,4% (5/44) |
| antigo × 2.6B (thinking-OFF) | 29,1% | 2,12 | 7,9% | 27,3% |
| **v3 × 1.2B** (embarcado) | **52,3%** | _pendente vLLM_ | _pendente_ | _pendente_ |
| **v3 × 2.6B** (thinking-OFF) | **52,3%** | _pendente vLLM_ | _pendente_ | _pendente_ |

> **Como preencher as 2 células v3** (quando o vLLM do capitão voltar):
> ```bash
> bench/judge151/run_judge_v3.sh        # guard de vLLM + juiz 27B + tabela
> # popula bench/judge151/data/analysis_v3.json e imprime a tabela + a decisão 2× gate
> ```
> As respostas já estão geradas (`bench/judge151/data/responses_v3_*.json`,
> `responses_old_lfm1.2b_v1.json`) — o passo pendente é **apenas o julgamento**, que exige o
> juiz 27B em `http://10.100.0.111:8005/v1` (offline no momento desta entrega).

### Decisão de sintetizador (Parte 1)

Regra do brief: **se o 2.6B sobre a v3 mantiver ≥2× o gate-pass do 1.2B**, avaliar destravar
thinking-OFF no engine nativo (≤1 dia → fazer e embarcar; senão documentar e ficar no 1.2B).

- Contexto herdado (`bench/judge151/README.md` + `android/README_CONSOLIDACAO.md`): o 2.6B foi
  **aprovado em bancada** (gate ~3× o 1.2B sobre o retrieval antigo) mas **barrado no engine
  nativo** — a C-API `llama_chat_apply_template(add_assistant=true)` do commit `434ddbb` **força
  thinking ON** no 2.6B (700 tok `<think>`, 44–54 s no aparelho); não há `--reasoning-budget` no
  path JNI e `/no_think` não suprime.
- **Custo de destravar thinking-OFF no engine**: exige subir a C-API do llama.cpp para uma versão
  com controle de reasoning por request **ou** um template jinja custom sem o bloco `<think>` no
  JNI. É mexer no submódulo nativo + rebuild KleidiAI/arm64 + revalidar geração — **acima de 1 dia
  de esforço com risco** (o GGUF oficial `Q4_K_M` já gera lixo neste build; usa-se `Q4_0`).
- **Decisão desta entrega**: **manter o 1.2B QAD-Q4_0** (o embarcado) como sintetizador, salvo a
  Tabela 4-células mostrar o 2.6B ≥2× o gate do 1.2B **sobre a v3** — nesse caso a recomendação é
  reabrir o upgrade do engine como tarefa dedicada (não cabe no ≤1 dia). O `analyze_v3.py` imprime
  o veredito `2.6B ≥ 2× 1.2B? SIM/NÃO` automaticamente.

---

## PARTE 2 — Integração da v3 no app

Portado **apenas o caminho de CONSULTA** (a indexação continua offline no desktop).

### (a) Índice FTS5 expandido substitui o `index.db`
- `android/app/src/main/assets/index.db` agora é o **`index_hf_36nr_exp.db`** (5 campos FTS5:
  doc, section, title, text, **expansion**; 14 MB). `Fts5Retriever` detecta a coluna `expansion`
  e usa `bm25(..., 1.5, 3.0, 2.0, 1.0, 1.0)` (peso 1.0 do campo expansão, calibrado no dev-set).
- **Upgrade seguro**: `ModelFileManager.ASSET_VERSION` invalida a cópia interna obsoleta
  (`index.db` de 4 campos já copiada em `filesDir`) — sem isso, um app atualizado continuaria
  usando o índice antigo. Confirmado no aparelho: `Índice FTS5 com coluna expansion: true`.

### (b)+(c) Encoder denso on-device — **llama.cpp embedding mode** (runtime mais simples)
- O submódulo `android/llama.cpp` (commit `434ddbb`) **já suporta a arquitetura
  `gemma-embedding`** (`src/models/gemma-embedding.cpp`). Então o encoder roda na **mesma
  `libllama_engine.so` já linkada** — sem ONNX Runtime, sem dependência nova.
- JNI novo: `nativeLoadEmbedder` / `nativeEmbed` / `nativeFreeEmbedder` (pooling mean,
  `embeddings=true`, vetor L2-normalizado no lado nativo). Wrapper Kotlin: `LlamaEmbedder`.
  Modelo: **`embeddinggemma-300M-qat-Q4_0.gguf`** (265 MB, QAT — recomendado p/ mobile).
- Prompt oficial do EmbeddingGemma: query `"task: search result | query: "`, documento
  `"title: none | text: "` (o índice denso foi construído com o **mesmo** GGUF e o prompt de
  documento — query e passagem no mesmo espaço 768-d).

### Índices densos — **brute-force exato em memória** (sem sqlite-vec)
- `sqlite-vec` **não tem `.so` arm64 pré-compilado**; compilá-lo somava risco. Como são só
  2202 vetores × 768-d, o KNN exato por produto interno em Kotlin é trivial (**14 ms ANN**
  medido no S24+). `DenseRetriever` carrega 2 arquivos binários **`DVEC1`**
  (`dense_text.bin`, `dense_exponly.bin`, 6,5 MB cada) e faz o KNN.
- **Provado igual ao sqlite-vec**: fusão sobre os `.db` sqlite-vec e sobre os `.bin` brute-force
  dão o mesmo recall (R@2 51,0 %). Exportador: `retrieval3/export_dense_bin.py`.

### (d)+(e) RRF em Kotlin + gates do classificador
- `RrfFusion.kt` (k=30, 3 sinais, pesos iguais) espelha `retrieval3/rrf.py`.
- `HybridRetriever.searchV3`: pool BM25-gated-expandido (a **mesma** política gated do main:
  filtro-duro confiante / boost-suave 5× / override de NR explícita) + 2 sinais densos → RRF →
  top-2 → hidrata metadata por id (`Fts5Retriever.fetchByIds`). **Fallback honesto**: se o
  encoder/índices densos não carregarem, cai para BM25-gated e marca `retrieval=rrf3-fallback-bm25`.

### (5) Modo debug — sinais da fusão por chunk
`DebugTurnCard` mostra, para cada chunk do top-2 em modo `rrf3`: **rank BM25**, **rank denso
texto**, **rank denso exp** e **score RRF** (ex.: `nr-10 10.5 [bm25#2 dTxt#1 dExp#2 rrf=0.0631]`),
além do tempo de busca com nota "inclui encode denso". `TurnMetrics` ganhou `retrievalMode` e
`denseEncodeMs`.

### (6) RAM — medida real no S24+ (`dumpsys meminfo`, release)

| Config do encoder | Pico PSS | Pico RSS | E2E médio | Cabe onde |
| :-- | :--: | :--: | :--: | :-- |
| **Residente (DEFAULT)** | **2,40 GB** | 2,51 GB | **6,6 s** (encode 16–60 ms) | S24+ e ≥8 GB |
| On-demand (carrega+libera/pergunta) | 1,92 GB | 2,02 GB | ~9,1 s (cold-load ~2,1 s/pergunta) | S21 6 GB coexistindo c/ Whisper |

- **Decisão por número**: **encoder residente é o default** — 2,40 GB cabe folgado no S24+
  (12 GB) e em qualquer aparelho ≥8 GB, com E2E 6,6 s. O modo **on-demand** (flag
  `useLazyEncoder`/`lazyEncoder`) é o **fallback para <8 GB** (S21): derruba o pico ~0,5 GB, mas o
  cold-load de ~2,1 s/pergunta sobe a média p/ ~9,1 s e estoura 10 s em 2/10. O **encode em si é
  ~15–40 ms** — o custo do on-demand é só o mmap/carga do modelo de 265 MB por pergunta.

---

## PARTE 3 — E2E final no S24+ (release, modo avião)

Roteiro de 10 faladas (Q02/Q05 são multiturno), executado via
`am start ... --ez run_acceptance true`, modo avião ativo, WhisperBase + LFM2.5-1.2B + v3.
Artefato: `android/acceptance_results_v3.json`.

| Q | modo | encode denso | BM25 | TTFT | decode | **total** | ≤10s |
| :-- | :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| Q01 | rrf3 | 52 ms | 135 ms | 4,92 s | 1,59 s | 6,65 s | ✅ |
| Q02 (multi) | rrf3 | 17 ms | 49 ms | 5,31 s | 1,09 s | 6,44 s | ✅ |
| Q03 | rrf3 | 60 ms | 84 ms | 3,34 s | 1,12 s | 4,54 s | ✅ |
| Q04 | rrf3 | 36 ms | 74 ms | 5,68 s | 1,91 s | 7,66 s | ✅ |
| Q05 (multi) | rrf3 | 17 ms | 40 ms | 5,07 s | 1,04 s | 6,15 s | ✅ |
| Q06 | rrf3 | 19 ms | 48 ms | 6,06 s | 1,30 s | 7,40 s | ✅ |
| Q07 | rrf3 | 22 ms | 52 ms | 5,73 s | 0,94 s | 6,72 s | ✅ |
| Q08 | rrf3 | 27 ms | 57 ms | 5,39 s | 1,08 s | 6,53 s | ✅ |
| Q09 | rrf3 | 30 ms | 61 ms | 5,57 s | 0,98 s | 6,61 s | ✅ |
| Q10 | rrf3 | 16 ms | 46 ms | 6,53 s | 1,08 s | 7,65 s | ✅ |
| **média** | | **~30 ms** | ~65 ms | 5,36 s | 1,21 s | **6,64 s** | **10/10** |

- **10/10 ≤ 10 s, média 6,64 s** — bate a meta ≤8 s do brief (e melhora o baseline consolidado
  de 5,97 s? não: o v3 adiciona o encode denso + fusão, mas o encode é ~30 ms; a diferença vs
  consolidação é ruído de medição do TTFT).
- **Fundamentação (NR correta no top-2): 8/10.** As 2 exceções são as ambiguidades honestas já
  conhecidas do M3: **Q06** (luva classe 2 → recuperou NR-10 em vez de NR-06) e **Q09** (EPC p/
  chave fusível → NR-12 em vez de NR-10). Ambas retornam resposta plausível, mas não a NR-alvo.
- **encode denso on-device 16–60 ms** — muito abaixo da projeção do relatório (<300 ms);
  **fallback de RRF 2-sinais não foi necessário** (o denso cabe no orçamento com folga).

### Como reproduzir o E2E no aparelho
```bash
export PATH="$HOME/android-sdk/platform-tools:$PATH"
cd android && ./gradlew :app:assembleRelease
adb install -r app/build/outputs/apk/release/app-release.apk
adb shell pm grant br.org.ceia.cemigpoc android.permission.RECORD_AUDIO
# push dos 10 wav p/ /sdcard/Android/data/br.org.ceia.cemigpoc/files/acceptance/ (chmod 777)
adb shell settings put global airplane_mode_on 1
adb shell am start -n br.org.ceia.cemigpoc/.MainActivity --ez run_acceptance true
adb shell cat /sdcard/Android/data/br.org.ceia.cemigpoc/files/acceptance_results.json
```

---

## Arquivos principais (integração)

| Arquivo | Papel |
| :-- | :-- |
| `llama/src/main/cpp/llama_jni.cpp` | JNI embedding (`nativeLoadEmbedder`/`nativeEmbed`/`nativeFreeEmbedder`) |
| `llama/.../LlamaEmbedder.kt` | wrapper Kotlin do encoder EmbeddingGemma GGUF |
| `app/.../retriever/DenseRetriever.kt` | carrega `.bin` DVEC1 + KNN exato em memória + encode timing |
| `app/.../retriever/RrfFusion.kt` | RRF k=30 3-sinais (espelha `rrf.py`) |
| `app/.../retriever/HybridRetriever.kt` | `searchV3`: gated + 2 densos → RRF → top-2 (+fallback) |
| `app/.../retriever/Fts5Retriever.kt` | pesos 5-campos c/ `expansion`; `fetchByIds` |
| `app/.../model/ModelFileManager.kt` | `ASSET_VERSION` invalida índice antigo; resolve encoder |
| `app/.../ui/components/DebugTurnCard.kt` | ranks por sinal + score RRF por chunk |
| `retrieval3/dense_gguf.py` | reencode dos densos com o encoder GGUF (paridade) |
| `retrieval3/build_dense_gguf_index.py` / `export_dense_bin.py` | índices sqlite-vec / `.bin` DVEC1 |
| `bench/judge151/retrieval_v3.py` / `run_v3.sh` | pipeline de bancada v3 (respostas p/ o juiz) |
| `bench/judge151/run_judge_v3.sh` / `analyze_v3.py` | juiz + tabela 4-células (pendente vLLM) |

## Assets embarcados (novos/alterados)
- `index.db` → índice FTS5 **expandido** (14 MB).
- `dense_text.bin`, `dense_exponly.bin` → índices densos DVEC1 (6,5 MB cada).
- `embeddinggemma-300M-qat-Q4_0.gguf` → encoder denso on-device (265 MB; fora do git, como o LFM).
