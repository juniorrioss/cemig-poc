# RESULTS — F7: embarque da melhor versão no S24+

Ordem do capitão: *"colocarmos a melhor versão lá … quero ver funcionando a nossa versão, e
pegue algumas perguntas já pré-prontas do benchmark do modo operacional"*. Configuração
decidida: ponto de operação da F6 (`thr0.7_f2+man1.75+margin` → top-5 → compressão 50%) com
o modelo `f4-r32-ep2` (LFM3-1.7B). O app é EVOLUÇÃO do `android/` da POC (tool-calling
híbrido, Nemotron ASR, index.db v4), não reescrita.

## Resumo

| marco | estado | número |
|---|---|---|
| GGUF Q4_0 do f4-r32-ep2 | ✅ validado | smoke 20 operacionais: **Q4_0 15/20 = bf16 15/20** ≈ HF/F6 15,7/20; pareado Q4×bf16 1/1; tokenização HF×llama.cpp **20/20 idêntica** |
| template LFM3 (XML `<tool_call>`) no app | ✅ byte a byte | 4 fixtures do `chat_template.jinja` do checkpoint (decisão, síntese, multiturno, reuso) |
| retriever do ponto op | ✅ portado | gate 0,7 + boost 2×, manuais w=1,75 com margem densa m=0, RRF k=10 → top-5 |
| compressão de contexto | ✅ portada | janelas fp16 + lexical 0,3 + prior 0,05, gulosa até 50%: **40/40 contextos idênticos byte a byte ao espelho** (21 com manuais) |
| perguntas-exemplo | ✅ | 8 reais do benchmark (`operational`): 5 norma + 3 manual, todas aprovadas nas 3 rodadas da F6 |
| APK no aparelho / medições | ⏸️ aguardando o celular | ver §5 |

## 1. Modelo: f4-r32-ep2 → GGUF Q4_0

- **llama.cpp mínimo para LFM3 = `19cce69`** (PR #31252 "add LiquidAI/LFM3-1.7B", 27/10/2026).
  O submódulo do app (`434ddbb`, 09/2026) e o build da Spark (`3057bb6`) eram anteriores.
  Usei **v0.9.0 (`cd959ba`, 04/11/2026)**, que contém o suporte e os fixes seguintes
  (tensor split, NVFP4, spec-dec). **Submódulo `android/llama.cpp` 434ddbb → cd959ba.**
  A API C usada pelo JNI não mudou (build Android ok).
- Merge (Spark, `spark/merge_f4.py`): LoRA `checkpoint-284` (r32 MLP) sobre o merged
  data2x-ep4 → HF bf16, `model.safetensors` md5 `a661ef377131b7fb222884d9f80afbe0`.
  `convert_hf_to_gguf.py` → bf16 (md5 `2fb9f998d1eabeb1875776106b3e5bdc`, 3,47 GB) →
  `llama-quantize Q4_0` (md5 **`d26176d36ed225a9933ad39ad91ece59`**, 1,05 GB). O chat
  template embutido no GGUF é o do checkpoint (= treino).
- **Smoke ANTES de embarcar** (`spark/smoke_gguf.py` + `score_smoke.py`, régua honesta da F6
  com juiz 27B): 20 perguntas `operational` do recorte (14 norma + 6 manual, seed 7), prompts
  IDÊNTICOS aos da célula `op_d50` da F6, mesmo seed 42 e o sampling da série LFM3
  (temp 0,1 / top_k 50 / rep 1,1) em `llama-server` CUDA:

  | | aprovadas | cobertura | alucin. | recusa | eco de tool_call |
  |---|--:|--:|--:|--:|--:|
  | bf16 GGUF | 15/20 | 0,780 | 3 | 1 | 1 |
  | **Q4_0 GGUF** | **15/20** | 0,772 | 2 | 2 | 1 |
  | HF/transformers (F6, média de 3 rodadas, mesmos ids) | 15,7/20 | — | — | — | — |

  Pareado Q4×bf16: 1 ganho / 1 perda. **Q4_0 não custa nada aqui** (ao contrário do custo Q4
  do sft_v2, −9 p.p.). O "eco de tool_call" é 1 pergunta em que o modelo, em vez de
  sintetizar, re-chama a ferramenta (multihop aprendido na F4); o app limita a 1 busca/turno
  e descarta a chamada (`Lfm3ToolFormat.cleanAnswer`).
  Artefatos: `data/smoke/{prompts_smoke20.jsonl, smoke20_gen.jsonl, scored_smoke20.jsonl, summary_smoke20.json}`.

## 2. Template e parser LFM3 no app (`Lfm3ToolFormat.kt`)

- Réplica determinística do `chat_template.jinja` do checkpoint com `enable_thinking=False`:
  generation prompt `<|im_start|>assistant\n<think></think>`, TODO turno assistant histórico
  prefixado com `<think></think>`, chamada XML
  `<tool_call>\n<function=buscar_norma>\n<parameter=consulta>\n…\n</parameter>\n<parameter=nr>\n…|null\n</parameter>\n</function>\n</tool_call>`,
  turno tool `<tool_response>\n…\n</tool_response>`. System BAKED (asset
  `lfm3_tools_system_prompt.txt`, SHA256 `921e6a1f…`, = `tool_schema_f4.baked_system_content`).
- `Lfm3ToolFormatParityTest`: 4 fixtures geradas pelo `apply_chat_template` real do tokenizer
  do checkpoint (`make_fixtures.py --template`, transformers 5.21 na Spark) — inclui `𝐌`
  (fora do BMP), aspas e barra. Parser = inverso de `bench/lfm3_f4/tool_parser.py`.
- A chamada injetada no prompt de síntese é a forma CANÔNICA da F4/F6 (consulta = fala crua,
  `nr` = NR explícita na fala ou `null`), a mesma da régua `op_d50`.

## 3. Retriever do ponto de operação (`HybridRetriever.searchOp`, `F6Fusion`, `ManualIndex`)

Espelho: `bench/lfm3_f5/retriever_f5.py` (`Cfg(conf_thr=0.7, factor=2, manual_w=1.75,
manual_gate="margin", manual_margin=0)`) + `bench/lfm3_f6/e2e_build_f6.py::Builder.cell`.

- BM25 das normas com gate thr 0,7 / boost 2× (antes 0,5 / 5×); override de NR explícita.
- Manuais = base própria `manuals.db` (cópia exata de `bench/lfm3_f5/cache/manuals.db`,
  schema inalterado: `mchunks` + `mchunks_fts`, ids 1.000.000+) + denso próprio
  `manuals_dense.bin` (DVEC1 de `manuals_dense.npz`). Entram como 2 sinais extras no RRF com
  peso 1,75 SÓ quando `max cos(manual) ≥ cos(top-1 denso-texto das normas)` (margem 0).
- RRF k=10, pesos (2, 1, 2, 1,75, 1,75) → top-5. O controle `v4app` (gate 0,5/5×, sem
  manuais) é rodado só para o orçamento da compressão.
- **Divergência corrigida**: `RrfFusion.kt` usava `HashMap` → empates de score saíam em ordem
  arbitrária; o espelho desempata pela ordem de 1ª aparição. Agora `LinkedHashMap`.

## 4. Compressão de contexto (`ContextCompressor.kt`)

Porte de `bench/lfm3_f5/compress.py::Compressor.compress(mode="hybrid", lam=0.3, prior=0.05)`
com as janelas dos manuais da F6 (`compress_f6.py`).

- Índice pré-computado (`export_assets.py`): `units.bin` (36.863 unidades de 3.294 chunks:
  texto, item, `starts_item`, radicais lexicais, custo em tokens da unidade e do cabeçalho —
  os MESMOS valores que o espelho calcula com o tokenizer LFM3) + `units_win3.f16` (janelas-3
  EmbeddingGemma, mmap).
- **fp16, não int8 (divergência deliberada do brief "int8 ~25 MB")**: simulado no espelho
  sobre os 344 contextos `op_d50`: fp32 344/344, **fp16 344/344**, int8 só 332/344 idênticos.
  Paridade é sagrada → fp16 (54 MB, mapeado, não carregado em heap).
- Score = cos(query, janela) com o vetor de query que o app JÁ calcula (EmbeddingGemma do
  retriever) + 0,3·sobreposição de radicais + 0,05/(rank+1); seleção gulosa até
  50% dos tokens do top-2 cru do controle; render com `[…]`, `(item X)` e cabeçalho de fonte.
- O orçamento conta tokens com o tokenizer do PRÓPRIO GGUF (JNI `nativeCountTokens`); o laço
  de corte do espelho (re-render se estourar) está portado, mas **não disparou em nenhum dos
  344** contextos do recorte.

### Paridade app × espelho (`F7ParityTest`, JVM)

40 perguntas do recorte (21 com manuais ligados pelo gate) — fixtures de
`make_fixtures.py --pipeline` (vetor de query, classificador, rankings por sinal, top-5,
orçamento e contexto do espelho, conferido == `prompts_recorte.jsonl` da F6):

| etapa | resultado |
|---|---|
| consulta FTS5 (`sanitizeFts5Query` == `app_fts_query`) | 40/40 |
| classificador (top-1, p1 Δ<1e-5) + modo do gate (thr 0,7) | 40/40 |
| denso dos manuais (top-60) + gate por margem | 40/40 |
| fusão RRF 5 sinais → top-5 | 40/40 |
| **contexto comprimido, byte a byte** | **40/40** |

O BM25/FTS5 em si não roda na JVM (SQLite com FTS5 é nativo Android); é o mesmo código
de produção desde a v3, com paridade já provada (R@2 51,7% nos `.bin` embarcados).

## 5. Engine nativo (JNI) — mudanças

- **Prompts como bytes UTF-8 padrão** (`byte[]`): `GetStringUTFChars` entrega *modified
  UTF-8* (surrogates CESU-8) para caracteres fora do BMP — há 211 `𝐌/𝐀…` no `index.db` —
  e a tokenização divergiria do tokenizer HF. Tokens voltam como bytes e um
  `Utf8Accumulator` junta caracteres multibyte partidos (antes `NewStringUTF` com pedaço
  inválido podia abortar).
- **Sampling da série LFM3**: temp 0,1 / top_k 50 / rep 1,1 / **min_p 0** (antes 0,05 fixo),
  cadeia na ordem do `common_sampler` do llama-server (onde o smoke foi validado) e histórico
  de penalidade alimentado com o prompt (= server/HF).
- `n_ctx` 4096 (LFM3: system ~550 tok + contexto + histórico; o 2048 do LFM2.5 não cabe).
- **Reuso de prefixo decisão → síntese** (`keepPrefix`): o prompt de síntese estende byte a
  byte o de decisão; o engine decodifica só o sufixo (~500 de ~1100 tokens). Ressalva honesta
  (harness CPU da Spark que replica o JNI, `h/jni_harness.cpp`, greedy): o texto gerado com
  reuso é idêntico ao do zero em 3/6 casos; nos outros 3 diverge em poucas palavras
  (decode em lote 512 vs. lote único muda o arredondamento do estado recorrente/conv do
  LFM3 — a mesma família de efeito já vista no LFM2.5). Do-zero×do-zero é determinístico
  6/6. Com temp 0,1 a resposta já é amostrada; o efeito é da mesma ordem. Medido no
  aparelho com A/B (`--ez keep_prefix false`), ver §6.

## 6. Aparelho (S24+) — PENDENTE

APK release (`app-release.apk`, 236 MB; pesos via override externo) pronto; script
`device/deploy_and_measure.sh <serial>` instala, empurra pesos, roda o E2E
(`RUN_F7_E2E`: 8 perguntas-exemplo + followup de reuso + saudação) e puxa log bruto + RAM.
Aguardando o celular aparecer no `adb`.

## 7. UI (mínimo, como pedido)

Paleta CEMIG/ceia (marinho #001a5b, azul #2866e5, teal #0cd4aa) e logo CEIA **já eram as do
app** (o app de referência `vinyteles/cemig-mobile` usa verde #00843d; mantida a nossa,
conforme o pedido). Mudanças: estado inicial com as 8 perguntas-exemplo clicáveis (selo
NORMA azul / MANUAL teal), subtítulo "LFM3 1.7B · Nemotron 3.5 · 36 NRs + 9 manuais",
Modo Debug mostrando o ponto op (manuais on/off, orçamento, tokens do prompt, tempo de
compressão). Nenhuma tela redesenhada.

## Assets (md5) — `data/assets_md5.json`

| asset | md5 | bytes |
|---|---|--:|
| `manuals.db` | `737c76cdbc8d00c656f56d7314d7138d` | 4.980.736 |
| `manuals_dense.bin` | `c6da02eaae1817ac94488c59581ae27f` | 3.359.005 |
| `units.bin` | `4cff21517c278a12fc3f5f27de40e548` | 8.851.465 |
| `units_win3.f16` | `5f9a2b7050476e47f62ed9bfdc5f3d03` | 56.621.581 |
| `lfm3_tools_system_prompt.txt` | `cee5d45f6110823d87a8aaecad732b39` | 2.002 |
| `index.db` (v4, inalterado) | `e90e94bde10a3399183f61d8a7d3d3cb` | 19.099.648 |
| `nr_classifier.bin` (inalterado) | `e2c98584843add31caa9828767ec1c68` | 4.030.439 |
| externo: `lfm3-1.7b-f4-r32-ep2-Q4_0.gguf` | `d26176d36ed225a9933ad39ad91ece59` | 1.049.614.880 |
| externo: `embeddinggemma-300M-qat-Q4_0.gguf` (inalterado) | `cf8b10c14ca6b271d86fb0dda808ef51` | — |

## Reprodução

```bash
# Spark: merge + build llama.cpp v0.9.0 + convert + quantize
bash bench/lfm3_f7/spark/run_convert.sh            # (copiado p/ ~/cemig-poc/lfm3_f7/)
# smoke bf16 x Q4 (2 llama-server :8420/:8421) e régua local
python bench/lfm3_f7/spark/smoke_gguf.py ; classifier/.venv/bin/python bench/lfm3_f7/score_smoke.py
# assets do app + fixtures de paridade
LFM3_TOKENIZER=<merged>/tokenizer.json classifier/.venv/bin/python bench/lfm3_f7/export_assets.py
python bench/lfm3_f7/make_fixtures.py --template   # transformers 5.x (Spark)
classifier/.venv/bin/python bench/lfm3_f7/make_fixtures.py --pipeline
cd android && ./gradlew test assembleRelease        # default: LFM3 + Nemotron
bench/lfm3_f7/device/deploy_and_measure.sh <serial>
```
