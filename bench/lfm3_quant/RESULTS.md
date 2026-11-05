# RESULTS — retenção de qualidade por formato de quantização (f4-r32-ep2)

Pergunta do capitão: *"avaliação de precisão entre os modos de quantização em comparação ao
bf16: qual retém mais performance para o nosso cenário? q4 em gguf ou litert? o formato de
gpu e cpu no litert é o mesmo?"*

Cenário: o pipeline RAG de NRs no ponto de operação da F6 (`thr0.7_f2+man1.75+margin` → top-5
→ compressão 50%), com o `f4-r32-ep2` mesclado (merged md5 `a661ef37…`, o mesmo do APK da F7).
Juiz 27B e régua honesta da F6. Tabela-mestra: `data/consolidation.json`.

## Resposta curta

1. **No GGUF, quanto mais bits, mais qualidade retida, e o Q4_0 do APK é o que mais perde:
   −4,9 p.p. contra o bf16 transformers (IC95 −8,9 a −1,2; sinal p=0,007).** O Q8_0 (−1,7) e
   o Q4_K_M (−2,3) ficam dentro do ruído. O Q4_K_M tem praticamente o tamanho do Q4_0 (1,10
   contra 1,05 GB) e retém +2,6 p.p. a mais (p=0,095, sugestivo mas não significativo).
2. **No LiteRT, quem decide a retenção é o backend, não a quantização.** O mesmo arquivo
   retém tudo na GPU e perde ~4 p.p. na CPU: int4 GPU 68,0 contra CPU 64,4 (Δ +3,6, p=0,008);
   int8 GPU 66,7 contra CPU 62,8 (Δ +3,9, p=0,004). O motivo está documentado na fonte (§3).
3. **GGUF Q4 contra LiteRT, no caminho que importa (CPU do celular): empate.** LiteRT int4 CPU
   64,4 contra GGUF Q4_0 63,1 (+1,3, IC95 −2,5 a +5,1, p=0,46) e contra Q4_K_M 65,7 (−1,3,
   p=0,47). O LiteRT só retém mais na GPU, e a GPU do S24+ decodifica a 11 tok/s (−63% contra
   o llama.cpp, medido pelo scout `lfm3-litert`). **O LiteRT não ganha em qualidade onde poderia
   ser embarcado, então não há troca.**
4. **O arquivo `.litertlm` é UM só para CPU e GPU** (mesmo `TFLiteModel` prefill+decode). O
   backend é escolhido em runtime. Mas a quantização não age igual nos dois caminhos: na CPU
   (XNNPACK) os pesos int4/int8 entram em kernels de *dynamic range quantization*, que
   quantizam as ativações para int8 a cada chamada. Na GPU os pesos são desquantizados e a
   conta roda com ativações fp16. Cada backend também gera um cache próprio no primeiro uso.

**Recomendação de qualidade:** manter GGUF. Se for trocar algo, o candidato é **Q4_0 → Q4_K_M**:
mesmo runtime, +50 MB, +2,6 p.p. retidos, recusa de norma volta ao nível do bf16. Mas o Q4_0
tem kernels KleidiAI/repack dedicados no ARM. **Antes de trocar, medir o decode do Q4_K_M no
S24+.** Na Spark, a CPU com 6 threads caiu só 5% (61 contra 65 tok/s), mas isso não prova nada
para o Exynos. O go/no-go de runtime fica com o scout litert, que já deu NO-GO.

## 1. Tabela-mestra (recorte 344, op_d50, médias de rodadas)

Aprovação = cobertura ≥ 0,5 ∧ ¬tautologia ∧ ¬alucinação ∧ ¬recusa ∧ ¬truncada. Δ é pareado
por pergunta (média das rodadas de cada lado), com IC95 bootstrap; w/l e p vêm do teste do
sinal rodada a rodada. Decode t/s: single-stream na Spark GB10, máquina ociosa (`data/speed/`).

| formato | GB | aprov. geral (rodadas) | **aprov. operacional** | norma | manual | alucin. | recusa | cita item-ouro | **Δ vs bf16 HF** [IC95] (p) | Δ vs GGUF bf16 | decode GPU | decode CPU 6t |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **bf16 transformers** (teto) | 3,46 | **68,0** (67,7 / 67,4 / 68,9) | 65,1 | 75,9 | 53,3 | 13,2 | 15,4 | 68,1 | — | +1,2 | 42,0 |  — |
| GGUF bf16 | 3,47 | 66,9 (66,0 / 67,7) | 66,3 | 75,7 | 50,4 | 15,0 | 13,2 | 65,0 | −1,2 [−4,2, +1,8] (0,67) | — | 67,4 | 5,8 |
| GGUF Q8_0 | 1,85 | 66,3 (66,0 / 66,6) | 64,0 | 73,7 | 52,5 | 15,0 | 13,9 | 65,7 | −1,7 [−4,8, +1,3] (0,40) | −0,6 | 116,6 | 54,3 |
| GGUF Q4_K_M | 1,10 | 65,7 (65,7 / 65,7) | 62,8 | 72,8 | 52,5 | 14,5 | 15,0 | 64,8 | −2,3 [−5,9, +1,2] (0,23) | −1,2 | 158,2 | 61,4 |
| **GGUF Q4_0 (APK)** | **1,05** | **63,1** (63,1 / 63,1) | 62,2 | 69,6 | 50,8 | 14,7 | 17,9 | 64,3 | **−4,9 [−8,9, −1,2] (0,007)** | **−3,8** [−7,3, −0,3] (0,019) | 184,0 | 64,6 |
| GGUF Q4_0, rep 1,0 (controle) | 1,05 | 65,4 (1 rodada) | 62,8 | 71,4 | 54,2 | 13,4 | 18,0 | 67,0 | −2,6 [−6,2, +1,3] (0,34) | −1,5 | = | = |
| LiteRT int8 GPU | 1,80 | 66,7 (67,4 / 66,0) | **67,4** | 75,9 | 49,6 | 13,2 | 15,1 | 67,5 | −1,3 [−4,4, +1,6] (0,58) | −0,2 | 80,4 | — |
| LiteRT int8 CPU | 1,80 | 62,8 (62,2 / 63,4) | 58,7 | 70,8 | 47,9 | 17,6 | 15,8 | 65,8 | **−5,2 [−8,6, −2,0] (0,001)** | −4,1 | — | 48,7 |
| **LiteRT mixed int4 GPU** | 1,44 | **68,0** (67,4 / 68,6) | 62,8 | 76,1 | 52,9 | 13,9 | 13,2 | 66,8 | **±0,0 [−2,9, +2,8] (0,83)** | +1,2 | 90,1 | — |
| LiteRT mixed int4 CPU | 1,44 | 64,4 (64,0 / 64,8) | 64,0 | 71,0 | 52,1 | 16,1 | 15,4 | 65,5 | **−3,6 [−6,7, −0,4] (0,025)** | −2,5 | — | 56,6 |

**Piso de ruído** (o mesmo formato, rodada contra rodada): |Δ| ≤ 1,7 p.p. em todos os formatos;
o bf16 HF dá +0,3/−1,2/−1,5 entre as 3 rodadas (`noise_intra`). Diferenças de ±2 p.p. entre
médias não são sinal. As de 4–5 p.p. com p<0,03 são.

**Operacional (a régua-alvo do capitão, n=86 por rodada):** todos os formatos ficam entre 58,7
e 67,4%. O IC95 do Δ vs bf16 no recorte operacional tem largura de ~±7 p.p., então **o
operacional sozinho não separa os formatos** (nenhum Δ operacional é significativo). Duas
leituras ainda assim:
- o pior é o mesmo do geral, **LiteRT int8 CPU (58,7%, Δ −6,4 [−13,2, +0,4])**;
- no Q4_0, a perda operacional (−2,9) é menor que a geral (−4,9) e vem de norma
  (operacional-norma 71,4 contra 73,8 do bf16), não de manual.

Contas pareadas extras (`consolidate_quant.paired`):

| comparação | Δ p.p. [IC95] | w/l | p |
|---|--:|--:|--:|
| LiteRT int4 **GPU vs CPU** (mesmo arquivo) | **+3,6** [+0,9, +6,5] | 54/29 | **0,008** |
| LiteRT int8 **GPU vs CPU** (mesmo arquivo) | **+3,9** [+0,9, +7,1] | 55/28 | **0,004** |
| LiteRT int4 CPU vs GGUF Q4_0 | +1,3 [−2,5, +5,1] | 63/54 | 0,46 |
| LiteRT int4 CPU vs GGUF Q4_K_M | −1,3 [−5,1, +2,6] | 58/67 | 0,47 |
| LiteRT int4 GPU vs GGUF Q4_0 | +4,9 [+1,2, +8,7] | 81/47 | 0,003 |
| GGUF Q4_K_M vs Q4_0 | +2,6 [−0,7, +6,1] | 61/43 | 0,095 |
| GGUF Q8_0 vs Q4_0 | +3,2 [−0,3, +6,7] | 67/45 | 0,047 |
| GGUF Q4_0 rep 1,0 vs rep 1,1 | +2,3 [−1,0, +5,8] | 25/17 | 0,28 |

### 1.1 Onde cada formato perde (modo de falha)

| formato | norma: aprov. / alucin. / **recusa** | manual: aprov. / **alucin.** / recusa |
|---|--:|--:|
| bf16 HF | 75,9 / 12,5 / 8,9 | 53,3 / 14,4 / 27,5 |
| GGUF Q4_K_M | 72,8 / 11,8 / 12,7 | 52,5 / 19,6 / 19,2 |
| **GGUF Q4_0** | 69,6 / 14,3 / **15,0** | 50,8 / 15,4 / 23,3 |
| LiteRT int4 GPU | 76,1 / 11,8 / 9,2 | 52,9 / 17,9 / 20,8 |
| **LiteRT int4 CPU** | 71,0 / 13,6 / 12,7 | 52,1 / **20,8** / 20,4 |
| **LiteRT int8 CPU** | 70,8 / **16,5** / 10,5 | 47,9 / **19,6** / 25,8 |

- **O Q4_0 perde por RECUSA nas perguntas de norma** (8,9 → 15,0%). Ele fica mais "inseguro"
  com contexto bom. A cobertura média quase não cai (0,74 → 0,72). O Q4_K_M devolve metade
  dessa recusa (12,7%).
- **O LiteRT CPU perde por ALUCINAÇÃO** (int8 norma 16,5%, int4 manual 20,8%). As ativações
  quantizadas dinamicamente distorcem o conteúdo, enquanto o Q4_0 distorce a confiança.
- Cobertura (0,72–0,75), citação de NR (97–99%) e item-ouro (64–68%) mudam pouco entre
  formatos. A quantização mexe na decisão de responder e na fidelidade, não na forma da
  resposta.
- Tamanho da resposta: 98–104 tok em todos os formatos. Nenhum formato alonga, trunca ou entra
  em laço com rep 1,1. Truncadas: 0 em todos (768 max).

## 2. Protocolo (o que é idêntico e o que muda)

- **Contexto congelado:** os 344 prompts `op_d50` da F6 (`data/prompts_op344.jsonl` =
  `bench/lfm3_f6/data/e2e/prompts_recorte.jsonl.gz` filtrado). Foram renderizados UMA vez com o
  chat template HF do merged (`enable_thinking=False`, `spark/render_prompts.py`), e esse
  texto exato vai para todos os motores. **Paridade de tokenização com o HF: 344/344 em todos
  os 17 conjuntos de geração** (llama.cpp `/tokenize` e `Engine.tokenize` do LiteRT).
- **Sampling igual:** temp 0,1 / top-k 50 / top-p 1 / repetition penalty 1,1 (janela 64 nos
  motores on-device, o mesmo do JNI do app), seeds 42 e 43, máx. 768 tokens. O controle
  "Q4_0 rep 1,0" mede o peso da penalty. A diferença dele contra o Q4_0 normal é de
  +2,3 p.p., não significativa, então a penalty não explica a perda do Q4_0.
- **Referência bf16 transformers:** as 3 rodadas `op_d50` da F6 (batch 8, `generate` HF),
  sem regerar. O bf16 também foi rodado em GGUF com o mesmo motor e sampler dos quantizados.
  Os dois tetos empatam (−1,2, p=0,67), então tanto faz qual dos dois serve de referência.
- **Juiz:** Qwen3.8-27B-FP8, o mesmo worker do `score_f6.py`, chamado por `score_quant.py`
  sem cópia de régua.
- **Onde rodou:** tudo na Spark GB10. llama.cpp v0.9.0 (`cd959ba`, build da F7) via
  `llama-server` CUDA, com `--parallel 4`. LiteRT-LM 0.19.0 Python, no venv do scout: CPU =
  XNNPACK com 6 threads, 2 processos; GPU = delegate WebGPU/Vulkan (ML Drift). O LiteRT int4 é
  a **conversão do scout** (`f4-r32-ep2_mixed_int4_ekv2048.litertlm`, 1.442.191.536 B,
  reusada). O **int8 foi convertido aqui** com a mesma receita (`build_lfm3_1_7b.py --files
  int8_2048`, H100 só CPU, 19 min, 1.802.113.088 B, sha256 `72692ee9…`), porque o
  `litert-converter` não tem wheel aarch64. Q8_0 e Q4_K_M foram quantizados do GGUF bf16 da
  F7 (md5 `054b257d…` e `da25fdf5…`).

### 2.1 LiteRT-LM: armadilhas que contaminariam a medição (evitadas)

1. **Estado vazando entre conversas** (issue #3165, reproduzida aqui). Reusar a `Engine` entre
   prompts muda a resposta do prompt seguinte: a mesma pergunta respondeu diferente depois de
   outra conversa na mesma Engine, e o mesmo acontece com `Session`. **Por isso usamos uma
   Engine nova por prompt** (init ~2,5 s com cache). Com Engine nova, a saída é
   determinística por seed (verificado).
2. **A `Session` (prompt cru) não aceita repetition penalty.** Sem ela, o int4 entra em laço
   (`"NR-35 35.5.1.1.1.1.1…"`). A saída foi a `Conversation` com **template identidade**:
   como o texto já vem renderizado, `render_message_to_string` confere byte a byte (344/344) e
   o runtime só acrescenta o BOS. Com isso a penalty 1,1/janela 64 fica disponível pela
   `RepetitionPenaltyConfig`. A primeira tentativa, com Session e Engine reusada, foi
   descartada inteira (`gen_old/` na Spark).

## 3. Pergunta técnica: o arquivo LiteRT é o mesmo para CPU e GPU?

**Sim, o arquivo é um só. Não, a quantização não age igual nos dois caminhos.** Evidências:

- **Arquivo único:** `litert-lm-peek` nos dois `.litertlm` mostra 4 seções:
  `LlmMetadataProto`, `ExecutorMetadataProto`, `HF_Tokenizer_Zlib` e **um único `TFLiteModel`
  (`model_type: tf_lite_prefill_decode`)**. Não há variante por backend. O backend é argumento
  de runtime: `Engine(model, backend=Backend.CPU()|Backend.GPU())` em Python;
  `EngineConfig(backend = Backend.CPU() // Or Backend.GPU() and Backend.NPU(...))` em Kotlin
  (`LiteRT-LM/docs/api/kotlin/getting_started.md`). O card da conversão comunitária roda o
  mesmo arquivo com `--backend gpu` e `--backend cpu`. A exceção é a NPU: ela pode usar
  arquivo AOT próprio, e o card diz "An NPU file is not part of this release".
- **O que é por backend é o cache, não o modelo.** No primeiro uso cada backend gera seus
  artefatos ao lado do modelo (`runtime/executor/executor_settings_base.h:205-214`): CPU
  `<modelo>.xnnpack_cache` (1,4 GB, pesos re-empacotados para o XNNPACK); GPU
  `<modelo>_<id>_mldrift_weight_cache.bin` (1,35 GB) e `_mldrift_program_cache.bin` (22 MB).
  Os três ficaram visíveis em `~/cemig-poc/lfm3_quant/lrt/` na Spark.
- **Por que a quantização afeta diferente:**
  - O recipe quantiza os pesos com `"compute_precision": "INTEGER"` e
    `"explicit_dequantize": false`, sem `activation_tensor_config` (`recipe_q1b.json` e
    `dynamic_wi8_afp32_recipe.json`). Pelo `ai-edge-quantizer` isso é **dynamic range
    quantization (DRQ)** (`common_utils.py:1146`, "Check if DRQ"). O README do quantizador
    diz: *"activations remain in a float format… The runtime kernel handles the on-the-fly
    quantization of these activations, as identified by compute_precision=integer and
    explicit_dequantize=False… Cons: on-the-fly quantization of activation tensors can affect
    model quality. Not supported in all hardware (e.g., some GPU and NPU)."*
  - **CPU:** o LiteRT-LM liga `TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED`
    (`runtime/executor/litert_compiled_model_executor_utils.cc:1042-1049`), e o
    FullyConnected roda em kernels int8 com ativações quantizadas por chamada.
  - **GPU:** a precisão padrão é FLOAT16 (`llm_litert_compiled_model_executor.cc:2074`,
    `GpuOptions::Precision::kFp16` em `llm_executor_settings_utils.cc:91-95`). O card confirma:
    "No file declares an activation type, so on a GPU LiteRT-LM runs them with fp16
    activations". Os pesos int4/int8 são desquantizados e a conta roda em fp16.
  - **Medido aqui:** a GPU retém **+3,6/+3,9 p.p.** a mais que a CPU com o mesmo arquivo
    (p≤0,008), nas duas quantizações. O card da conversão comunitária mostra o mesmo em
    concordância de top-token com o fp32: int8 95,7% na GPU contra 88,7% na CPU; int4 79,9%
    contra 77,9%.
- **Ressalva:** a GPU da Spark (WebGPU/Vulkan, NVIDIA) não usa o mesmo kernel da GPU do S24+
  (OpenCL ML Drift, Xclipse). A retenção da GPU no aparelho não foi medida aqui. Ela é
  irrelevante para o embarque, porque essa GPU decodifica a 11 tok/s (scout). Já a CPU usa
  XNNPACK DRQ nos dois lados, e espera-se a mesma perda.

## 4. Recomendação de embarque (QUALIDADE; o runtime é com o scout litert)

| opção | retenção vs bf16 | custo | veredito de qualidade |
|---|--:|---|---|
| GGUF Q4_0 (hoje) | −4,9 (sig.) | 1,05 GB, mais rápido em tudo | aceitável, mas é o que mais perde entre os GGUF |
| **GGUF Q4_K_M** | −2,3 (ruído) | +50 MB; decode na Spark −14% (GPU) / −5% (CPU) | **melhor troca dentro do GGUF, condicionada a medir o decode no S24+** |
| GGUF Q8_0 | −1,7 (ruído) | +0,8 GB, decode CPU −16% | qualidade ≈ bf16, caro em RAM/disco para o S24+ |
| LiteRT int4 CPU | −3,6 (sig.) | +0,39 GB + cache XNNPACK de 1,4 GB, decode −24% no S24+ | **não supera o Q4_0** (+1,3, p=0,46): não troca |
| LiteRT int4/int8 GPU | ±0 / −1,3 | GPU do S24+ a 11 tok/s | retém o bf16, mas é inviável no aparelho |

**Decisão:** o critério do brief era trocar só se o LiteRT retivesse mais qualidade ou igualasse
a qualidade com velocidade ou RAM muito melhores. **Nenhuma das duas condições foi atendida no
caminho embarcável (CPU). Fica o GGUF.** A alavanca de qualidade que sobrou está dentro do GGUF
(Q4_0 → Q4_K_M, ~+2,6 p.p.) e é decisão do capitão depois de medir no S24+. Reabrir o LiteRT
apenas se a frota passar a ter GPU Adreno rápida (no Spark a GPU int4 empata com o bf16).

## Reprodução

```bash
make -C bench/lfm3_quant prompts                    # célula op_d50 congelada da F6
# Spark (~/cemig-poc/lfm3_quant; copiar spark/*.py|sh + data/prompts_op344.jsonl):
#   python render_prompts.py                         # venv ~/cemig-poc/lfm3_f2/.venv
#   setsid nohup bash run_gguf.sh   > run_gguf.log 2>&1 &   # bf16/Q8_0/Q4_0 ×2 + Q4_0 rep1.0 + llama-bench
#   setsid nohup bash run_q4km.sh   > run_q4km.log 2>&1 &   # Q4_K_M ×2
#   bash run_litert.sh <arquivo.litertlm> int4|int8 gpu|cpu # LiteRT ×2 (venv do scout ~/cemig-poc/lfm3_litert/.venv)
#   bash bench_speed_litert.sh                       # velocidade single-stream (máquina ociosa) + hf_speed.py
# local: copiar gen/*.jsonl -> data/gen/ e bench_*.json/speed_* -> data/speed/
make -C bench/lfm3_quant score consolidate export   # juiz 27B (REGUA_JUDGE_URL)
```

Artefatos:
- `data/gen/gen_<formato>-r<k>.jsonl`: geração crua, com paridade de tokens e decode t/s
  durante a geração (com concorrência, só indicativo).
- `data/e2e/scored_*.jsonl` e `summary_*.json`: régua.
- `data/speed/`: llama-bench, CLI benchmark do LiteRT e `speed_hf_bf16.json`.
- `data/consolidation.json`.
- `data/eval/*.parquet`: 17 parquets formato × rodada, no schema da F6 mais as colunas
  `engine_format`, `round`, `seed`, `tok_parity`, `decode_tps` e `prefill_tps`.

Pesos na Spark (não versionados): `~/cemig-poc/lfm3_quant/f4-r32-ep2-{Q8_0,Q4_K_M}.gguf` e
`lrt/f4-r32-ep2_{mixed_int4,int8}_ekv2048.litertlm`. O int8 também está na H100 em
`/workspace/cemig/litert/out_f4_int8/`.
