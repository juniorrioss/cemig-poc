# RESULTS — upgrade Q4_0 → Q4_K_M do LFM3 f4-r32-ep2 no S24+

Autorização do capitão: testar a troca do GGUF Q4_0 do APK pelo Q4_K_M. O quant-eval
(`bench/lfm3_quant/`) mediu Q4_0 −4,9 p.p. contra o bf16 (p=0,007) e Q4_K_M −2,3 p.p. (ruído).
A condição era o decode não piorar no aparelho, porque K-quants podem ser mais lentos em ARM.
**Gate: se o decode do Q4_K_M cair mais de 10% contra o Q4_0, a troca para.**

## Decisão: APROVADO e APLICADO (com um custo de prefill que precisa ser visto)

| | Q4_0 (F7) | **Q4_K_M** | Δ |
|---|--:|--:|--:|
| tamanho GGUF | 1,05 GB | 1,10 GB | +50 MB |
| **decode** S24+ (llama-bench, tg300, 3 rodadas) | 29,5 ± 1,2 t/s | **27,7 ± 1,3 t/s** | **−6,0%** (gate −10%: **passa**) |
| prefill S24+ (llama-bench, pp450, 3 rodadas) | 190 ± 6 t/s | 142 ± 5 t/s | **−25,3%** |
| E2E app: TTFT da síntese (média de 8) | 3,79 s | 4,44 s | +0,65 s |
| E2E app: decisão (prefill do system + fala) | 3,85 s | 4,32 s | +0,47 s |
| E2E app: **1º token visível** | 7,75 s | 8,86 s | **+1,1 s** |
| E2E app: total da resposta | 11,3 s | 12,8 s | +1,5 s |
| E2E app: decode | 26,4 t/s | 25,5 t/s | −3% |
| RAM pico PSS (Nemotron + LFM3 + EmbeddingGemma) | 3.700 MB | 3.843 MB | +143 MB |
| smoke 20 operacionais (Spark, régua honesta, juiz 27B) | 15/20 | **16/20** | +1 (1 ganho / 0 perda pareado) |
| 8 perguntas-exemplo no aparelho (régua honesta) | 8/8 | **8/8** | = |

- **O que o gate mede passa com folga**: o decode cai 6% no llama-bench e 3% no app.
- **O que o gate não media custa mais**: o prefill cai 25%. O 1º token visível sobe ~1,1 s
  (7,7 → 8,9 s) e a resposta inteira sobe ~1,5 s. Isso fica dentro da faixa já aceita na F7
  (~8,4 s de 1º token, ~12 s total), mas é uma piora real de latência que o voz sente.
- Qualidade: o ganho de −4,9 → −2,3 p.p. vem do quant-eval (344 perguntas, 2 rodadas). O smoke 20
  (16 vs 15) e as 8/8 só confirmam que não há regressão. Com n=20 o smoke não prova ganho.
- **Aplicado**: o APK agora carrega `lfm3-1.7b-f4-r32-ep2-Q4_K_M.gguf` por padrão. Está instalado
  no S24+ e foi revalidado. Para voltar atrás basta 1 flag de build:
  `./gradlew assembleRelease -Pcemig.lfm3Quant=q4_0`.
  Se o capitão achar que +1,1 s no 1º token pesa mais que +2,6 p.p. de qualidade, é só esse
  rebuild. Os pesos Q4_0 continuam no aparelho e no HF.

## 1. GGUF Q4_K_M

Reaproveitei o arquivo do quant-eval, sem reconverter. Na Spark, ele está em
`~/cemig-poc/lfm3_quant/f4-r32-ep2-Q4_K_M.gguf`. Foi gerado com `llama-quantize Q4_K_M`
(llama.cpp v0.9.0, sem imatrix) a partir do GGUF bf16 do merged `f4-r32-ep2`
(`bench/lfm3_quant/spark/run_q4km.sh`). Tem 1.099.627.040 bytes, 5,05 BPW, md5
**`da25fdf5066d70f594fd6d752aee5f86`** e sha256 `487c5a55…`. O FFN down sai em q6_K e o resto em
q4_K, que é o mix padrão do Q4_K_M. Tokenização HF × llama.cpp: 20/20 idêntica.

## 2. Velocidade no S24+ (SM-S926B, Exynos 2400)

**Harness** (`device/run_bench.sh`): `llama-bench` compilado do **mesmo llama.cpp do app**
(submódulo `android/llama.cpp` = `cd959ba` v0.9.0). Usei as flags de
`android/llama/build.gradle.kts`: `-march=armv8.4-a+dotprod+i8mm+fp16`, KleidiAI ON, sem
OpenMP/llamafile, NDK 26.1, android-31. Os parâmetros são os do JNI (`llama_jni.cpp`): 6
threads, `n_batch=n_ubatch=512`. Cenário: prompt de 450 tokens (o contexto comprimido) e geração
de 300 tokens. Fiz 3 rodadas por formato, **intercaladas e com a ordem alternada** (AB, BA, AB),
com 90 s de resfriamento entre elas para o throttling térmico não favorecer nenhum formato.

| rodada | Q4_0 pp450 | Q4_0 tg300 | Q4_K_M pp450 | Q4_K_M tg300 |
|---|--:|--:|--:|--:|
| 1 | 183,7 | 30,9 | 142,5 | 28,8 |
| 2 | 193,3 | 28,7 | 146,4 | 26,3 |
| 3 | 193,1 | 29,0 | 137,2 | 28,1 |
| **média** | **190,0** | **29,5** | **142,0** | **27,7** |

**Sharp edge: medições invalidadas (guardadas em `data/device/invalid_*`, fora da tabela).**
1. A 1ª série rodou com o aparelho em **economia de energia** (`settings global low_power=1`,
   com os clocks dos núcleos travados em 0,6–2,0 GHz). O decode do Q4_0 oscilou entre 14 e
   31 t/s.
2. Na 2ª série a tela entrou em **Doze** no meio da série, mesmo com o tempo de tela em 30 min.
   O script agora força `low_power=0` e manda `KEYCODE_WAKEUP` a cada 15 s, e o log registra o
   estado de cada rodada. Terminada a medição, restaurei `low_power=1` e o tempo de tela de 30 s.

**Por que o prefill cai mais que o decode**: não medi a causa. O decode é GEMV, limitado por
banda de memória, e acompanha o tamanho do arquivo (+5%). O prefill é GEMM, limitado por
computação, e sente o custo dos kernels. No `cd959ba`, a KleidiAI tem caminho para Q4_0, Q4_K
e Q6_K (este repassado como int8). Os kernels de K-quant (escalas por sub-bloco) e o
`ffn_down` em q6_K→int8 custam mais por token do que o Q4_0 puro. Para separar essas causas
seria preciso perfilar por tensor; não era necessário para esta decisão.

## 3. Smoke de qualidade (20 operacionais)

Protocolo **idêntico** ao da F7 (`spark/smoke_q4km.py` = `bench/lfm3_f7/spark/smoke_gguf.py`
trocando só o GGUF): mesmos 20 prompts `op_d50` (14 norma + 6 manual), seed 42, sampling LFM3
0,1/50/rep 1,1, `llama-server` v0.9.0 CUDA na Spark. Régua honesta da F6 com o juiz 27B
(`score_smoke.py`). As respostas Q4_0 e bf16 da F7 foram **rejulgadas no mesmo lote** como
controle do juiz, e o juiz concordou **20/20** com o histórico nas duas.

| | aprov. | norma | manual | cobertura | alucin. | recusa | eco tool_call |
|---|--:|--:|--:|--:|--:|--:|--:|
| bf16 GGUF (F7) | 15/20 | 12 | 3 | 0,780 | 3 | 1 | 1 |
| Q4_0 GGUF (F7) | 15/20 | 12 | 3 | 0,772 | 2 | 2 | 1 |
| **Q4_K_M GGUF** | **16/20** | 12 | 4 | 0,772 | **0** | 2 | 1 |

Pareado Q4_K_M × Q4_0: 1 ganho, 0 perda. Pareado Q4_K_M × bf16: 2 ganhos, 1 perda.

## 4. APK revalidado no S24+

- Código (`android/`): quant do LFM3 por build com `-Pcemig.lfm3Quant={q4_k_m (padrão), q4_0}`,
  que gera `BuildConfig.LFM3_QUANT` e `LLM_MODEL_NAME = lfm3-1.7b-f4-r32-ep2-<quant>.gguf`. Os
  rótulos da UI (carregamento e Modo Debug) vêm do BuildConfig. Como na F7, o GGUF entra por
  override externo (`getExternalFilesDir`), não dentro do APK. `testReleaseUnitTest`: 37/37.
- Build feito numa cópia scratch em `/mnt/d` (o disco local estava 100% cheio), com
  `./gradlew --stop` no fim. O APK release tem 235 MB.
- E2E (`device/run_e2e.sh` → `bench/lfm3_f7/device/deploy_and_measure.sh`, broadcast
  `RUN_F7_E2E`): as 8 perguntas-exemplo (5 norma + 3 manual) + followup + saudação. Rodei o
  **APK Q4_K_M e o controle Q4_0 na mesma sessão** (mesmo código, só a flag muda) e depois
  reinstalei o Q4_K_M. Resultados: **8/8 aprovadas nos dois**, e 7/8 citam NR nos dois.
  Followup reusou o contexto sem nova busca, e a saudação não buscou, nos dois.
- Fix no script da F7: o broadcast implícito `am broadcast -a …RUN_F7_E2E` **não chega** ao
  receiver de manifesto (Android 8+). Passei a usar `-p br.org.ceia.cemigpoc`. O script também
  aceita `APK`, `LFM3_QUANT`, `LFM3_GGUF`, `OUT` e `ADB`.

## 5. HF

Subi no repo de modelo existente: `cemig-lfm3-1.7b-nr-Q4_K_M.gguf` (sha256 `487c5a55…`,
conferido no HF) e um card atualizado (tabela de arquivos e métricas). **O repo foi renomeado**:
`juniorrios/cemig-lfm3-1.7b-nr` hoje redireciona (307) para
`juniorrios/automobil-ultra-semnexo`. É o mesmo repo, com o mesmo conteúdo e o `Q4_0.gguf` do
release. Commits: https://huggingface.co/juniorrios/automobil-ultra-semnexo/commit/8d39a1babfad8087c049c3cc37239bc48a7fcd10
e https://huggingface.co/juniorrios/automobil-ultra-semnexo/commit/12da9e42284351f6bc8a9393101c7e978dab5f75.
O padrão do `inference.py` do HF **continua Q4_0**: não mexi, para preservar a replicação já
validada do release. O Q4_K_M fica disponível ao lado.

## Reprodução

```bash
# llama-bench Android do submódulo do app (flags do android/llama/build.gradle.kts) -> /data/local/tmp/lfm3q
bench/lfm3_q4km/device/run_bench.sh <serial> 3 90        # pré-condição de energia + keepalive embutidos
# Spark: smoke (copiar spark/* p/ ~/cemig-poc/lfm3_q4km/)
~/cemig-poc/lfm3_q4km/run_smoke.sh ; classifier/.venv/bin/python bench/lfm3_q4km/score_smoke.py
# APK + E2E (Q4_K_M é o default; controle com -Pcemig.lfm3Quant=q4_0)
cd android && ./gradlew assembleRelease && ./gradlew --stop
bench/lfm3_q4km/device/run_e2e.sh <serial> Q4_K_M <apk> <gguf> q4km
classifier/.venv/bin/python bench/lfm3_q4km/score_device.py
python3 bench/lfm3_q4km/consolidate.py                    # -> data/consolidation.json
```
