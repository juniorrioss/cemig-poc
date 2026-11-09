# RESULTS — F9: promoção do v3s400 (P2 RLVR) a modelo do app (APK v1.1)

Ordem do capitão: promover o campeão do RLVR, o **`v3s400`** (`bench/lfm3_p1p2/RESULTS.md`), a
modelo do app. O formato continua **Q4_0**, por decisão de latência (`bench/lfm3_q4km/`).
Nada muda no motor, no retriever, nos assets nem na UI. Só troca o GGUF.

## Resumo

| marco | estado | número |
|---|---|---|
| merge LoRA v3s400 → f4-r32-ep2 mesclado | ✅ | `model.safetensors` md5 `4213d9eccc54e9c2d6f61e2f9d809a7a` (agregado `d590a0b4…`); template e tokenizer idênticos aos do f4 (md5 `c04953ad…` / `f16e3a32…`) |
| GGUF (llama.cpp v0.9.0 `cd959ba` = submódulo do app) | ✅ | bf16 md5 `df2b7b62…`; **Q4_0 md5 `b43caa0dc68f02fbe0eafd58bb2934b3`**, sha256 `689735f3…92d6d8`, 1.049.614.880 bytes |
| smoke 20 operacionais (protocolo F7, juiz 27B) | ✅ | **Q4_0 15/20 × bf16 16/20** (pareado 0 ganho / 1 perda); tokenização HF×llama.cpp 20/20; controle f4 Q4_0 15/20 (juiz concorda 20/20 com o histórico da F7) |
| APK v1.1 (`versionName 1.1.0`, `versionCode 2`) | ✅ | testes JVM 37/37; sha256 `784dc8bb…0b4e` = `base.apk` puxado do S24+ |
| S24+: 8 perguntas-exemplo | ✅ | **8/8 em texto (2 rodadas) e 8/8 em voz** (régua honesta, juiz 27B); controle f4 na mesma sessão 8/8 |
| S24+: latência / RAM | ✅ | 1º token visível **8,1–8,3 s** (f4 na mesma sessão 8,1 s); total 11,9–12,3 s (f4 11,6 s); voz 15,8 s com o ASR; **pico PSS 3,69 GB** (f4 3,71) |
| HF `juniorrios/automobil-ultra-semnexo` | ✅ | `v1.1-rlvr/` (safetensors + GGUF) e `apk/cemig-assistente-v1.1-rlvr.apk`; tag `v1.0-f4` = estado anterior; card v1.1 |

## 1. Modelo

Merge e conversão foram feitos na **H100, GPU 3** (container `cemig_p1p2_walcy`). A Spark estava
fora da tailnet. Script: `h100/run_promote.sh`.

- `merge_policy.py` (o mesmo do P1/P2): `/workspace/cemig/p1p2/f4-r32-ep2-merged` + LoRA
  `lfm3-p2-rlvr-v3/checkpoint-400` → `/workspace/cemig/f9/v3s400-merged` (bf16). O
  `chat_template.jinja` do checkpoint vai junto e é byte a byte o do f4.
- `convert_hf_to_gguf.py --outtype bf16` → `llama-quantize Q4_0`. Usei o llama.cpp do **mesmo
  commit do submódulo do app** (`android/llama.cpp` = v0.9.0 `cd959ba`), compilado com CUDA sm_90
  dentro do container. Sem imatrix, como na F7.

## 2. Smoke Q4_0 × bf16 (20 operacionais)

O protocolo é idêntico ao da F7 (`h100/smoke_gguf.py` = `bench/lfm3_f7/spark/smoke_gguf.py` com
outro modelo): os mesmos 20 prompts `op_d50` (14 norma + 6 manual), template HF do merged com
`enable_thinking=False`, seed 42 e sampling 0,1/50/rep 1,1, em `llama-server` CUDA. A pontuação
usa a régua honesta da F6 com o juiz 27B (`score_smoke.py`). As respostas do f4 da F7 foram
rejulgadas no mesmo lote como controle do juiz.

| | aprov. | norma | manual | cobertura | alucin. | recusa | eco tool_call | tok |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **v3s400 Q4_0** | **15/20** | 12 | 3 | 0,837 | 3 | 2 | 0 | 122 |
| v3s400 bf16 | 16/20 | 13 | 3 | 0,853 | 2 | 1 | 0 | 120 |
| f4-r32-ep2 Q4_0 (F7) | 15/20 | 12 | 3 | 0,772 | 2 | 2 | 1 | 106 |
| f4-r32-ep2 bf16 (F7) | 15/20 | 12 | 3 | 0,780 | 3 | 1 | 1 | 107 |

- No pareado, v3s400 Q4 × bf16 dá 0 ganho e 1 perda; v3s400 Q4 × f4 Q4 dá 1 ganho e 1 perda. Com
  n=20 o smoke só atesta **ausência de regressão grosseira do Q4**. A qualidade vem da bateria
  completa do P2.
- A cobertura sobe (0,77 → 0,84) e o eco de `<tool_call>` na resposta some (1 → 0). As respostas
  ficam mais longas (106 → 122 tok), como a bateria já mostrava (+14%).

## 3. APK v1.1 e validação no S24+

- Código: há uma nova flag de build, `-Pcemig.lfm3Model={v3s400 (padrão), f4}`, que gera
  `BuildConfig.LFM3_MODEL` e `LLM_MODEL_NAME = lfm3-1.7b-<modelo>-Q4_0.gguf`. O v3s400 só existe
  em Q4_0: o build rejeita `lfm3Quant=q4_k_m` com v3s400. Os rótulos da UI (carregamento e Modo
  Debug) mostram o modelo. A versão vai para **1.1.0 (versionCode 2)**. Como desde a F7, o GGUF
  entra por override externo (`getExternalFilesDir`), não dentro do APK.
- `bench/lfm3_f7/device/deploy_and_measure.sh` ganhou `LFM3_MODEL` (o padrão passa a ser v3s400),
  e `device/run_e2e.sh` acrescenta a pré-condição de energia e o keepalive da `lfm3_q4km`.
- Na mesma sessão rodei **v1.1 texto**, a réplica dele, um **controle f4** (mesmo código, com
  `-Pcemig.lfm3Model=f4`) e a **v1.1 em voz** (WAV TTS da F7 → Nemotron). Por último reinstalei o
  APK de produção v1.1.

| rodada | modelo | aprov. (régua) | cita NR | 1º token visível | total | decode | tok resposta | pico PSS |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| **v11_text** | v3s400 Q4_0 | **8/8** | 7/8 | 8,26 s | 12,26 s | 26,0 t/s | 104 | 3.691 MB |
| v11_text_r2 | v3s400 Q4_0 | **8/8** | 8/8 | 7,94 s | 11,93 s | 26,4 t/s | 105 | 3.695 MB |
| **v11_voice** | v3s400 Q4_0 | **8/8** | 8/8 | 8,82 s (+ ASR 2,5 s) | 15,79 s c/ ASR | 25,8 t/s | 116 | 3.728 MB |
| ctrl_f4 | f4-r32-ep2 Q4_0 | 8/8 | 7/8 | 8,12 s | 11,57 s | 26,5 t/s | 91 | 3.707 MB |

O 1º token visível soma decisão, busca e TTFT da síntese, em média sobre os 8 casos com busca. O
followup reusou o contexto sem nova busca e a saudação não buscou, nas 3 rodadas de texto.

- **Sem custo de latência no 1º token.** O resultado é o esperado: mesma arquitetura, mesmo
  formato e mesmo prompt (8,1 × 7,9–8,3 s). O total sobe 0,4–0,7 s porque o v3s400 responde ~14
  tokens a mais. A RAM é igual.
- Este controle f4 (8,1 s) ficou acima do 6,99 s medido na `lfm3_q4km`. Isso cabe na variação
  entre sessões (térmica e bateria a 32%). Por isso a comparação válida é a desta sessão.
- O APK instalado no fim é o de produção v1.1: o sha256 do `base.apk` puxado do aparelho bate com
  `784dc8bb…`, e o GGUF no aparelho tem md5 `b43caa0d…`. `./gradlew --stop` ao final;
  `low_power` restaurado.

## 4. HF

Repo `juniorrios/automobil-ultra-semnexo`, que é o antigo `cemig-lfm3-1.7b-nr` com redirect:

- **tag `v1.0-f4`**: o estado anterior (raiz = f4-r32-ep2, APK v1.0), criado antes do upload.
- `v1.1-rlvr/`: safetensors bf16 + config/código/tokenizer/template + `merge_md5.json` +
  `cemig-lfm3-1.7b-nr-v3s400-Q4_0.gguf`; `apk/cemig-assistente-v1.1-rlvr.apk`. Commit
  https://huggingface.co/juniorrios/automobil-ultra-semnexo/commit/52a066e9f1e4a1a4c24548e18ab47d0ff7ac1958
  (sha256 do GGUF, do safetensors e do APK conferidos no HF).
- Card v1.1, com a tabela da bateria 1536 + 151 e a nota "modelo do APK v1.1", e `inference.py`
  com `CEMIG_MODEL_VARIANT=v1.1-rlvr`, que baixa da subpasta (o padrão segue a raiz). Commit
  https://huggingface.co/juniorrios/automobil-ultra-semnexo/commit/b88025dadb5957e75abbb35262c27931b0c726e2.
  Espelho: `release/README.md` e `release/inference.py`. O download anônimo pela subpasta foi
  conferido (`hf_hub_download(..., subfolder="v1.1-rlvr")`), assim como o
  `AutoModelForCausalLM.from_pretrained(..., subfolder="v1.1-rlvr")` (Lfm3ForCausalLM, 1,73 B parâmetros).
- A raiz continua com o f4-r32-ep2, para não quebrar quem já usa o quickstart. O modelo do app é o
  `v1.1-rlvr/`.

## Reprodução

```bash
# H100 (container cemig_p1p2_walcy, GPU 3): llama.cpp do submódulo em /workspace/cemig/p1p2/llama.cpp-cd959ba
bash h100/run_promote.sh          # merge -> bf16 GGUF -> Q4_0
bash h100/run_smoke.sh            # 2 llama-server + smoke_gguf.py
classifier/.venv/bin/python bench/lfm3_f9/score_smoke.py
# APK (padrão = v3s400 Q4_0) + E2E S24+
cd android && ./gradlew testReleaseUnitTest assembleRelease && ./gradlew --stop
bench/lfm3_f9/device/run_e2e.sh <serial> <apk> <v3s400-Q4_0.gguf> v11_text
classifier/.venv/bin/python bench/lfm3_f9/score_device.py
```

Pesos fora do git: H100 `/raid/aluno_walcy/cemig_mobile/f9/` (merged, bf16/Q4_0 GGUF); cópia local
`/mnt/d/cemig-scratch/f9/` e `~/models-poc/lfm3_f7/v3s400-Q4_0.gguf`.
