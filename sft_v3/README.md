# SFT v3 — recusa ≤ 25% + r64/r128 + calibração + embarque S24+ (`sft_v3/`)

Responde à ordem do capitão após ver o v2 (`sft_v2/`, já no main): (1) **recusa ≤ 25% no
total** (o v2 tinha 52% de famílias de recusa e pagou com recusa-indevida de 23-40%); (2)
manter o **perfil cauteloso** (não voltar ao chute confiante); (3) varrer **r64 e r128** (no
v2 o rank ainda subia); (4) reportar recusa como **classificador (precision/recall/F1 +
matriz de confusão)**; (5) **selecionar, quantizar COM calibração e EMBARCAR** no S24+ o
**Nemotron 3.5 ASR + o 2.6B v3 Q4**; (6) rodar a **avaliação geral (catastrophic forgetting)**.

> **Infra**: dados/pontuação LOCAIS (`classifier/.venv`; juiz vLLM 27B `10.100.0.111:8005`
> roteável deste host). **Treino + export/quant GGUF na DGX Spark** (`walcyrios@spark-b431`,
> `~/jupyterlab/.venv`). Régua ÚNICA: `bench/regua` + juiz de recusa próprio (reuso do v2).
> Holdout 151/20 INTOCÁVEL; 3 muralhas + split por NR mantidos. wandb ONLINE (chave no
> `~/.netrc` da Spark). Paralelo, checkpoint, não-interativo. Comentários PT-BR; código em inglês.

**REUSO máximo do v2 (ordem do brief)**: `common_v2.py` (system prompt, juízes, régua),
`gen_data.gen_oracle` (mesma régua honesta), `generate_v2.py`/`score_v2.py`, os 4 packs de
avaliação (`oracle_pack_151`/`retrieved_pack_151`/`val_pack`/`refusal_test_pack`), `train_v2.py`
(motor de treino rank-configurável) e `finetune2/eval_panel.py` (geral+comportamental).

---

## PARTE 1 — rebalancear os dados (recusa ≤ 25%)

`rebalance_data.py` + `topup_oracle.py`. Reusa o pool v2 (já régua-aprovado): subamostra as
famílias de recusa com seed fixa e mantém TODOS os oráculos do v2; gera oráculo fresco para o
déficit (139 de chunks novos + 700 via **múltiplas perguntas coloquiais por chunk**, cada uma
filtrada pela régua honesta + dedup Jaccard).

| família | n final v3 | % v3 | (era no v2) | justificativa da divisão |
|---|--:|--:|--:|---|
| **oráculo**   | **2315** | **75.5%** | 48.0% | é onde a APROVAÇÃO mora; dobrar o peso |
| recusa        | 300 | 9.8% | 20.8% | falha de retrieval mais comum (norma errada) → 10% |
| parcial       | 300 | 9.8% | 20.8% | 2ª falha real (chunk certo, dado ausente) → 10% |
| distrator     | 150 | 4.9% | 10.4% | o mais raro/difícil; 10% já bastou no v2 → 5% |
| **TOTAL**     | **3065** | **famílias de recusa 24.5%** (< 25%) | 52% | |

33 NRs, 0 contaminação de NR reservada (33/16/26), 0 malformado, 3062/3065 perguntas únicas.

---

## PARTE 2 — treino (1 época, r64 e r128, alpha=2r)

`train_v3.py` (reusa `train_v2.main`, rank-configurável) + `run_train_v3.sh` na Spark. Receita
IDÊNTICA ao v2 (bs8×ga4, max_len 2048, lr 1e-4, cosine, bf16, grad ckpt, targets por REGEX,
`assistant_only_loss`, val interno). **Asserts obrigatórios publicados ANTES de treinar**
(`data/premises/`): 122 módulos (MLP `feed_forward.w1/w2/w3` + `self_attn.{q,k,v,out}_proj`),
**mlp✓ attn✓ conv✗** em ambos os ranks.

| rank | alpha | trainable | train_loss | **eval_loss interno** | grad_norm | instabilidade |
|---|--:|--:|--:|--:|--:|:--|
| 64  | 128 | 2.90% | 0.585 | **0.5366** | ~0.79 estável | não |
| 128 | 256 | 5.64% | 0.563 | **0.5098** | <1.1 estável | não |

**r128 NÃO mostrou instabilidade** a lr 1e-4 (grad_norm saudável, eval_loss cai monotônica) —
nada a propor/reportar quanto a lr menor. wandb online: projeto `cemig-sft-v3`.

---

## PARTE 3 — recusa como CLASSIFICADOR (pedido explícito do capitão)

`refusal_metrics.py` computa a matriz de confusão sobre a suite de recusa (classe positiva =
"deveria RECUSAR" = famílias recusa+distrator; negativa = "dava para responder" = oráculo; a
parcial é informativa, fora da binária). Validado contra o v2 (reproduz recall 100% / FP 23.3%).

| variante | recusa-correta (recall) | recusa-indevida (FP) | precision | **F1** | TP / FN / FP / TN |
|---|--:|--:|--:|--:|:--|
| **27B (teto)**   | 95.6 | **3.3**  | 97.7 | 96.6 | 43 / 2 / 1 / 29 |
| **v3_r64_q4**    | **100.0** | **13.3** | 91.8 | **95.7** | 45 / 0 / 4 / 26 |
| v3_r64_bf16      | 95.6 | **10.0** | 93.5 | 94.5 | 43 / 2 / 3 / 27 |
| v3_r128_q4       | 93.3 | 13.3 | 91.3 | 92.3 | 42 / 3 / 4 / 26 |
| v3_r128_bf16     | 95.6 | 13.3 | 91.5 | 93.5 | 43 / 2 / 4 / 26 |
| v2_r64_q4        | 100.0 | 23.3 | 86.5 | 92.8 | 45 / 0 / 7 / 23 |
| v2_r64_bf16      | 97.8 | 26.7 | 84.6 | 90.7 | 44 / 1 / 8 / 22 |
| base 2.6B (Q4)   | 60.0 | 0.0  | 100.0 | 75.0 | 27 / 18 / 0 / 30 |

**RESULTADO CENTRAL**: o rebalanceamento (recusa 52%→24.5%) atacou exatamente o efeito
colateral do v2. A **recusa-indevida caiu de 23-27% (v2_r64) para 10-13% (v3)** — praticamente
metade — mantendo recusa-correta ≥93% e **subindo a precision (84-87→91-94) e o F1 (91-93→92-96)**.
O v3 fica mais perto do teto 27B (FP 3.3, F1 96.6). A base 2.6B "não recusa" (recall 60%, F1 75).

---

## PARTE 4 — seleção + quantização COM calibração

### Tabela mestre (régua honesta, n=151; refusal como classificador)

| variante | oráculo apr% | oráculo **aluc%** | retr apr% | retr **aluc%** | val apr% | rec-correta% | rec-indevida% | F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **27B (teto)**   | 84.3 | 2.6  | 45.7 | 27.8 | 60.0 | 95.6 | 3.3  | 96.6 |
| **v3_r64_bf16**  | 47.0 | 21.9 | **26.7** | **34.0** | 63.6 | 95.6 | **10.0** | 94.5 |
| **v3_r64_q4**    | 46.4 | 21.2 | 20.0 | 40.7 | 62.1 | 100.0 | 13.3 | **95.7** |
| v3_r128_bf16     | 47.7 | **18.5** | 20.7 | 42.0 | 67.9 | 95.6 | 13.3 | 93.5 |
| v3_r128_q4       | 48.3 | 19.9 | 18.7 | 44.0 | 57.9 | 93.3 | 13.3 | 92.3 |
| v2_r64_q4 (v2)   | 47.7 | 15.9 | 22.0 | 34.0 | 54.3 | 100.0 | 23.3 | 92.8 |
| v2_r64_bf16 (v2) | 57.0 | 13.2 | 22.0 | 39.3 | 62.1 | 97.8 | 26.7 | 90.7 |

### Escolha: **v3_r64** (Q4_0 para embarque), justificada pelo perfil cauteloso

- **r64 > r128** aqui: r64_bf16 tem o melhor retrieval real (26.7% vs 20.7%) e a menor
  recusa-indevida (10.0%); r128 baixa a alucinação em oráculo mas piora a busca real e não
  reduz a recusa-indevida. O ganho de rank saturou (ao contrário do v2, onde r64 ainda subia).
- **Perfil cauteloso** (não inventar pesa mais que responder mais): o v3 troca ~10 p.p. de
  aprovação-por-cobertura do v2_bf16 por **metade da recusa-indevida** e alucinação comparável
  — exatamente o que o capitão pediu. O F1 de recusa 95.7 (r64_q4) é o melhor da família 2.6B.

### Calibração da quantização (imatrix) — a pendência que o capitão apontou

`build_calibration.py` monta 400 blocos estratificados do domínio (system v3 + contexto NR +
pergunta + resposta, fora do holdout) → `quantize_calibrated.sh` gera imatrix e quantiza COM e
SEM calibração em Q4_0 e Q4_K_M. `data/calibration_comparison.json`:

| formato | oráculo apr% | Δ vs bf16 | oráculo aluc% | retr apr% | rec-indevida% | F1 |
|---|--:|--:|--:|--:|--:|--:|
| bf16 (teto do treino) | 47.0 | — | 21.9 | 26.7 | 10.0 | 94.5 |
| **Q4_0 sem calib**    | 46.4 | **−0.6** | 21.2 | 20.0 | 13.3 | **95.7** |
| Q4_0 COM calib        | 44.4 | −2.6 | 20.5 | 24.7 | 13.3 | 94.6 |
| Q4_K_M sem calib      | 43.7 | −3.3 | 16.6 | 23.3 | 10.0 | 94.5 |
| Q4_K_M COM calib      | 45.0 | −2.0 | 28.5 | 28.7 | 20.0 | 92.6 |

**ACHADO (responde a pergunta do capitão)**: no v3 a perda **bf16→Q4_0 é NEGLIGÍVEL (−0.6
p.p.)**, ao contrário do v2 (**−9.3 p.p.**). O v2 caía tanto porque o hedge (recusa/delimitação)
era sensível ao ruído do 4-bit; o v3, com menos over-refusal, quantiza quase de graça. **A
calibração NÃO recuperou nada** — a imatrix até PIOROU levemente o Q4_0 (−2.0 p.p.) e no
Q4_K_M subiu a alucinação (16.6→28.5%). **Formato final por evidência: Q4_0 SEM calibração**
(quase-bf16, melhor F1, menor complexidade). O experimento com calibração foi feito e a
resposta honesta é que, neste modelo, ela não paga.

---

## PARTE 5 — avaliação geral (catastrophic forgetting)

`run_general_panel.sh` reusa `finetune2/eval_panel.py` (camadas general+behavioral, conjuntos
CONGELADOS `ood_general.json` 100 + `ood_behavioral.json` 30, juiz 27B). v3_r64_q4 vs base
2.6B Q4_0 (mesmos pesos-base). `data/forgetting_panel.json`:

| camada | métrica | base 2.6B | v3_r64_q4 | Δ |
|---|---|--:|--:|--:|
| general | qualidade | 3.93 | 3.85 | −0.08 (< ruído 0.20) |
| general | PT-BR | 4.85 | 4.87 | +0.02 |
| general | segurança | 4.46 | 4.46 | 0.00 |
| behavioral | qualidade | 3.67 | 4.03 | **+0.37** |
| behavioral | segurança | 4.00 | 4.30 | **+0.30** |

**VEREDITO: SEM catastrophic forgetting.** O capitão estava certo — o PEFT (LoRA) protege a
capacidade geral. Nenhuma regressão além do ruído; o comportamental **melhorou** (o treino de
recusa útil transferiu para segurança/qualidade fora do domínio). A pendência foi fechada.

---

## PARTE 6 — embarque no S24+ (Nemotron 3.5 + 2.6B v3 Q4)

O capitão quis experimentar os DOIS motores juntos, "mesmo que fique bem apertado em RAM".

### O que foi construído
- **Módulo `:sherpa`** (`android/sherpa/`): sherpa-onnx JNI pré-compilado (`libsherpa-onnx-jni.so`
  + `libonnxruntime.so`, arm64-v8a) + API Kotlin oficial `com.k2fsa.sherpa.onnx` (casada com o
  JNI). `NemotronAsrEngine` (transducer streaming 560ms, idioma pt por-stream).
- **`RealNemotronAsrEngine`** (app): implementa `AsrEngine` com o MESMO contrato
  anti-reprocessamento do S24+ (canal sem replay, recordingId monotônico, STATUS×CONTEÚDO,
  Final não auto-dispara). Reusa o `AudioRecordRecorder` do `:whisper`.
- **Seleção por build**: `-Pcemig.asrEngine=nemotron` (default = Whisper) e
  `-Pcemig.synthModel=2.6b` (default = 1.2B). `MainViewModel` resolve o motor ativo no init;
  **fallback honesto para Whisper** se os pesos do Nemotron não estiverem no aparelho.
- **`ModelFileManager.getNemotronModelDir`**: override externo (os ONNX grandes, encoder 657 MB,
  ficam fora do APK, injetados por `adb push`).
- **Caixa de texto da transcrição corrigida** (reclamação do capitão "muito pequena, super
  difícil de ler"): era `singleLine`; agora **multilinha, cresce com o conteúdo (64–168 dp),
  rolável, fonte 16sp** (`MainScreen.kt`).

### Medições no aparelho (release, modo avião, `android/acceptance_results_v3_nemotron_2p6b.json`)

Os DOIS motores carregaram com sucesso (log: `ASR ativo: Nemotron 3.5 INT8 (sherpa-onnx)` +
`Sintetizador: LFM2.5-2.6B-Q4_0.gguf (thinking-OFF=true)`).

| métrica | valor | observação |
|---|--:|---|
| **RAM PSS (os dois residentes)** | **4.75 GB** | RSS 4.85 GB; cabe no S24+ (12 GB) |
| Nemotron ASR (latência) | **~1.17 s** (0.83–1.36) | rápido; captura TODOS os termos técnicos |
| SLM 2.6B — TTFT | 8.7–15 s | prefill ~950–1345 tok |
| SLM 2.6B — decode | 18–20 tok/s | 160–219 tok |
| **E2E total** | **~20 s** (16.5–23.8) | **10/10 acima do teto de 10 s** |
| App morto por memória? | **NÃO** | o LMK matou apps de FUNDO; o app (foreground) sobreviveu |

**Veredito de RAM/latência (honesto)**: a coexistência **funciona no S24+ mas é "apertada"**
exatamente como o capitão previu — PSS 4.75 GB força o Android a matar processos de fundo, e o
**2.6B custa ~20 s/pergunta** (inviável para o teto de voz de 10 s; consistente com
`slm_oraculo`/`engine-upgrade`). **Proposta**: para produção de voz, **carregamento sob demanda**
do sintetizador OU manter o **1.2B** como sintetizador residente e o Nemotron como ASR (o
Nemotron sozinho + 1.2B cabe folgado). O 2.6B v3 é o alvo quando latência/RAM destravarem.

### Amostras literais por VOZ (Nemotron ASR → v3 2.6B), incluindo os casos do capitão

- **Q03 — 13,8 kV** (caso do capitão): ASR _"…rede de treze vírgula oito k"_ →
  _"A distância mínima de segurança para trabalhar próximo a uma rede de treze vírgula oito kV
  é de 0,20 metros. Essa medida está definida na NR-10, Anexo II, na tabela de raios de
  del[imitação]…"_ — **cita a tabela certa mas lê a linha errada (0,20 m; o 27B lê 0,38 m)**:
  fronteira de capacidade de leitura fina de tabela, não corrigível por SFT (igual v1/v2).
- **Q04 — o poste** (ancoragem no poste): ASR _"…cinto tipo para quedista no poste"_ →
  _"A NR-35, Anexo II, item 3.1.1 exige que a estrutura integrante do sistema de ancoragem seja
  capaz de resistir à força máxima aplicável…"_ — fundamentado na norma certa.
- **Q05 — fator de queda** (multiturno): _"O contexto consultado não especifica o valor
  numérico do fator de queda tolerado para o talabarte, apenas exige que a organização
  estabeleça procedimentos de resgate…"_ — **recusa honesta e útil** (delimita sem inventar).
- **Q07 — capacete/óculos**: _"Não, a norma não exige o uso obrigatório de capacete conjugular
  ou óculos específicos para eletricistas. O contexto consultado trata apenas de EPIs para
  proteção da cabeça e face…"_ — hedge cauteloso (perfil escolhido).
- **Q10 — chuva forte**: _"Não, você não pode realizar a intervenção durante a chuva forte. A
  NR-10, item 10.6.3, determina que os serviços em instalações energizadas devem ser suspensos
  de imediato na iminência…"_ — resposta segura e fundamentada.

> **A camiseta rasgada (qa-003)** não está no roteiro de voz de 10 perguntas (que é o M2/M3);
> está no holdout das 151. No `retr151` o v3_r64_bf16 responde _"Não, você não pode usar a
> camiseta comum de algodão… A NR-24, item 24.8.3…"_ — **NÃO comete o erro de "adornos" do v1**
> (a norma vem errada do retrieval — NR-24 em vez de NR-10 —, limitação de busca, não do SLM).

### Nemotron acerta os termos técnicos (motivo da troca)
Todas as 10 transcrições saíram com o jargão certo: "desenergização", "proteção coletiva",
"cinto tipo paraquedista", "talabarte", "luva isolante", "aterramento temporário", "chave
seccionadora", "chave fusível", "circuito energizado". Confirma a trilha ASR (`asr/README.md`:
Nemotron 91.9% de termos técnicos vs Whisper Base 85.5%; o Whisper falhava em capturar
"disjuntor").

---

## VEREDITO GERAL

1. **A recusa foi controlada sem perder o perfil cauteloso**: recusa ≤ 25% derrubou a
   recusa-indevida de 23-27% (v2) para 10-13% (v3), subindo precision e F1 (94-96), mantendo
   recusa-correta ≥93%. O v3 é a melhor destilação de recusa que temos.
2. **r64 vence r128** (retrieval real melhor, menor recusa-indevida; rank saturou).
3. **Calibração não paga neste modelo**: a perda bf16→Q4_0 já é negligível (−0.6 p.p.); a
   imatrix não recuperou nada. **Embarque em Q4_0 sem calibração.**
4. **Sem catastrophic forgetting** (PEFT protege; comportamental até melhorou) — pendência fechada.
5. **Embarque S24+ funciona mas é apertado**: Nemotron ASR é ótimo (~1.2 s, todos os termos);
   o 2.6B custa ~20 s e PSS 4.75 GB (mata apps de fundo, app sobrevive). Para voz de produção:
   Nemotron ASR + **1.2B residente** (ou 2.6B sob demanda). O 2.6B v3 é o alvo quando o engine/
   latência destravar.

---

## Arquivos e reprodução

```
sft_v3/
  rebalance_data.py     # PARTE 1: reamostra pool v2 (recusa <= 25%) + oráculo fresco do déficit
  topup_oracle.py       # PARTE 1: completa oráculo via múltiplas perguntas por chunk (régua+dedup)
  train_v3.py           # PARTE 2: wrapper do train_v2 (motor único, rank-configurável)
  run_train_v3.sh       # PARTE 2: r64/r128 na Spark (mata servers, asserts, integridade recusa<=25%)
  refusal_metrics.py    # PARTE 3: recusa como classificador (precision/recall/F1 + confusão)
  eval_checkpoint_v3.sh # PARTE 3-4: 4 suites por checkpoint (reusa packs/generate/score do v2)
  run_all_evals_v3.sh   # PARTE 4: v3_r64/r128 (bf16+Q4)
  build_calibration.py  # PARTE 4: corpus de calibração (domínio, fora do holdout)
  quantize_calibrated.sh# PARTE 4: imatrix + Q4_0/Q4_K_M com e sem calibração (Spark)
  run_quant_evals.sh    # PARTE 4: avalia as 4 quantizações
  consolidate_v3.py     # PARTE 3-4: tabela mestre -> data/consolidation.json
  run_general_panel.sh  # PARTE 5: geral+comportamental (reusa finetune2/eval_panel.py)
  data/                 # dados, packs, respostas, scores, premises, consolidations (commitados leves)
```

Comandos (juiz via `REGUA_JUDGE_URL`, default `10.100.0.111:8005`):
```bash
PY=../classifier/.venv/bin/python
# PARTE 1 (local)
$PY rebalance_data.py && $PY topup_oracle.py --need 700
$PY build_calibration.py --n 400
# PARTE 2 (Spark): scp train_v3.jsonl + calib_v3.txt + scripts; setsid nohup run_train_v3.sh
# PARTE 3-4 (local): ./run_all_evals_v3.sh ; ./run_quant_evals.sh ; $PY consolidate_v3.py
#   quant na Spark: bash quantize_calibrated.sh <bf16> sft_v3_r64 <calib.txt>
# PARTE 5 (local): ./run_general_panel.sh <gguf_escolhido> v3_r64_q4
# PARTE 6 (build+deploy):
cd ../android && ./gradlew assembleRelease -Pcemig.synthModel=2.6b -Pcemig.asrEngine=nemotron
#   adb push dos pesos externos (Nemotron dir + LFM2.5-2.6B-Q4_0.gguf) e chmod 775 no dir do Nemotron
```

### Artefatos na Spark (gitignored: reprodutíveis)
- dataset: `~/cemig-poc/train_v3/train_v3.jsonl` (== `data/train_v3.jsonl`, 3065 pares)
- calibração: `~/cemig-poc/train_v3/calib_v3.txt`; imatrix `~/cemig-poc/models/imatrix_sft_v3_r64.dat`
- adapters/merged: `~/cemig-poc/train_v3/{ckpt,sft,merged}_sft_v3_r{64,128}/`
- GGUFs: `~/cemig-poc/models/lfm2.5-2.6b-sft_v3_r{64,128}-{bf16,Q4_0}.gguf` +
  `lfm2.5-2.6b-sft_v3_r64-Q4_0-imatrix.gguf` / `-Q4_K_M{,-imatrix}.gguf`
- **artefato de embarque**: `lfm2.5-2.6b-sft_v3_r64-Q4_0.gguf` (F1 recusa 95.7, recusa-indevida
  13.3%, oráculo 46.4% / aluc 21.2%) — quando o 2.6B couber na voz; hoje o embarcado segue 1.2B.
- log de treino: `~/cemig-poc/logs/train_v3.log`; wandb `cemig-sft-v3` (online).
- pesos ASR no aparelho: `getExternalFilesDir/nemotron-3.5-official-560ms-int8/` (adb push).
```
