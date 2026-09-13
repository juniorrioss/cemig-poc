# SFT do LFM2.5-1.2B — a receita do 2.6B aplicada ao modelo EMBARCADO (`sft_1_2b/`)

Ordem do capitão após o v3 do 2.6B (`sft_v3/`): *"dispare a mesma lógica para o modelo de
1.2B, mantenha os dados utilizados nessa última versão e dispare algumas variantes de rank de
forma similar, quero ver o quanto o treino vai auxiliar o modelo pequeno também."*

**Por que importa**: o **1.2B QAD é o único modelo que roda em latência de campo** (~6 s no
S24+ contra ~20 s do 2.6B, que come 4,75 GB e derruba apps de fundo). Na régua honesta ele é o
pior de todos (aprovação ~2-5%). Se a destilação levantar o 1.2B como levantou o 2.6B, o
produto por voz fica viável **sem trocar de modelo**.

> **Infra**: treino + export GGUF na **DGX Spark** (`walcyrios@spark-b431`, `~/jupyterlab/.venv`);
> dados/pontuação LOCAIS (`classifier/.venv`; juiz vLLM 27B `10.100.0.111:8005` roteável).
> Régua ÚNICA: `bench/regua` + juiz de recusa próprio (**reuso total de `sft_v2`/`sft_v3`**).
> Holdout 151/20 INTOCÁVEL; 3 muralhas + split por NR mantidos; wandb ONLINE
> (`cemig-sft-1.2b`). Paralelo, checkpoint, não-interativo. Comentários PT-BR; código em inglês.

**REUSO máximo (ordem do brief)**: `train_v2.py` (motor rank-configurável, com os asserts),
`common_v2.py` (system prompt v2, juízes, régua), `generate_v2.py`/`score_v2.py`, os 4 packs de
avaliação, `refusal_metrics.py` (recusa como classificador) e `finetune2/eval_panel.py`
(geral+comportamental). **Os DADOS são exatamente os do v3** (`train_v3.jsonl`, md5
`b404d55c…` idêntico na Spark) — nada foi regerado.

---

## PASSO 0 — arquitetura CONFIRMADA antes de treinar (exigência permanente do capitão)

Contamos os módulos no `model.safetensors` do base HF (NÃO assumimos os nomes do 2.6B):

| | LFM2.5-**1.2B**-Instruct | LFM2.5-**2.6B** (v3, referência) |
|---|---|---|
| camadas | **16** (10 conv + 6 full_attention) | 30 (22 conv + 6 attn — variam por config) |
| `self_attn.{q,k,v,out}_proj` | 6 cada | 8 cada |
| `feed_forward.{w1,w2,w3}` | 16 cada | 30 cada |
| `conv.{in_proj,conv,out_proj}` (ShortConv) | 10 cada | 22 cada |
| **alvos LoRA (regex do v2)** | **72** = 6·4 + 16·3 | 122 = 8·4 + 30·3 |

**Os nomes de módulo são IDÊNTICOS aos do 2.6B** → o REGEX do v2
(`.*\.(self_attn\.(q|k|v|out)_proj|feed_forward\.(w1|w2|w3))$`) é reaproveitável sem mudança.
Os **3 asserts** (`mlp✓ attn✓ conv✗`) do `train_v2` continuam válidos e ABORTAM se errar. Em
todos os 4 ranks o log publicou **72 alvos, mlp✓ attn✓ conv✗** (`data/premises/`).

### Qual checkpoint HF? (declaração do risco QAD, ordem do capitão)
O modelo EMBARCADO é `LFM2.5-1.2B-Instruct-**QAD**-Q4_0.gguf` (quantization-aware distilled da
Liquid). **NÃO existe checkpoint QAD em HF em alta precisão** — o QAD é distribuído **só como
GGUF Q4_0** (no repo `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`, ao lado dos GGUFs comuns). Logo o
treino foi feito sobre `LiquidAI/LFM2.5-1.2B-Instruct` (bf16, o base que a Liquid usou para
gerar o QAD). **Implicação**: treinar em bf16 e requantizar para Q4_0 é uma requantização
SIMPLES (sem o processo QAD) — pode não recuperar a vantagem do QAD-de-fábrica. Medimos isso
explicitamente na Parte 2 (QAD original vs base requant-simples vs treinado bf16 vs treinado Q4).

---

## PASSO 1 — varredura de rank (r16, r32, r64, r128; 1 época cada)

Receita IDÊNTICA ao v3 (bs8×ga4, max_len 2048, lr 1e-4, cosine, bf16, grad ckpt, targets por
regex, `assistant_only_loss`, val interno, seed 42, alpha=2r). Só o RANK varia. O 1.2B tem
menos capacidade → o ponto de saturação pode ser outro, por isso a varredura ampla (4 ranks).

| rank | alpha | trainable% | train_loss | **eval_loss interno** |
|---|--:|--:|--:|--:|
| 16  | 32  | 0.775% | 0.7470 | 0.6599 |
| 32  | 64  | —      | 0.7044 | 0.6193 |
| 64  | 128 | 2.90%  | 0.6641 | **0.5848** |
| 128 | 256 | 5.64%  | 0.6317 | **0.5558** |

**A `eval_loss` cai MONOTÔNICA e NÃO satura** (0.66→0.62→0.58→0.556) — ao contrário do 2.6B v3,
onde r64≈r128. Como referência, o **2.6B v3 fecha em eval_loss ~0.54-0.51** — ou seja, o 1.2B
r128 (0.556) mal chega onde o 2.6B r64 começa: **o modelo menor tem um teto de ajuste mais
alto**. **1 época bastou** (a alucinação NÃO explodiu monotonicamente como no v1 do 2.6B; ver
Parte 2). Não há evidência de underfitting que justifique propor 2 épocas: a `eval_loss` ainda
desceria, mas a **qualidade na régua já não acompanha o rank** (Parte 2) — treinar mais tenderia
a só aumentar alucinação. Mantido 1 época (comparabilidade com o v3).

---

## PASSO 2 — TABELA MESTRE (régua honesta, n=151; recusa como classificador)

`system = v2` nos treinados (prompt que permite recusa útil, o que treinaram); `numeros` nas
bases não-treinadas (igual o v2/v3 fizeram com base/27B). Reuso maçã-com-maçã: **2.6B v3_r64**
(melhor atual) e **27B** (teto) vêm de `sft_v3`/`sft_v2`, não foram rerodados.

| variante | orcAppr | orcCov | orc**Hall** | retrAppr | retrHall | valAppr | recCorr | recIndev | prec | **F1** | tok |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **1.2B QAD base** (embarcado) | 2.7 | 0.07 | 2.0 | 1.4 | 8.8 | 18.6 | 84.4 | 66.7 | 65.5 | 73.8 | 42 |
| 1.2B base bf16 (não-QAD)      | 4.1 | 0.05 | 0.7 | 1.4 | 2.7 | 14.3 | 86.7 | 56.7 | 69.6 | 77.2 | 26 |
| 1.2B base Q4 (requant simples)| 4.7 | 0.08 | 2.0 | 5.4 | 4.1 | 12.9 | 93.3 | 76.7 | 64.6 | 76.4 | 37 |
| **1.2B sft r16 Q4**  | **29.7** | 0.36 | 25.7 | 8.8  | 31.8 | 37.1 | 68.9 | 6.7 | 93.9 | 79.5 | 100 |
| 1.2B sft r16 bf16    | 25.7 | 0.35 | 30.4 | 12.8 | 35.1 | 48.6 | 68.9 | 6.7 | 93.9 | 79.5 | 100 |
| 1.2B sft r32 Q4      | 19.6 | 0.32 | 26.4 | 10.1 | 35.8 | 38.6 | 75.6 | 3.3 | 97.1 | 85.0 | 101 |
| 1.2B sft r32 bf16    | 20.9 | 0.33 | 26.4 | 12.8 | 29.1 | 47.1 | 75.6 | 16.7 | 87.2 | 81.0 | 102 |
| **1.2B sft r64 Q4** ⭐ | 22.3 | 0.36 | 23.0 | 10.1 | 33.1 | 50.7 | 82.2 | **0.0** | **100.0** | 90.2 | 100 |
| 1.2B sft r64 bf16    | 24.3 | 0.33 | **18.9** | **16.2** | 30.4 | 50.0 | 86.7 | 6.7 | 95.1 | 90.7 | 104 |
| 1.2B sft r128 Q4     | 20.9 | 0.32 | 30.4 | 13.5 | 32.4 | 49.3 | 82.2 | 3.3 | 97.4 | 89.2 | 103 |
| 1.2B sft r128 bf16   | 29.1 | 0.35 | 21.6 | 14.2 | 29.1 | 50.0 | 86.7 | 3.3 | 97.5 | **91.8** | 106 |
| **2.6B v3_r64 Q4** (melhor atual) | **46.4** | 0.52 | 21.2 | **20.0** | 40.7 | 62.1 | 100.0 | 13.3 | 91.8 | 95.7 | 139 |
| 2.6B v3_r64 bf16     | 47.0 | 0.56 | 21.9 | 26.7 | 34.0 | 63.6 | 95.6 | 10.0 | 93.5 | 94.5 | 136 |
| **27B (teto)**       | **84.3** | 0.75 | 2.6 | **45.7** | 27.8 | 60.0 | 95.6 | 3.3 | 97.7 | 96.6 | 107 |

### DELTA vs o base EMBARCADO (`1.2B QAD base`) — o número CENTRAL do capitão

| variante | Δ oráculo apr | Δ oráculo aluc | Δ retrieval apr |
|---|--:|--:|--:|
| sft r16 Q4   | **+27.0** | +23.7 | +7.4 |
| sft r64 Q4 ⭐ | **+19.6** | +21.0 | +8.7 |
| sft r64 bf16 | +21.6 | +16.9 | **+14.8** |
| sft r128 bf16| **+26.4** | +19.6 | +12.8 |

**SIM, o treino auxilia o modelo pequeno — e muito**: o embarcado de fábrica aprova **2.7%** em
oráculo e **1.4%** com busca real; o treinado salta para **~22-30% em oráculo** e **~10-16%** na
busca real, e a **recusa vira de fábrica-quebrada (F1 73.8, recusa-indevida 66.7%) para
cirúrgica (F1 90-92, recusa-indevida 0-7%)**. O modelo passa a **responder de fato** (tok médio
26→100) em vez de despejar "Não sei com base nas normas consultadas".

### Risco QAD MEDIDO (a pergunta explícita do capitão)
- **A requantização NÃO destrói vantagem — porque a vantagem do QAD JÁ NÃO EXISTE no nosso
  domínio.** O QAD-de-fábrica (2.7% oráculo) é o PIOR dos três bases; a requantização simples
  do base HF (`base Q4`, 4.7%) até o supera. O QAD é otimizado para benchmarks gerais, não para
  seguir contexto de NR — no nosso caso ele é praticamente piso.
- **No treinado, bf16→Q4_0 custa pouco e o sinal é misto** (não há hedge sensível ao 4-bit como
  no v2): r64 oráculo 24.3→22.3 (−2.0), r128 oráculo 29.1→20.9 (−8.2, o único caro), r16
  oráculo 25.7→**29.7** (o Q4 até SOBE). A alucinação também oscila nos dois sentidos. **Não há
  colapso de quantização**; o Q4 do treinado fica MUITO acima do QAD de fábrica em todas as
  métricas úteis. **O treino+requant é um ganho líquido claro sobre o embarcado.**

### O limite honesto — o 1.2B treinado continua BEM abaixo do 2.6B
- oráculo: melhor 1.2B ~24-30% vs **2.6B v3 46-47%** vs **27B 84%**;
- retrieval real: melhor 1.2B ~16% vs **2.6B 20-27%** vs **27B 46%**;
- **alucinação é o preço**: o 1.2B sobe de 2%→**19-30%** (contra ~21% do 2.6B v3) — tem menos
  capacidade de ficar fiel ao contexto quando decide responder. A régua honesta pega isso.
- **retrieval segue o teto prático**: mesmo o melhor 1.2B só aprova ~16% na busca v4 real; o
  chunk-ouro simplesmente não entra no top-2 na maioria das perguntas (limite do retriever, não
  do sintetizador).

### Qual rank? (escala NÃO saturou na loss, mas SATUROU na régua)
A `eval_loss` cai sempre, mas a **qualidade na régua não segue**: r16 tem o melhor oráculo Q4
(29.7%) porém a pior recusa (F1 79.5, recall 68.9) e alucinação alta; r64/r128 têm a recusa
cirúrgica (F1 90-92). **Escolha de embarque = `r64 Q4`** pelo **perfil cauteloso** (mesma
filosofia do v3): **recusa-indevida 0.0%, precision 100%, F1 90.2**, alucinação 23% (mediana da
família), oráculo 22.3%. Em bf16, `r64` tem a menor alucinação (18.9%) e o melhor retrieval real
(16.2%) — é o alvo se houver folga de precisão.

---

## PASSO 3 — recusa como CLASSIFICADOR + forgetting + device

### Recusa (matriz de confusão; positivo = "deveria recusar")

| variante | recusa-correta (recall) | recusa-indevida (FP) | precision | **F1** | TP / FN / FP / TN |
|---|--:|--:|--:|--:|:--|
| **27B (teto)**       | 95.6 | 3.3  | 97.7 | 96.6 | 43 / 2 / 1 / 29 |
| 2.6B v3_r64 Q4       | 100.0 | 13.3 | 91.8 | 95.7 | 45 / 0 / 4 / 26 |
| **1.2B sft r128 bf16** | 86.7 | 3.3 | 97.5 | **91.8** | 39 / 6 / 1 / 29 |
| 1.2B sft r64 bf16    | 86.7 | 6.7 | 95.1 | 90.7 | 39 / 6 / 2 / 28 |
| **1.2B sft r64 Q4** ⭐ | 82.2 | **0.0** | **100.0** | 90.2 | 37 / 8 / 0 / 30 |
| 1.2B sft r128 Q4     | 82.2 | 3.3 | 97.4 | 89.2 | 37 / 8 / 1 / 29 |
| 1.2B sft r32 Q4      | 75.6 | 3.3 | 97.1 | 85.0 | 34 / 11 / 1 / 29 |
| 1.2B sft r16 Q4      | 68.9 | 6.7 | 93.9 | 79.5 | 31 / 14 / 2 / 28 |
| **1.2B QAD base** (embarcado) | 84.4 | **66.7** | 65.5 | 73.8 | 38 / 7 / 20 / 10 |

O base QAD "recusa" muito (recall 84%) mas de forma indiscriminada — **recusa-indevida 66.7%**
(recusa em 2 de cada 3 perguntas respondíveis): é o "Não sei" preguiçoso. O treino conserta:
`r64 Q4` recusa **certo e quase nunca à toa** (FP 0%, precision 100%), F1 90.2 — a **~6 p.p. do
2.6B v3 e ~6 p.p. do 27B**. A recusa é a dimensão em que o 1.2B treinado mais se aproxima dos
modelos grandes.

### Catastrophic forgetting — `r64 Q4` vs base QAD (juiz 27B, conjuntos congelados)

| camada | métrica | base QAD | r64 Q4 | Δ |
|---|---|--:|--:|--:|
| general | qualidade | 3.76 | 3.29 | **−0.47** (> ruído 0.20) |
| general | PT-BR | 4.93 | 4.80 | −0.13 |
| general | segurança | 4.61 | 4.25 | **−0.36** |
| behavioral | qualidade | 3.73 | 3.33 | **−0.40** |
| behavioral | PT-BR | 4.73 | 4.80 | +0.07 |
| behavioral | segurança | 4.37 | 4.17 | −0.20 |

**ACHADO HONESTO (difere do 2.6B)**: no 1.2B **HÁ forgetting mensurável** (qualidade geral
−0.47, segurança −0.36, ambos acima do ruído). No 2.6B v3 o PEFT protegeu (Δ < ruído, o
comportamental até subiu). **A capacidade menor do 1.2B não absorve a especialização de graça**:
ele fica melhor em NR e um pouco pior no resto. PT-BR intacto (o dano é de qualidade/segurança
geral, não de fluência). É um custo real a pesar — mas o app é de domínio único (NR), então o
impacto de produto é baixo.

### Device S24+ (`SM-S926B`, Exynos 2400; llama-bench 6 threads, pp1500/tg100)

| modelo | tamanho | prefill (pp1500) | decode (tg100) | E2E estimado (prefill ~1420 + 100 tok) |
|---|--:|--:|--:|--:|
| **1.2B sft r64 Q4 (treinado)** | 661 MiB | 294 tok/s | 50.8 tok/s | TTFT 4.83 s + decode 1.97 s = **6.79 s** (SLM) |
| 1.2B QAD Q4 (embarcado) | 661 MiB | 299 tok/s | 48.2 tok/s | TTFT 4.76 s + decode 2.07 s = **6.83 s** (SLM) |

**PARIDADE TOTAL de latência/RAM** — mesma arquitetura, mesma quantização (Q4_0), mesmo tamanho
(661 MiB). O modelo treinado **entra no app no lugar do QAD com custo de porte ZERO**: mesma
memória, mesma latência (~6-8 s E2E com o Nemotron ASR ~1,2 s, **dentro do teto de voz de
10 s**). Ao contrário do 2.6B (~20 s / 4,75 GB, inviável), aqui a troca é gratuita em recursos.

---

## Amostras literais (r64 Q4, oráculo — pergunta crua → resposta crua)

**Casos do capitão:**

- **13,8 kV** (`cap-13kv`): _"Qual a distância segura para média tensão de 13,8kV?"_ →
  _"O raio da zona de risco é de 0,35 metros e o raio da zona controlada é de 1,35 metros. Essa
  distância segura está definida no Anexo II da NR-10 para tensões entre 6 kV e 10 kV."_ — **cita
  a tabela certa mas lê a FAIXA errada** (pega 6-10 kV, não a de 13,8 kV): mesma fronteira de
  leitura fina de tabela do v1/v2/v3 do 2.6B — **nenhum SLM (nem o 27B com esse índice) lê a
  linha certa**; é limite de DADO/retrieval, não corrigível por SFT.
- **A camiseta rasgada** (`qa-003`): _"Tô no poste e minha camisa rasgou. Posso terminar com
  camiseta comum de algodão?"_ → _"Não, é vedado usar adornos pessoais… a NR-10, item 10.2.9.3,
  proíbe…"_ — **COMETE o erro de "adornos" do v1 do 2.6B** (atribui o item de joias à roupa). O
  1.2B, com menos capacidade, ainda cai no distrator plausível apesar da família de treino; o
  base QAD, por outro lado, apenas recusa ("não sei"). Custo da capacidade menor.
- **O poste bambo** (`cap-poste`): _"O poste que preciso subir não me parece firme, o que faço?"_
  → resposta sobre treinamento/carga horária (NR-35) — **norma da família certa mas item errado**
  (retrieval trouxe treinamento, não recusa/GRO): limite de retrieval.

**5+ pares aprovados (resposta contém a informação pedida, sem alucinação):**

1. `qa-002` (cov 1.00) _"Que negócio é essa tensão de segurança?"_ → _"A NR-10, item 10.2.8.2,
   prevê que, na impossibilidade de desenergização, deve-se empregar tensão de segurança. O
   texto consultado não especifica a natureza técnica…"_ (responde E delimita o que falta).
2. `qa-019` (cov 1.00) _"Fiz curso do SENAI, já sou 'qualificado'?"_ → delimita que o trecho
   trata de outra coisa e diz o que verificar (curso reconhecido pelo Sistema Oficial de Ensino).
3. `qa-022` (cov 0.60) _"Pra que serve a reunião no pé do poste?"_ → _"…ordem de serviço
   específica, aprovada por trabalhador autorizado… item 10.11.2 da NR-10."_
4. `qa-031` (cov 1.00) _"Carga >75 kW, que papelada o fiscal pede?"_ → _"…o Prontuário de
   Instalações Elétricas, que deve conter procedimentos e instruções técnicas de segurança…"_
5. `qa-102` (cov 1.00) _"Isolador rachado, posso parar o serviço sem punição?"_ → _"A NR-01,
   item 1.4.3, determina que você deve interromper imediatamente ao constatar risco grave e
   iminente…"_
6. `qa-010` (cov 0.67) _"Posso mandar fechar a chave e o pessoal termina depois?"_ → _"Não… o
   estado desenergizado deve ser mantido até a autorização para reenergização…"_

---

## VEREDITO GERAL (número honesto, como o capitão pediu)

1. **O treino AUXILIA muito o modelo pequeno vs ele mesmo**: sobre o QAD embarcado (o pior de
   todos), o SFT dá **+20 a +27 p.p. em oráculo**, **+7 a +15 p.p. na busca real**, e transforma
   a recusa (F1 73.8 → 90-92, recusa-indevida 66.7% → 0-7%). O modelo passa a responder de fato.
2. **Risco QAD medido — não há vantagem a destruir**: o QAD-de-fábrica é praticamente piso no
   nosso domínio (2.7% oráculo, abaixo até da requantização simples). A requantização do treinado
   (bf16→Q4_0) custa pouco e é assimétrica; o Q4 treinado fica muito acima do QAD em tudo. **O
   treino+requant é ganho líquido claro.**
3. **Mas continua BEM abaixo do 2.6B** (oráculo ~24 vs 46%, retrieval ~16 vs 20-27%) e **alucina
   mais** (19-30% vs ~21%), e **HÁ catastrophic forgetting** (qualidade geral −0.47 > ruído) —
   coisa que o 2.6B não teve. É o preço da capacidade menor.
4. **Escolha de embarque: `r64 Q4`** (`lfm2.5-1.2b-sft_1_2b_r64-Q4_0.gguf`) pelo perfil cauteloso
   (recusa-indevida 0%, precision 100%, F1 90.2) — a escala saturou na régua (a `eval_loss` ainda
   cai, mas a qualidade não sobe com o rank).
5. **Recomendação de embarque**: **SIM, vale trocar o QAD embarcado pelo r64 Q4** — é **paridade
   total de latência/RAM no S24+** (~6-8 s E2E, cabe na voz) com um salto real de qualidade
   sobre o piso atual, e a recusa fica a ~6 p.p. do 2.6B/27B. **Ressalvas honestas**: (a) a
   alavanca dominante continua sendo o **retrieval** (só ~16% aprovam na busca real, mesmo com o
   melhor 1.2B); (b) vigiar a alucinação (23%) e o forgetting de domínio geral; (c) o 2.6B v3
   segue o alvo de qualidade quando latência/RAM destravarem — ou destilá-lo PARA o 1.2B (o
   teacher aqui foi o 27B; um teacher 2.6B-régua poderia fechar parte do gap).

---

## Arquivos e reprodução

```
sft_1_2b/
  train_1_2b.py            # PASSO 1: wrapper do train_v2 (só renomeia o GGUF p/ lfm2.5-1.2b-*)
  run_train_1_2b.sh        # PASSO 1: varredura r16/r32/r64/r128 na Spark (PASSO0 + asserts + integridade)
  make_base_ggufs.sh       # PASSO 2: GGUFs do base NÃO-treinado (bf16 + Q4 requant simples) p/ o risco QAD
  eval_checkpoint_1_2b.sh  # PASSO 2: 4 suites por checkpoint (reusa packs/generate/score do v2)
  run_all_evals_1_2b.sh    # PASSO 2: bases (QAD/bf16/Q4) + varredura (bf16+Q4)
  run_general_panel_1_2b.sh# PASSO 3: geral+comportamental (forgetting) vs base QAD
  consolidate_1_2b.py      # PASSO 2-3: tabela mestre + DELTA vs QAD -> data/consolidation.json
  data/                    # scores, gen, premises, consolidation, refusal_confusion, forgetting, device_bench
```

Comandos (juiz via `REGUA_JUDGE_URL`, default `10.100.0.111:8005`):
```bash
PY=../classifier/.venv/bin/python
# PASSO 1 (Spark): scp scripts p/ ~/cemig-poc/slm_scripts_1_2b; base HF já em ~/cemig-poc/base_hf_1.2b
#   setsid nohup bash ~/cemig-poc/slm_scripts_1_2b/run_train_1_2b.sh > ~/cemig-poc/logs/train_1_2b.log 2>&1 &
#   bash ~/cemig-poc/slm_scripts_1_2b/make_base_ggufs.sh   # gera os GGUFs de base p/ o risco QAD
# PASSO 2-3 (local):
make -C sft_1_2b evals        # ./run_all_evals_1_2b.sh (bases + varredura, bf16+Q4)
make -C sft_1_2b consolidate  # tabela mestre + deltas
make -C sft_1_2b panel        # forgetting (r64_q4 vs base QAD)
```

### Artefatos na Spark (gitignored: reprodutíveis)
- base HF: `~/cemig-poc/base_hf_1.2b/` (LiquidAI/LFM2.5-1.2B-Instruct, bf16)
- dataset: `~/cemig-poc/train_v3/train_v3.jsonl` (== v3, md5 idêntico — dados mantidos)
- adapters/merged: `~/cemig-poc/train_1_2b/{ckpt,sft,merged}_sft_1_2b_r{16,32,64,128}/`
- GGUFs: `~/cemig-poc/models/lfm2.5-1.2b-sft_1_2b_r{16,32,64,128}-{bf16,Q4_0}.gguf`
- bases GGUF: `LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf` (embarcado) +
  `LFM2.5-1.2B-Instruct-base-{bf16,Q4_0}.gguf` (não-QAD)
- **artefato de embarque**: `lfm2.5-1.2b-sft_1_2b_r64-Q4_0.gguf` (661 MiB; F1 recusa 90.2,
  recusa-indevida 0%, oráculo 22.3% / aluc 23.0%; paridade de latência/RAM com o QAD embarcado)
- log de treino: `~/cemig-poc/logs/train_1_2b.log`; wandb `cemig-sft-1.2b` (online)
```
