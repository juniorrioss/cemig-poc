# RESULTS — P2 v5 (pool v3 + contrato v4) + pendências formais do P2

Ordem do capitão: um **v5** que junta o pool do v3 (o que domou o distrator) e o contrato
escalonado do v4 (menos grupos sem gradiente, mais conciso), avaliado contra **f4-r32-ep2
(base)** e **v3s400 (modelo do app)**. Junto, as pendências formais do P2: closed-book ×1536
do v3s400 e o rollout P1 completo do SFT. Código reaproveitado de `bench/lfm3_p1p2/` sem
reescrita (`train_p2_rlvr.py --contract v4` sobre `p2_prompts_v3.jsonl`). Juiz 27B, régua
honesta, GPU 3 da H100 compartilhada com o Qwen. Tabelas: `data/consolidation_recorte.json`,
`data/consolidation_full.json`, `data/efficiency_v5.json`, `data/p1_completion.json`.

## Veredito

**O v5 NÃO é promovido; o v3s400 continua no app.** O critério do capitão pedia v5 ≥ v3s400
em distrator, e2e e 151 **e** ganho em concisão ou em outra métrica relevante. O v5s400 só
cumpre a concisão (e2e 118 → 107 tok, −9%; 151 104 → 87 tok, −17%). Fica atrás nas três
métricas de guarda:

| ×1536 / 151 | v3s400 | v5s400 | pareado v5 × v3 |
|---|--:|--:|---|
| distrator: alucinação | **15,7** | 17,5 | 68 pior / 40 melhor, **p = 0,009** |
| e2e do app: aprovação | **71,8** | 70,3 | 50 ganhos / 73 perdas, **p = 0,047** |
| POC 151 coloquiais: aprovação | **21,2** | 19,2 | 1 / 4 (n.s.) |
| tokens no e2e do app | 118,0 | **107,2** | — |

**Pareto honesto.** O v5 fica entre o v3 e o v4: mais conciso que o v3 e menos conservador
que o v4. Nas 151 ele **alucina menos que o v3** (27,2 vs 31,1) e recusa mais (42,4 vs 35,1).
Contra a base, ganha tudo o que o v3 ganha (distrator 23,8 → 17,5, e2e 64,1 → 70,3, 151 12,6 → 19,2),
só que com margem menor. Não é um avanço sobre o v3s400. Fica registrado como a variante
"v3 enxuta", sem superioridade estatística.

## 1. Run v5

- **Receita**: `train_p2_rlvr.py --run lfm3-p2-rlvr-v5 --prompts p2_prompts_v3.jsonl
  --contract v4 --refuse-oversample 1 --lr 5e-6 --micro-batch 4 --max-prompt-tokens 2200
  --max-steps 800 --save-steps 100`. Pool **byte-idêntico** ao do v3 (md5 `9d481ad9…`;
  5.369 prompts após o corte de 2.200 tok: 3.496 answer / 1.873 refuse). LoRA novo r32 α64
  w1/w2/w3 sobre o f4-r32-ep2 mesclado, asserts de sempre (72 lora_A em feed_forward,
  0 em attn/gate/conv). ~14,6 s/step. wandb `lfm3-1.7b-cemig` / run **`lfm3-p2-rlvr-v5`**
  (`w4ub72o7`). Ckpts em `/workspace/cemig/checkpoints/lfm3-p2-rlvr-v5/checkpoint-{100..600}`.
- **Parada antecipada no step 605** pela regra do capitão (reward achatado por 100+ steps):
  a média de reward por janela de 100 steps subiu +0,064 (até 200), +0,049 (300) e +0,022 (400),
  depois **−0,002 (401–500) e +0,007 (501–600)**. Foram 200 steps planos, e o 800 não foi
  treinado. A poda ficou com 400 e 600.

**frac_reward_zero_std** (fração de grupos sem gradiente), médias por janela de 50 steps:

| run | 1–50 | 51–100 | 101–150 | 151–200 | 201–250 | 251–300 | 301–350 | 351–400 | 401–600 | média 1–400 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| v2 (binário, pool v2) | 0,31 | 0,34 | 0,51 | 0,54 | 0,58 | 0,61 | 0,70 | 0,74 | — | **0,54** |
| v3 (binário, pool v3) | 0,32 | 0,36 | 0,45 | 0,48 | 0,55 | 0,59 | 0,65 | 0,69 | — | 0,51 |
| v4 (v4, pool v4s refiltrado) | 0,18 | 0,24 | 0,28 | 0,34 | 0,38 | 0,33 | 0,40 | 0,45 | — | **0,33** |
| **v5 (v4, pool v3)** | 0,28 | 0,28 | 0,34 | 0,35 | 0,39 | 0,44 | 0,42 | 0,49 | 0,48–0,55 | **0,37** |

Leitura: com o mesmo pool, o contrato v4 derruba os grupos sem gradiente de **0,51 (v3) para
0,37 (v5), −27%**. O v4 ficou em 0,33 porque o pool dele foi **refiltrado pelo próprio reward
v4** (só grupos com variância). Da queda total v3 → v4 (0,51 → 0,33), **~3/4 vêm do contrato**
(0,51 → 0,37) e ~1/4 da refiltragem do pool. O resíduo do v5 continua vindo das recusas
(35% do pool, reward binário), e a curva sobe até 0,55 em 600 steps, a mesma saturação do
binário, só que mais lenta. Os tokens médios no treino ficaram estáveis (83–88, média 85,
contra 86 do v3).

## 2. Poda (recorte 344, 4 células + e2e + probe)

| tag | ranking | rec-indev. | distrator aluc. | distr. recusa | oráculo | closed | e2e app | cita item | aluc. app | probe | tok |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| f4-r32-ep2 (base) | 77,6 | 3,2 | 22,4 | 61,0 | 89,0 | 7,8 | 68,3 | 69,0 | 14,5 | 3,38 | 86,2 |
| v3s400 (app) | 84,6 | 2,9 | 15,4 | 70,3 | 91,9 | 9,0 | 75,0 | 73,3 | 13,1 | 3,35 | 94,3 |
| v4s400 | 81,1 | 4,4 | 14,5 | 72,7 | 87,2 | 7,6 | 73,8 | 71,7 | 12,5 | 3,47 | 86,7 |
| **v5s400** | 81,4 | 3,2 | **17,2** | 68,9 | 89,5 | 6,1 | 73,8 | 72,3 | 12,2 | 3,37 | 87,7 |
| v5s600 | 83,7 | 1,7 | 21,2 | 62,8 | 92,2 | 8,1 | 74,4 | 73,0 | 12,5 | 3,46 | 91,2 |

Pareado vs base: v5s400 distrator 38/11 (p = 0,0001), e2e 27/8 (p = 0,002); v5s600 distrator
25/19 (p = 0,45), e2e 27/6 (p = 0,0003), ranking 36/15 (p = 0,005).

**Escolha do ckpt: s400.** O distrator é a métrica que define o v3, e o s600 devolve quase
todo o ganho (21,2 contra 22,4 da base, sem significância). Mais steps com o v4 sobre o pool v3
**desaprendem a recusa estática** e ainda gastam mais tokens (91). O s600 ganha em oráculo e
ranking, mas está dentro do ruído do recorte (±3 p.p. com sampling 0,1).

## 3. Bateria completa (v5s400 × base × v3s400; ×1536 + 151)

| tag | distrator aluc. | distr. recusa | ranking | oráculo | closed | e2e app | aluc. app | cita item | 151 aprov. | 151 aluc. | 151 recusa | tok e2e | tok todas |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| base | 23,8 | 66,0 | 78,1 | 88,5 | 8,3 | 64,1 | 14,1 | 64,3 | 12,6 | 28,5 | 53,0 | 103,8 | 86,6 |
| **v3s400 (app)** | **15,7** | **73,1** | **81,6** | **90,0** | 7,3 | **71,8** | 15,0 | **69,2** | **21,2** | 31,1 | 35,1 | 118,0 | 93,2 |
| v4s400 (ref.) | 13,7 | 75,9 | — | — | — | 70,4 | 14,0 | 67,4 | 17,9 | 25,8 | 45,7 | 106,0 | — |
| **v5s400** | 17,5 | 72,3 | 81,2 | 89,6 | 7,7 | 70,3 | 15,0 | 68,9 | 19,2 | 27,2 | 42,4 | **107,2** | **87,6** |

Pareados ×1536 (teste do sinal; "ganha/perde" na aprovação; "pior/melhor" na alucinação):

| célula | v5 × base | v5 × v3s400 |
|---|---|---|
| distrator: alucinação pior/melhor | 67/164 (p < 1e-6) | **68/40 (p = 0,009)** |
| ranking | 135/87 (p = 0,002) | 77/83 (n.s.) |
| oráculo | 92/74 (n.s.) | 50/55 (n.s.) |
| closed-book | 58/68 (n.s.) | 63/57 (n.s.) |
| e2e do app | 149/54 (p < 1e-6) | **50/73 (p = 0,047)** |
| POC 151 | 16/6 (p = 0,05) | 1/4 (n.s.) |

Fora do recorte (1.192 queries de normas fora do recorte 344): distrator v5 17,4 contra
v3 15,9; e2e v5 69,3 contra v3 70,8. É o mesmo padrão do total, então não há efeito de norma.
Tokens por célula (v3 → v5): ranking 108,6 → 103,4, oráculo 105,1 → 100,5,
closed 64,6 → 60,6, distrator 69,5 → 66,2. O contrato v4 encurta as respostas em todas as
células sem que a recusa indevida mude (ranking 6,2 → 6,1, oráculo 3,3 → 3,4).

Por que o v5 não herdou o distrator do v3: no v4 o crédito parcial premia "responder com
cobertura e citação" em ~65% do pool (answer), enquanto a recusa segue binária. O
gradiente relativo puxa para responder; com mais steps (s600) isso piora visivelmente no
distrator estático. **Próxima alavanca honesta** (não executada): recusa também graduada,
ou peso maior da recusa no formato estático, ou pool v4s refiltrado + mais âncoras
estáticas.

## 4. Pendências formais do P2

### (a) closed-book ×1536 do v3s400 — FECHADA

| ×1536 | base | v3s400 | v5s400 |
|---|--:|--:|--:|
| closed-book aprov. | 8,3 | 7,3 | 7,7 |
| alucinação | 67,5 | 68,6 | 69,3 |
| tok | 59,5 | 64,6 | 60,6 |

v3s400 × base: 54/70 (p = 0,18, n.s.). **Não há regressão significativa**: o closed-book
fica no piso (~8%, o modelo não sabe as NRs de cor e a célula serve como controle de
memorização). No recorte o v3s400 tinha dado 7,8 → 9,0, e o ×1536 mostra que aquilo era
ruído. O quadro da promoção do v3s400 está completo nas 4 células ×1536 + e2e + 151.

### (b) rollout P1 completo do SFT — FECHADA (`bench/lfm3_p1p2/data/p1_report.json` atualizado)

Universo 100% medido: data2x **11.151/11.151**, F4 **4.533/4.533**, pool RLVR 5.537 (já
completo). Total de 21.221 itens, n = 4, temp 1,0, mesma política (f4-r32-ep2) e mesma régua.
Corretude 27B nos 3.225 zeros (1.467 novos).

| fonte | n | fácil | produtiva | difícil | mal esp. | % descartado | acc |
|---|--:|--:|--:|--:|--:|--:|--:|
| **total** | 21.221 | 7.922 | 10.074 | 1.255 | 1.970 | **46,6** (era 49,2 parcial) | 0,635 |
| data2x | 11.151 | 3.903 | 5.133 | 807 | 1.308 | 46,7 (era 48,1) | 0,599 |
| F4 | 4.533 | 1.705 | 2.330 | 209 | 289 | 44,0 (era 56,2) | 0,663 |
| pool RLVR | 5.537 | 2.314 | 2.611 | 239 | 373 | 48,5 | 0,686 |

**O viés de comprimento era real, sobretudo na F4.** O parcial tinha medido os prompts mais
curtos primeiro. Os itens novos têm o dobro de prompt (data2x 1.610 → 3.204 tok; F4 1.794 →
4.000) e, na F4, são bem mais difíceis: acc 0,758 → 0,627 nos novos. **Multihop** (403, acc
0,34) e **troca** (603, acc 0,64) nem apareciam no parcial e vão quase inteiros para a banda
produtiva. Com isso o descarte da F4 caiu de 56,2% para 44,0%. Na prática, o dataset-P1 filtrado
(fora do git, `p1_{data2x,f4}_train.jsonl`) passou de **2.349 → 5.940** linhas data2x e de **548 → 2.539** F4.
É esse o insumo do próximo ciclo de dados. As leituras qualitativas do P1 se mantêm:
`sem_ferramenta` 86% fácil, `closed` acc 0,05 (âncora de SFT, nunca alvo de RL).

Correção de bug encontrada no caminho: `read_jsonl` usava `splitlines()`, que quebra em
U+2028 (o modelo gerou esse caractere em 2 rollouts e o `json.dumps(ensure_ascii=False)` grava
cru). As linhas viravam JSON inválido e eram descartadas em silêncio, então o rollout
reentrante repetia o item. Agora o split é por `\n` e o `score` deduplica por uid
(`bench/lfm3_p1p2/p1_pipeline.py`).

## 5. Infra

GPU 3 da H100, coabitando com o `qwen27b_cemig` (que não foi derrubado), no container
`cemig_p1p2_walcy`. As GPUs 0/1/2 não foram tocadas. Filas desacopladas (`h100/queue_*.sh`,
`docker exec -d`), acompanhadas por sondas curtas. Na primeira tentativa, as 4 células ×1536 do
v5s400 caíram por memória: o rollout P1 que a fila anterior encadeava ainda segurava a GPU.
Elas foram refeitas na fila de recuperação `h100/queue_cells.sh`; o `gen_f2.py` é reentrante.
Julgamento local em lotes (`score_p2.py`, 48 workers).

## Reprodução

```
# H100, container cemig_p1p2_walcy (GPU 3); código em /workspace/cemig/p1p2/code (= bench/lfm3_p1p2)
bash queue_v5.sh        # treino v5 + recorte 400/600(/800) + closed ×1536 base/v3s400 + rollout P1
bash queue_full.sh      # bateria ×1536 + e2e ×1536/151 do v5s400 (+ queue_cells.sh se as células caírem)
# local (gerações puxadas p/ /mnt/d/cemig-scratch/p1p2/eval; tags ×1536 = p2-<tag>1536)
PY=classifier/.venv/bin/python
$PY bench/lfm3_p1p2/eval/score_p2.py --tag p2-v5s400            # recorte
$PY bench/lfm3_p1p2/eval/score_p2.py --tag p2-v5s4001536 --skip e2e,probe
$PY bench/lfm3_p1p2/eval/score_p2.py --tag x --e2e-tag p2-v5s400-1536 --skip probe
$PY bench/lfm3_p1p2/eval/score_p2.py --tag x --e2e-tag p2-v5s400-151 --e2e-suite 151 --skip probe
$PY bench/lfm3_p1p2/eval/consolidate_p2.py --tags p2-v3s400 p2-v4s400 p2-v5s400 p2-v5s600 --out bench/lfm3_p2v5/data/consolidation_recorte.json
$PY bench/lfm3_p2v5/consolidate_v5.py --tags base v3s400 v4s400 v5s400
python3 bench/lfm3_p2v5/curves_v5.py --log train_v5.log
# P1 completo: p1_pipeline.py score && correctness && bands (preserve p2_prompts.jsonl: bands o sobrescreve)
```
