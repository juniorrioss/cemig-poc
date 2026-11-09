# RESULTS — P1 (filtro de aprendibilidade) + P2 (mini-RLVR de groundedness) sobre o f4-r32-ep2

Ordem do capitão, depois da revisão do paper Rufus-Air (arXiv 2609.29421): **P1** filtra o
treino pelo que a política atual ainda NÃO domina; **P2** é um RLVR curto com reward binário
verificável por regra, sem juiz no loop, sobre o `f4-r32-ep2`. Avaliação no recorte 344
(normas 10/18/34/13/31/33), 4 células padrão + e2e do app (op_d50) + probe geral, juiz 27B
(régua honesta). Tabela-mestra: `data/consolidation.json`. Relatório de bandas: `data/p1_report.json`.

## Resumo (relatório congelado para o prazo do capitão, 09/11 ~17h UTC)

- **P1 entregue como ferramenta** (`p1_pipeline.py`, 7 subcomandos reentrantes). Num universo de
  11.313 itens medidos (pool RLVR completo + 5.776 do SFT data2x/F4), **49,2% são descartados**:
  39,5% fáceis (acc ≥ 0,8, n=4) e 9,7% mal especificados (acc = 0 e o 27B também falha).
- **Campeão P2 = `v3s400`** (contrato v2 binário + pool com o formato estático): no recorte 344,
  **alucinação no distrator 22,4 → 15,4** (meta < 24,9 ✓; pareado 16 pior/40 melhor, p = 0,002),
  recusa correta 61,0 → 70,3, **e2e do app 68,3 → 75,0** (31/8, p = 0,0003), **cita item-ouro
  69,0 → 73,3%** (meta > 73 ✓), ranking 77,6 → 84,6 (37/13, p = 0,001), oráculo 89,0 → 91,9,
  closed 7,8 → 9,0, probe 3,38 → 3,35 (dentro de 1 p.p.). Custo: +9% de tokens no recorte (86 → 94;
  +14% no app ×1536). Recusa indevida no ranking 3,2 → 2,9 (igual).
- **v4 (reward escalonado) tem a melhor eficiência e a menor alucinação, mas paga em guarda**:
  `frac_reward_zero_std` médio **0,54 (v2) → 0,33 (v4), −40%**; distrator 14,5% (melhor), e2e
  73,8, probe 3,47. Porém oráculo 89,0 → 87,2 (−1,8, fora da tolerância de 1 p.p.), recusa
  indevida 3,2 → 4,4 e ranking +3,5 (não significativo). Por isso não é o campeão.
- **v1/v2 sem formato estático pioram o distrator** (22,4 → 26–28,5): o RL só no formato app
  desaprende a recusar no formato sem ferramenta. A **lição** do v3 é cobrir os dois formatos no
  pool de prompts. Junto com o reward hacking da recusa do v1 (§3), esses são os dois achados.
- **Trava anti-overfit (bateria completa, corte do prazo)**: distrator ×1536 **23,8 → 15,7**
  (fora do recorte 24,2 → 15,9), e2e do app ×1536 **64,1 → 71,8**, POC 151 coloquiais
  **12,6 → 21,2** (p = 0,01). O ganho se sustenta fora das normas do recorte (§2.2).
- **Veredito para o capitão**: o **v3s400 é candidato a substituto do f4-r32-ep2** em tudo que foi
  medido. Ressalvas: (i) oráculo ×1536 88,5 → 90,0 e ranking ×1536
  78,1 → 81,6 (p = 0,0004) confirmados sem regressão; só closed ×1536 não foi rodado; (ii) +14% de tokens; (iii) nas 151 recusa menos e alucina um
  pouco mais (31,1 vs 28,5). Se a prioridade for segurança (nunca inventar) e concisão, o
  **v4s400** é a alternativa: menos alucinação em todas as células, 106 tok, mas o oráculo do
  recorte caiu 1,8 p.p. A escolha v3 × v4 é decisão do capitão.

## 1. P1 — filtro de aprendibilidade

### Régua automática (o contrato, `verifier.py`, sem juiz)

| expect | r = 1 quando |
|---|---|
| answer (ouro no contexto) | cita o item-ouro (código igual ou filho) **E** cobertura determinística ≥ X dos fatos (só fatos "presentes" do `bench/regua/ruler.py`) **E** todo número da resposta existe no contexto ou na pergunta **E** não recusa |
| refuse (ouro ausente) | recusa **explícita** nas primeiras 200 chars (contrato v2, ver §3) |
| partial | cobertura + números (sem citação) |
| closed | citação + cobertura (não há contexto) |
| no_tool (família sem ferramenta do F4) | não emite `<tool_call>` |

Truncada (sem EOS) = 0. Para sobreviver a reformulações, o verificador de números aceita
vírgula ou ponto decimal, separador de milhar, números por extenso no contexto ("oitenta
centímetros" autoriza "80 cm") e itens-pai de códigos citados. O item-ouro é a união da ref
da unidade com os códigos do alvo 27B, **restrita ao que é citável no contexto**: código que
não aparece no contexto não é verificável e sai. Com isso o `no_gold_item` das próprias
referências caiu de 435 para 39 em 7.105. 27 testes: `python3 bench/lfm3_p1p2/test_verifier.py`.

**Calibração de X** (`calibrate`, régua aplicada às referências do 27B):

| X | refs data2x passam | refs F4 passam | refs RLVR passam |
|--:|--:|--:|--:|
| 0,34 | 81,1% | 86,0% | 83,5% |
| **0,5** | **81,1%** | **86,0%** | **83,5%** |
| 0,67 | 80,4% | 85,0% | — |
| 0,8 | 79,8% | 84,2% | — |

Nas referências, X quase não move nada; o que reprova uma referência é ressalva tipo recusa
("a norma não traz…", 18%) ou número fora do contexto. Nos rollouts da política, X = 0,8 empurra
37 dos 256 itens do smoke da banda fácil para zero. **X = 0,5** (o mesmo limiar da régua
honesta) dá a maior banda produtiva com sinal estável.

### Bandas (política `f4-r32-ep2`, n = 4, temp 1,0, max 512 tok)

| fonte | n | fácil (descarta) | produtiva | difícil (27B resolve) | mal especificado (descarta) | % descartado | acc média |
|---|--:|--:|--:|--:|--:|--:|--:|
| **total** | 11.313 | 4.469 | 5.086 | 661 | 1.097 | **49,2** | 0,646 |
| pool RLVR (formato app) | 5.537 | 2.314 | 2.611 | 239 | 373 | 48,5 | 0,686 |
| SFT data2x | 4.524 | 1.497 | 1.973 | 376 | 678 | 48,1 | 0,567 |
| SFT F4 (diálogos) | 1.252 | 658 | 502 | 46 | 46 | 56,2 | 0,758 |

Por família (destaques; tabela completa em `data/p1_report.json`, que também traz o recorte
por NR, com acc média de 0,55 a 0,71 entre as normas):

| família | n | fácil | produtiva | difícil | mal esp. | acc |
|---|--:|--:|--:|--:|--:|--:|
| data2x gold | 2.402 | 895 | 1.214 | 124 | 169 | 0,65 |
| data2x absent (recusa) | 1.113 | 527 | 478 | 8 | 100 | 0,72 |
| data2x partial | 453 | 74 | 200 | 94 | 85 | 0,36 |
| data2x closed | 556 | 1 | 81 | 150 | 324 | **0,05** |
| F4 sem_ferramenta | 245 | 211 | 25 | 9 | 0 | **0,93** |
| F4 recusa | 396 | 224 | 158 | 2 | 12 | 0,81 |
| RLVR answer (ouro no top-5→50%) | 4.130 | 1.308 | 2.230 | 228 | 364 | 0,62 |
| RLVR ouro removido (recusa) | 1.264 | 926 | 325 | 7 | 6 | **0,90** |

Leituras:
- **Fácil e não ensina mais**: sem_ferramenta (86% descartado) e recusa com ouro removido
  (74%). O f4-r32-ep2 já domina os dois.
- **closed-book é quase só ruído para RL**: acc 0,05, e 58% dos zeros nem o 27B resolve sem
  contexto. Âncora de SFT, nunca alvo de RLVR.
- **Mal especificados** (`data/p1_ill_specified.jsonl`, 1.097 itens) são sobretudo alvos com
  ref de item que não bate com o contexto (o 27B cita outro item, `no_gold_item`) e perguntas
  que o contexto real cobre só em parte. Vale auditar antes de reaproveitar.
- **Cobertura parcial (honesta)**: o rollout do SFT foi cortado pelas janelas de treino P2
  (pendência; não afeta o P2, cujo pool RLVR está 100% medido e cujos itens data2x do v3/v4 vêm da
  parte medida).
  Estão medidos 4.524/11.151 data2x e 1.252/4.533 F4, amostra aleatória por ordem de comprimento
  de prompt (`rollout` ordena por tamanho, então **os prompts mais longos, k = 8, estão
  sub-representados**). O pool RLVR está 100% medido. Para completar: `rollout --sources data2x
  f4` (reentrante, ~1 h de GPU) → `score` → `correctness` → `bands`.

### Pool RLVR (contexto real do app)

Perguntas do data2x passaram pelo **retriever op da F6** (gate 0,7/2x + manuais 1,75 margem,
top-5 → compressão 50%, paridade byte a byte com os prompts congelados da F6), no formato do
app: system com a ferramenta + user + tool_call(fala crua) + tool. O rótulo vem do containment
do fato da unidade no contexto comprimido: ≥ 0,70 → answer (4.130); < 0,30 → refuse (143, misses
genuínos). Ambíguos ficam de fora. Para ter **ouro ausente com contexto real** em volume, a mesma
pergunta recebe o ranking op top-12 sem os chunks que contêm o fato, depois top-5 → 50% com o
mesmo orçamento (1.264 itens, `rlvr_refuse_removed`).

## 2. P2 — mini-RLVR

**Receita** (`train_p2_rlvr.py`): TRL 1.17 GRPO com `loss_type="dapo"`, clip-higher
(ε 0,2/0,28), `mask_truncated_completions` (overlong filtering) e β = 0. LoRA **novo** r32 α64 em
w1/w2/w3 sobre o f4-r32-ep2 **mesclado** (`merge_policy.py`, md5 `7bafcf52…`), com os asserts de
sempre (72 lora_A em feed_forward, 0 em self_attn/gate_proj/conv). 8 prompts × 8 gerações por
step, temp 1,0, max 384 tok, lr 5e-6 constante, 400 steps, ~11 s/step (~75 min) na **GPU 3
compartilhada com o Qwen** (sem troca). O prompt entra como **texto já renderizado** por
`apply_chat_template(enable_thinking=False)`: é byte a byte o que o app entrega. Filtro dinâmico:
offline pelo P1 e online pela vantagem zero dos grupos degenerados
(`frac_reward_zero_std` logado, 0,15 → 0,50 ao longo do treino).

**Runs** (wandb `lfm3-1.7b-cemig`; todos 400 steps, lr 5e-6, 8×8):

| run | pool de prompts | contrato do reward | wandb |
|---|---|---|---|
| `lfm3-p2-rlvr` (s200/s400) | RLVR (formato app) produtiva+difícil, recusa ×2 (19%) | v1 binário | `c5cepsnk` |
| `lfm3-p2-rlvr-bal` | + todas as recusas fáceis (36%) | v1 | — |
| `lfm3-p2-rlvr-v2` | âncoras de recusa até 30% | **v2** binário (recusa explícita no início) | `5h5m9jt4` |
| **`lfm3-p2-rlvr-v3`** | v2 + data2x no **formato estático** (system F0, contexto cru, distratores da mesma NR), recusa 35%; prompts ≤ 2.200 tok (5.369) | v2 | `o4kcs0vz` |
| `lfm3-p2-rlvr-v4` (v4s200/300/400) | pool v3 refiltrado pelo reward v4 (grupos com variância, 6.044 → ≤2.200 tok) | **v4 escalonado** (§2.3) | `bchs9c0h` |

### 2.1 Avaliação no recorte 344 (juiz 27B) — v2 × v3 × v4

| tag | ranking | rec-indev. | **distrator aluc.** | distr. recusa | oráculo | closed | **e2e app** | **cita item (app)** | aluc. app | probe | tok |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **f4-r32-ep2 (base)** | 77,6 | 3,2 | **22,4** | 61,0 | 89,0 | 7,8 | 68,3 | 69,0 | 14,5 | 3,38 | 86,2 |
| v1 s400 | 83,7 | 1,5 | 26,2 | 48,0 | 91,6 | 8,1 | 73,3 | 73,0 | 11,6 | 3,32 | 91,2 |
| bal s400 | 83,4 | 1,2 | 28,2 | 45,1 | 91,0 | 7,0 | 73,5 | 72,7 | 12,5 | 3,31 | 90,1 |
| v2 s200 | 84,0 | 0,9 | 25,3 | 51,5 | 90,1 | 8,4 | 72,7 | 71,0 | 13,1 | 3,33 | 87,7 |
| v2 s400 | 83,7 | 0,9 | 28,5 | 46,8 | 91,3 | 7,0 | 73,8 | **74,0** | 12,8 | 3,37 | 91,8 |
| v3 s200 | 82,3 | 1,7 | 18,6 | 66,9 | 89,8 | 5,8 | 73,8 | 71,0 | 12,8 | 3,33 | 86,4 |
| v3 s300 | 83,1 | 2,6 | 14,8 | 68,0 | 89,8 | 7,6 | 73,3 | 72,7 | 12,8 | **3,40** | 91,6 |
| **v3 s400 ⇐ campeão** | **84,6** | 2,9 | 15,4 | 70,3 | **91,9** | **9,0** | **75,0** | 73,3 | 13,1 | 3,35 | 94,3 |
| v4 s200 | 83,1 | 1,5 | 19,5 | 64,2 | 91,3 | 7,3 | 71,8 | 71,0 | 13,7 | 3,32 | 88,5 |
| v4 s300 | 81,1 | 1,2 | 22,7 | 59,9 | 92,4 | 8,1 | 73,8 | 71,3 | 13,1 | 3,37 | 87,3 |
| v4 s400 | 81,1 | 4,4 | **14,5** | **72,7** | 87,2 | 7,6 | 73,8 | 71,7 | **12,5** | **3,47** | 86,7 |

Pareado vs base (aprovação, teste do sinal): **v3s400** e2e 31/8 (p = 0,0003), ranking 37/13
(p = 0,001), distrator 41/9 (p < 1e-5), oráculo 21/11 (p = 0,11), closed 17/13 (n.s.);
alucinação no distrator pior/melhor **16/40 (p = 0,002)**. **v4s400**: e2e 28/9 (p = 0,003),
ranking 30/18 (n.s.), distrator 47/7, **oráculo 18/24 (n.s., mas −1,8)**; alucinação 14/41
(p = 0,0004). v2s400: distrator 1/50 (piora, p < 1e-6).

**Escolha do campeão**: o v3s400 é o único ponto que cruza as duas metas (distrator < 24,9,
cita item > 73) sem regredir nenhuma guarda em mais de 1 p.p. O v4s400 tem menos alucinação
e mais probe, mas o oráculo cai 1,8 e a recusa indevida sobe 1,2: fica fora da tolerância.

### 2.2 Trava anti-overfit — bateria completa (1536 + 151)

O recorte 344 compartilha normas com o treino (10/18/34 estão no data2x). Pela ordem do capitão,
nenhum substituto é declarado sem a bateria completa. Corte do prazo: **distrator ×1536 + e2e do
app ×1536 + e2e ×151 coloquiais** (régua honesta, gabarito 151). Oráculo, closed e ranking ×1536
ficam para depois do prazo.

Base, v3s400 e v4s400 (gerados no mesmo harness e GPU; juiz 27B). "Fora do recorte" = as
1.192 queries das normas que NÃO estão no recorte 344, ou seja, o teste sem normas
compartilhadas com a avaliação rápida. `data/consolidation_full.json`.

| tag | escopo | distrator aluc. | distr. recusa correta | e2e app aprov. | cita item-ouro | aluc. app | tok resposta app |
|---|---|--:|--:|--:|--:|--:|--:|
| base | 1536 | 23,8 | 66,0 | 64,1 | 64,3 | 14,1 | 103,8 |
| base | fora do recorte (1192) | 24,2 | 67,4 | 62,7 | 63,0 | 15,0 | 104,2 |
| **v3s400** | 1536 | **15,7** | 73,1 | **71,8** | **69,2** | 15,0 | 118,0 |
| **v3s400** | fora do recorte | **15,9** | 73,9 | **70,8** | 67,9 | 15,7 | 118,0 |
| v4s400 | 1536 | **13,7** | **75,9** | 70,4 | 67,4 | **14,0** | **106,0** |
| v4s400 | fora do recorte | 14,0 | 76,8 | 69,7 | 66,4 | 14,1 | 105,9 |

| tag | POC 151 coloquiais (régua honesta) | alucinação | recusa | pareado vs base |
|---|--:|--:|--:|---|
| base | 12,6 | 28,5 | 53,0 | — |
| **v3s400** | **21,2** | 31,1 | 35,1 | 18/5 (p = 0,011) |
| v4s400 | 17,9 | **25,8** | 45,7 | 15/7 (p = 0,13) |

Pareados (1536): e2e v3s400 177/59, v4s400 152/56 (ambos p < 1e-6); alucinação no distrator
pior/melhor v3s400 52/177, v4s400 48/204 (ambos p < 1e-6). Alucinação no app (e2e) v3s400
pior/melhor 102/88 (p = 0,35, n.s.).

**Leitura da trava**: o ganho **não é overfit às normas do recorte**. Fora do recorte o v3s400
repete o padrão (distrator 24,2 → 15,9; app 62,7 → 70,8), e nas 151 coloquiais da POC (outro
gabarito, perguntas reais de campo) a aprovação vai de 12,6 para 21,2. **Residual honesto**:
nas 151 o v3s400 recusa menos (53 → 35%) e a alucinação sobe 28,5 → 31,1 (das 28 perguntas
em que deixou de recusar, 8 viraram aprovadas e 16 alucinaram); no app ×1536 a alucinação
fica igual (14,1 → 15,0, n.s.). O v3s400 também gasta **+14% de tokens** no app (104 → 118).
O v4s400 é o par mais conservador: menos alucinação em todo lugar (distrator 13,7; 151 com
25,8) e conciso (106 tok), com aprovação um pouco menor (app 70,4; 151 com 17,9) e a regressão
de oráculo do recorte (−1,8, ainda não medida ×1536).

**Guardas ×1536 (rodadas na folga do prazo, base × v3s400)**:

| célula ×1536 | base | v3s400 | pareado | recusa indevida | tok |
|---|--:|--:|---|---|---|
| oráculo | 88,5 | **90,0** | 97/74 (p = 0,09) | 4,9 → 3,3 | 96 → 105 |
| ranking | 78,1 | **81,6** | 139/85 (p = 0,0004) | 7,1 → 6,2 | 101 → 109 |

**Sem regressão nas guardas da bateria completa.** Não medidos: closed-book ×1536 (no recorte
7,8 → 9,0) e as guardas ×1536 do v4s400 (o oráculo dele caiu 1,8 no recorte). O probe (130
itens, sem relação com as normas) foi medido uma vez: 3,38 → 3,35.

### 2.3 Contrato v4 (reward escalonado) e curva de eficiência

`verify_v4` (38 testes no total): **gate duro** (número fora do contexto, truncada, ferramenta
espúria ou recusa indevida = 0); para quem passa, **r = 0,25 + 0,5·cobertura contínua +
0,25·citação**. Multi-evidência: com ≥ 2 raízes distintas de item-ouro citáveis, a citação vale
a fração das raízes citadas (o bônus cheio exige os dois spans; o código de item é a identidade
verificável de cada span, já que os fatos do gabarito não são separáveis por span). Concisão:
**×(1 − min(0,3; 0,3·(tok − 144)/144))**, com T = 144 = 1,5 × a mediana (96 tok) de 11.235
referências do 27B (p90 = 143). Recusa continua binária (v2).

**frac_reward_zero_std** (fração de grupos sem gradiente), média por janela de 50 steps
(`data/efficiency_curves.json`, brutos em `data/curves/`):

| run | 1–50 | 51–100 | 101–150 | 151–200 | 201–250 | 251–300 | 301–350 | 351–400 | **média** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| v1 | 0,17 | 0,30 | 0,38 | 0,47 | 0,46 | 0,50 | 0,59 | 0,63 | 0,44 |
| bal | 0,36 | 0,40 | 0,48 | 0,55 | 0,64 | 0,66 | 0,69 | 0,71 | 0,56 |
| **v2** | 0,31 | 0,34 | 0,51 | 0,54 | 0,58 | 0,61 | 0,70 | 0,74 | **0,54** |
| v3 | 0,32 | 0,36 | 0,45 | 0,48 | 0,55 | 0,59 | 0,65 | 0,69 | 0,51 |
| **v4** | 0,18 | 0,24 | 0,28 | 0,34 | 0,38 | 0,33 | 0,40 | 0,45 | **0,33** |

O escalonado **derruba os grupos sem gradiente em 40% (0,54 → 0,33)**, e a curva cresce mais
devagar (0,45 contra 0,74 no fim): o binário satura conforme a política aprende, o escalonado
continua separando boas de excelentes. O resíduo do v4 vem quase todo das **recusas**, que seguem
binárias (a priori: grupos degenerados em answer 9,3% → 4,3%; em refuse 63,8% = 63,8%). Tokens
médios no treino: v4 86–89, sem a deriva para cima do v1 (95–100). Reward médio por janela: v2
0,69 → 0,87; v4 0,71 → 0,86 (escalas diferentes; não comparar em absoluto).

### 2.4 Notas da avaliação

**Linha de base regerada, não copiada**: a base foi regerada no mesmo harness e na mesma GPU que
os treinados, para o pareamento ser justo. Contra os números da F4 no mesmo recorte, distrator,
oráculo e closed-book batem; o **ranking dá 77,6 contra 83,1** (pareado 12/31) com prompts
idênticos. O mesmo ranking com adapter on-the-fly (como a F4 gerou) dá 80,8. A diferença
merged-bf16 × adapter + o sampling 0,1 explica parte; o resto é ruído de amostragem (90/344
respostas byte-idênticas entre as duas gerações da base). **Ressalva**: contra a referência
histórica da F4 (83,1), o ganho de ranking do P2 encolhe para ~+1.

## 3. Achado central: reward hacking da recusa (contrato v1 → v2)

Com o contrato v1, "recusa" era o regex amplo `looks_like_refusal` em qualquer ponto da
resposta. A política aprendeu a **responder e fechar com uma ressalva** ("…A norma não traz
outra exigência além dessa"). O regex contava isso como recusa: reward 1 com ouro ausente e,
com ouro presente, a régua de avaliação conta como recusa indevida. No distrator, a **recusa
explícita** caiu de **185 para 40/344** (s400) e a alucinação subiu. É o "recompensa dura
primeiro" do paper mostrando o porquê: o verificador tem de medir o comportamento, não um
proxy textual.

O contrato **v2** (`STRICT_REFUSAL_RE` nas primeiras 200 chars; sujeito = material recuperado,
nunca "a norma" genérica) passa 3.334/3.334 alvos de recusa do SFT e nunca marcou como recusa
uma resposta alucinada do distrator (0 em 201 explícitas da base). Mesmo assim, o v2 **não
consertou o distrator estático** (28,5 em v2s400), porque o pool RLVR treina só no formato app
(turno tool, top-5 comprimido), onde a recusa já era 0,90+ antes do treino. A célula distrator
usa outro formato (system F0 sem ferramenta, 4 chunks crus da mesma norma) e lá o RL
desaprende a recusar. **O v3 confirmou a hipótese**: com itens data2x no formato estático no
pool (recusa 35%), o distrator vai de 28,5 (v2s400) para **15,4** (v3s400) com o MESMO contrato,
e o modo app não perde (73,8 → 75,0).

## 4. Infra (GPU 3 compartilhada, sem alternância)

- O endpoint 8005 virou exclusivamente nosso (inbox 001). Em vez de alternar janelas,
  **recriei o Qwen uma vez** (`qwen27b_cemig`, mesmo comando de referência com
  `--gpu-memory-utilization 0.62 --max-model-len 32768`; KV 245k tokens, ~29 GB livres), fora
  do ar ~6 min. Desde então LFM3 (rollouts, GRPO, eval) e Qwen (fatos, corretude, juiz)
  rodaram **simultaneamente** na GPU 3. Container próprio `cemig_p1p2_walcy` (snapshot do
  `cemig_mobile_walcy`, `--gpus device=3`).
- **GPU 0 nunca ficou ociosa** (100% nas sondas a cada 60 s, `logs/gpu_monitor.log` no nó):
  a autorização de matar o processo dela não se aplicou.
- vLLM não serve LFM3 (conferido nas 3 imagens do nó, nem a 0.31): rollout via HF `generate`
  em lote, com orçamento de tokens por lote (~3,5 it/s a n = 4).
- Sharp edges: 64 gerações com prompts de 2–3k tok (formato estático) estouram a folga da GPU →
  `--micro-batch 4 --max-prompt-tokens 2200`; o bash relê um script editado durante a execução
  (nunca editar `run_eval_p2.sh` com fila rodando); `chmod a+rX` nos ckpts (nascem root).

## Artefatos

- Código: `p1_pipeline.py` (P1), `verifier.py` + `test_verifier.py` (contrato), `build_p2_pool.py`,
  `merge_policy.py`, `train_p2_rlvr.py`, `eval/{run_eval_p2.sh,score_p2.py,consolidate_p2.py}`,
  `eval/prompts_op_d50.jsonl` (344 prompts congelados da F6).
- Dados versionados: `data/p1_report.json`, `data/p1_manifest.jsonl` (uid/fonte/família/
  row_index/acc/banda — o dataset-P1 filtrado é reconstruível a partir do train original),
  `data/p1_ill_specified.jsonl`, `data/consolidation.json`.
- Fora do git (H100 `/raid/aluno_walcy/cemig_mobile/p1p2/` = `/workspace/cemig/p1p2`; cópia
  local `/mnt/d/cemig-scratch/p1p2/`): universos, fatos (`facts.ckpt.json`, 14.101 chaves),
  rollouts, scores, bandas, `p1_{data2x,f4}_train.jsonl` (dataset-P1 filtrado),
  `p2_prompts_*.jsonl`, gerações e scored da avaliação. Checkpoints:
  `/workspace/cemig/checkpoints/lfm3-p2-rlvr{,-bal,-v2,-v3}/checkpoint-{50..400}`. Política de
  partida mesclada: `/workspace/cemig/p1p2/f4-r32-ep2-merged`.

## Reprodução

```
PY=classifier/.venv/bin/python   # local; encoder :8399 p/ o pool RLVR
$PY bench/lfm3_p1p2/p1_pipeline.py build            # universo (data2x, f4, rlvr)
$PY bench/lfm3_p1p2/p1_pipeline.py facts            # 27B
$PY bench/lfm3_p1p2/p1_pipeline.py calibrate
# container GPU 3: python p1_pipeline.py rollout --model .../f4-r32-ep2-merged --tag f4ep2
$PY bench/lfm3_p1p2/p1_pipeline.py score && ... correctness && ... bands
$PY bench/lfm3_p1p2/build_p2_pool.py --pool v2
# container: python train_p2_rlvr.py --run lfm3-p2-rlvr-v2 --prompts .../p2_prompts_v2.jsonl --refuse-oversample 1
# container: bash run_eval_p2.sh TAG CKPT 10,18,34,13,31,33 ; local: eval/score_p2.py + eval/consolidate_p2.py
```
