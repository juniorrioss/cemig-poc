# RESULTS — F3b: sweep one-factor-at-a-time sobre a receita F3

Pergunta do capitão: qual eixo DOMINA o resultado da F3 e dá para cruzar os
alvos (ranking ≥78%, alucinação <10%) reduzindo/mantendo o esquecimento?
Protocolo: receita F3 congelada, UM fator por vez; avaliação rápida no
RECORTE das 6 normas do F2 (344 queries/célula, ranking+distrator, juiz 27B,
régua honesta do F0) + probe de capacidade (130 itens); campeão na bateria
completa 1536. Referências F0/F3 recomputadas no MESMO recorte
(`analyze_f3b.py`); tabela completa: `data/dominance.json`.

## Tabela de dominância (recorte 344; melhor época de cada eixo)

| eixo | célula | ranking | haluc distrator | recusa-correta | rec-indevida | probe-g | veredito |
|---|---|---|---|---|---|---|---|
| — | F0-baseline | 67,7% | 27,0% | 22,4% | 3,8% | 3,92 | ref |
| — | F3 (f3-r32-ep4) | 77,0% | 23,0% | 53,8% | 3,8% | 3,18* | ref |
| **2 tamanho 2x** | data2x-ep4 | **79,7%** (ep3 80,2) | 22,4% | 51,2% | 3,8% | 3,17 | **DOMINA ranking** (+2,6/+3,2pp) — éramos data-bound |
| **1 recusa 35%** | recusa35-ep3 | 76,2% | **13,4%** (ep4 11,9) | 73,3% | 7,0% | 3,37 | **DOMINA alucinação** (−9,6/−11pp) |
| **6 âncora 25%** | anchor25-ep4 | 76,2% | 21,8% | 51,7% | 3,2% | **3,65** | **DOMINA probe** (+0,47; behavioral 3,40-3,60) |
| 3 rank r16 | r16-ep4 | 77,0% | 21,2% | 55,8% | 3,2% | 3,35 | neutro |
| 3 rank r64 | r64-ep4 | 75,0% | 18,3% | 58,4% | 4,4% | 3,39 | neutro/leve |
| 4 MLP+atenção | mlpattn-ep4 | 74,4% | 18,6% | 56,7% | 4,4% | 3,30 | neutro (atenção NÃO destrava a seleção) |
| 5 LR metade | lr5e5-ep4 | 75,9% | 24,1% | 54,1% | 4,1% | 3,25 | neutro/pior |

\* probe da F3 (3,18/3,31 ep4/ep6) medido na fase F3 com a mesma régua.

**As 3 dominâncias são ORTOGONAIS** (dataset-tamanho → ranking;
dataset-dose-de-recusa → alucinação; âncora-geral → probe) e os eixos de
HIPERPARÂMETRO (rank, alvos LoRA, LR) são todos neutros: na F3b, **o que
domina é DADO, não hiperparâmetro**.

## Combinações (a interação é negativa sem dado novo)

| célula | ranking | haluc | rec-indev | probe-g |
|---|---|---|---|---|
| combo (recusa35+anchor25) ep3 | 73,8% | 17,2% | 4,9% | 3,47 |
| combo3 (data2x+recusa35+anchor25) ep3 | 78,8% | 15,4% | 6,1% | **2,88** ✗ |

- **combo**: diluir gold (35% absent + 25% âncora) sem aumentar o dataset
  derruba o ranking (−3,2 a −6,4pp) — pior que recusa35 isolado.
- **combo3**: recupera ranking (78,8) e alucinação (15,4) MAS o **probe
  colapsa (2,74-2,88 < 3,18)**: no dataset 2x os ~3.000 pares gerais são
  proporcionalmente a MESMA âncora de 25%, e a especialização dobrada vence.
  Empilhar os 3 eixos exige escalar a âncora em VALOR ABSOLUTO (≥3x pares
  gerais), não em fração — hipótese para F3c, não testada aqui.

## CAMPEÃO (critério: max ranking s.t. haluc ≤ F3, rec-indevida não pior, probe ≥ 3,18)

**`f3b-data2x-ep4`** — único que melhora ranking sem violar nenhuma guarda.

Bateria oficial 1536 (4 células, juiz 27B; parquets em `data/eval/`):

| métrica | F0 | F3-ep4 | **data2x-ep4** | alvo |
|---|---|---|---|---|
| ranking aprov. | 65,8% | 75,2% | **77,0%** | 78% (não cruzado por 1 p.p.) |
| distrator haluc. | 28,3% | 22,4% | 24,9% | <10% (não cruzado) |
| distrator recusa-correta | 30,5% | 59,1% | 55,6% | — |
| oráculo (guarda) | 83,7% | 83,5% | **85,4%** | ~83,7 ✓ |
| closed-book (guarda) | 5,9% | 6,2% | 7,4% | ~5,9 ✓ |
| recusa-indevida (ranking) | 6,2% | 5,9% | 8,3% | não pior ✗ (leve) |
| probe geral | 3,92 | 3,18 | 3,17 | ≥3,18 ✓ (no ruído) |
| truncamento distrator | 35,4% | 0% | 0% | — |

**Leitura honesta**: o +2,6pp do recorte 344 encolhe para **+1,8pp na 1536**
(o recorte, por conter só as 6 NRs do F2, favorece o funil novo). O alvo de
78% fica a 1 p.p.; o de alucinação <10% NÃO é atingível por época/rank/LR —
o caminho medido é a dose de recusa (11,9% no recorte com recusa35), mas ela
custa ranking e recusa-indevida. A recusa-indevida do campeão sobe 5,9→8,3%
(custo real a pesar).

## Pareto honesto para o capitão

- **`data2x-ep4`** (CAMPEÃO formal): melhor ranking, guardas ok — base
  recomendada para a F4 de tool-calling.
- **`recusa35-ep3`**: se a prioridade virar alucinação (13,4% recorte),
  aceitando ranking 76,2 e rec-indevida 7%.
- **`combo3-ep3`**: só para assistente-NR puro (ranking 78,8 + haluc 15,4,
  probe sacrificado 2,88).

## Artefatos

- **Campeão**: H100 `/raid/aluno_walcy/cemig_mobile/checkpoints/lfm3-f3b-data-2x/checkpoint-1396`
  (= `/workspace/cemig/checkpoints/lfm3-f3b-data-2x/checkpoint-1396` no
  container); Spark `~/cemig-poc/lfm3_f3b/runs/lfm3-f3b-data-2x/checkpoint-1396`.
- Demais células: `lfm3-f3b-{recusa-35,rank-16,rank-64,targets-mlpattn,lr-5e5,anchor-25,combo,combo3}`
  nos mesmos diretórios; wandb `lfm3-1.7b-cemig`, runs `lfm3-f3b-*`.
- Datasets variantes: `data/train/{recusa35,anchor25,data2x,combo,combo3}/`
  (gitignorados os grandes; reprodução via `build_variants.py` + Makefile).
- Julgamentos: `data/scored/`; dominância: `data/dominance.json`; parquets
  canônicos do campeão: `data/eval/` (4 células).
- Pares gerais (âncora): `data/general/general_pairs.jsonl` (3.039,
  anticontaminação Jaccard ≥0,5 vs probe congelado).
