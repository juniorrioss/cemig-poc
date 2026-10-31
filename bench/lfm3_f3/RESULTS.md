# RESULTS — F3: robustez de uso de contexto (LoRA r32 sobre LFM3-1.7B instruct limpo)

Grade completa: 8 épocas × 4 células × 1536 queries, juiz 27B, régua honesta do
F0 (mesma seed de embaralhamento, mesmo system prompt). Parquets canônicos em
`data/eval/` (schema estendido). Métricas completas: `data/consolidation.json`.

## Tabela mestre (all, 1536 queries por célula)

| checkpoint | ranking aprov. (alvo ≥78%) | distrator alucin. (alvo <10%) | distrator recusa-correta | oráculo aprov. (guarda ~83,7%) | closed-book aprov. (guarda ~5,9%) | recusa-indevida (ranking) | recusa-indevida (oráculo) |
|---|---|---|---|---|---|---|---|
| F0-baseline | 65,8% | 28,3% | 30,5% | 83,7% | 5,9% | 6,2% | 5,9% |
| f3-r32-ep1 | 74,2% | 25,7% | 56,6% | 80,7% | 5,0% | 5,1% | 3,1% |
| f3-r32-ep2 | 72,5% | 26,0% | 50,6% | 80,8% | 6,1% | 5,6% | 3,5% |
| f3-r32-ep3 | 74,9% | 22,8% | 56,4% | 80,5% | 6,8% | 4,9% | 3,5% |
| **f3-r32-ep4** | **75,2%** | **22,4%** | **59,1%** | **83,5%** | **6,2%** | **5,9%** | **3,1%** |
| f3-r32-ep5 | 75,5% | 27,8% | 45,9% | 84,7% | 5,9% | 5,2% | 3,4% |
| **f3-r32-ep6** | 74,4% | **21,3%** | **60,8%** | 82,9% | 6,8% | 5,7% | 4,2% |
| f3-r32-ep7 | 73,8% | 25,1% | 51,4% | 83,5% | 6,1% | 5,3% | 3,9% |
| f3-r32-ep8 | 74,5% | 23,5% | 54,8% | 82,9% | 4,8% | 5,0% | 3,6% |

eval_loss (val interno 139): 0,587 → 0,553 → 0,550 → 0,564 → 0,569 → 0,600 →
0,632 → 0,651 (mínimo em ep3; régua oscila num platô de ep4 a ep8).

**Candidatos**: `ep4` (melhor equilíbrio: ranking 75,2 + guardas intactas +
recusa-indevida igual à do F0) ou `ep6` (menor alucinação 21,3 / maior recusa
correta 60,8, ranking 0,8 p.p. abaixo). Probe de capacidade favorece ep6
(3,31 vs 3,18) — ver limite 3 abaixo.

## Veredito honesto vs alvos

1. **Ranking-1-de-5: 65,8 → 75,2% (+9,4 p.p.; alvo 78% NÃO cruzado, faltam
   ~3 p.p.)**. O ganho fecha ~52% do gap F0→oráculo (65,8→83,7). Por
   vocabulário (ep4): clean 70,6→82,0, technical 73,2→82,3, operational
   62,0→73,7, lay 57,6→62,8 — clean/technical já operam NO nível do oráculo
   do F0; o teto restante é o *lay* (abismo lexical conhecido da POC).
2. **Alucinação distrator-puro: 28,3 → 21,3-22,4% (alvo <10% NÃO cruzado)**,
   MAS a decomposição muda de natureza: recusa correta 30,5→59,1-60,8%
   (**2×**), truncamento 35,4→0,0% (o F0 ruminava no thinking até estourar
   3072 tokens; o F3 responde/recusa em ~110 tokens), e 17-18% seguem
   acertando do conhecimento interno mesmo sem contexto útil. No lay a
   alucinação cai 42,2→25,0%.
3. **Guardas**: oráculo 83,5% (ep4; ≈83,7 do F0, dentro do ruído), closed-book
   6,2% (≈5,9), recusa-indevida nas respondíveis MENOR que o F0 (ranking 5,9
   vs 6,2; oráculo 3,1 vs 5,9) — **o trade-off da sft_v3 (recusa-indevida
   alta como preço da robustez) NÃO ocorreu**; o treino de recusa ficou
   contido no caso certo. **Probe de capacidade geral: HÁ custo** — qualidade
   geral 3,92 (F0) → 3,18 (ep4) / 3,31 (ep6), comportamental 3,80 → 2,83/3,00
   (PT-BR intacto 4,3-4,4). Mesmo padrão de forgetting do 1.2B da POC
   (capacidade pequena não absorve especialização de graça); amostras ruins
   são respostas mais curtas/rasas em temas gerais, não contaminação do
   formato de recusa (8/55 notas ≤2 com fraseologia de recusa).

## Decomposição do distrator (o que mudou de verdade)

| | F0 | ep4 | ep6 |
|---|---|---|---|
| recusa correta | 30,5% | 59,1% | 60,8% |
| alucinação | 28,3% | 22,4% | 21,3% |
| acertou do conhecimento interno | 20,2% | 18,4% | 17,5% |
| truncou ruminando | 35,4% | 0,0% | 0,0% |

## Custo/latência implícito

Tokens médios por resposta: F0 ~700-1500 (thinking longo) → F3 ~110 (emite
`</think>` imediato e responde direto). A geração das 4 células caiu de ~14 h
(F0, GB10) para ~20-40 min/célula (Spark/H100, batch 48).

## Limites e leitura

- Os 2 alvos numéricos NÃO foram cruzados; o que o treino de robustez compra
  sozinho (com o retriever real como fonte de distratores) está no platô
  74-76% / 21-26% — épocas extras oscilam sem tendência (eval_loss já sobe).
- O resíduo de alucinação concentra-se em *operational/lay* (25-29%) — mesmo
  abismo lexical que limita o retrieval da POC; alavanca seguinte não é mais
  época/rank, é dado (paráfrases lay dos casos absent) ou capacidade.
- O forgetting do probe é o custo real a pesar antes de promover o adapter:
  se o uso é assistente de NR puro, é aceitável; se o modelo serve a
  perguntas gerais, não embarcar sem mitigação (mix de dados gerais).
- `answered_correct_anyway` (18%) no distrator é comportamento legítimo
  (conhecimento interno correto), mas o alvo de treino pedia recusa — há
  tensão de desenho entre "recusar sem contexto útil" e "usar o que sabe".

## Artefatos

- Adapters (H100): `/raid/aluno_walcy/cemig_mobile/checkpoints/lfm3-f3-robustez-r32/checkpoint-{218,436,654,872,1090,1308,1526,1744}`
  (ep1-8; cópias ep1-6 na Spark `~/cemig-poc/lfm3_f3/runs/r32-ep*`).
- wandb: `lfm3-1.7b-cemig/lfm3-f3-robustez-r32` (kde9xyam, ep1-4) e
  `lfm3-f3-robustez-r32-ext8` (rcavhk37, ep5-8).
- Gerações brutas: `data/gen/` (gitignored, reproduzíveis); vereditos:
  `data/scored/`; parquets canônicos: `data/eval/` (32 células + registry).
- Probe: `bench/lfm3_f2/probe/probe_scored_F3-r32-ep{4,6}.jsonl`.
