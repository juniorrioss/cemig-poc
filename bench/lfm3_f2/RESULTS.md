# RESULTS — F2: injeção de conhecimento no LFM3-1.7B

Recorte de avaliação: 344 queries do benchmark_mobile (6 normas × 4 vocab):
**tratadas NR-10/18/34** (treino viu essas normas) + **controle NR-13/31/33**
(nunca vistas; guarda de especificidade). Régua honesta idêntica ao F0
(cobertura de fatos, juiz 27B, citação fora do gate). Parquets por célula em
`data/eval/` (schema de transparência; ver `data/eval/README.md`).

Baseline F0 (as-is) no MESMO recorte: closed-book **tratadas 8,3% / controle
5,5%**; oráculo 83,2%, ranking 67,7%, distrator 22,4% (agregado do recorte).

## Closed-book — métrica primária (Δ tratadas vs F0)

### Braço A — LoRA no instruct (w1/w2/w3, 72 módulos, 12 épocas, LR 5e-5)

| ep | r8 trat | r8 ctrl | r8 haluc_t | r32 trat | r32 ctrl | r32 haluc_t |
|---|---|---|---|---|---|---|
| 1 | 11,1% | 7,3% | 43,9% | 10,6% | 4,3% | 38,9% |
| 2 | 7,2% | 8,5% | 32,8% | **17,2%** | 7,9% | 36,1% |
| 3 | 12,2% | 5,5% | 41,1% | — | — | — |
| 4 | 10,6% | 4,3% | 49,4% | 8,3% | 6,1% | 64,4% |
| 5 | 11,7% | 8,5% | 53,3% | — | — | — |
| 6 | **15,0%** | 7,3% | 43,9% | 10,6% | 2,4% | 68,9% |
| 7 | 12,2% | 6,7% | 50,0% | — | — | — |
| 8 | 13,9% | 7,3% | 53,9% | 11,1% | 1,2% | 72,2% |
| 9–11 | 12,2–12,8% | 4,9–8,5% | 44–54% | — | — | — |
| 12 | 12,2% | 4,9% | 52,2% | 10,0% | 1,8% | 71,1% |

(A-r8 tem todas as 12 épocas; A-r32 e braços B avaliados nas épocas
informativas 2/4/6/8/12 — decisão registrada após a curva do r8 saturar ~ep6.)

### Braços B e C

| checkpoint | trat | ctrl | haluc_t |
|---|---|---|---|
| B LoRA-no-base puro (r8, melhor ep) | 7,8% | 3,0% | 83–90% |
| B LoRA-no-base puro (r32, melhor ep) | 8,3% | 1,8% | ~86% |
| **B task vector sum-r8** (base+LoRA, +delta) | 7,8% | 3,7% | 26,7% |
| B task vector prop-r8 (delta × média) | 7,2% | 1,8% | 35,6% |
| B task vector sum-r32 | 6,7% | 1,2% | 21,7% |
| B task vector prop-r32 | 2,2% | 1,8% | 22,8% |
| **C K2V (TRL-GRPO, reward answer-gated)** | 7,2% | 4,9% | 45,6% |

## Guardas

| célula | F0 | A-r8-ep6 | A-r8-ep12 | B-taskvec-sum-r8 |
|---|---|---|---|---|
| oráculo | 83,2% | 79,1% | 77,6% | 62,2% |
| ranking | 67,7% | 61,9% | 60,8% | 48,0% |
| distrator (recusa correta) | 22,4% | 32,6% | 38,1% | 42,7% |

Probe de capacidade geral (réguas OOD congeladas, qualidade 1–5 juiz 27B):

| checkpoint | geral | comportamental | pt |
|---|---|---|---|
| F0 as-is | 3,92 | 3,80 | 4,64 |
| A-r8-ep6 | 3,48 | 3,80 | 4,32 |
| A-r32-ep2 | 3,78 | 3,60 | 4,49 |

## Leitura honesta (critério do brief)

1. **A injeção FUNCIONA mas é modesta**: melhor célula A-r32-ep2 **17,2%
   tratadas (+8,9 p.p., 2,1× o F0)** com controle estável (7,9%); A-r8-ep6
   15,0% (+6,7 p.p.). O ganho é específico das normas tratadas (controle
   nunca acompanha) — é conhecimento injetado, não estilo.
2. **Mais época ≠ mais conhecimento**: r32 colapsa cedo (ep2 pico; alucinação
   36→72% até ep12); r8 satura ~ep6. A curva época×conhecimento×esquecimento
   pedida é clara: o que cresce com época é ALUCINAÇÃO, não acerto.
3. **Guardas: tradeoff real** — oráculo/ranking caem 4–7 p.p. no A (uso de
   contexto piora um pouco) e a probe geral cai 0,14–0,44 (r32-ep2 preserva
   melhor que r8-ep6). Distrator "melhora" (22→33–38%) mas em parte porque o
   modelo treinado recusa mais no geral. Pelo critério estrito
   (guardas ESTÁVEIS), **nenhum braço passa limpo**; o mais próximo é
   A-r32-ep2 (ganho máximo, dano mínimo na probe, mas -4 p.p. em oráculo*).
4. **Task vector (hipótese do capitão)**: a re-aplicação do delta FUNCIONA
   como restaurador de comportamento — alucinação despenca de 83–90% (LoRA no
   base puro) para 22–36% —, mas **não preserva o conhecimento injetado**
   (trat ≤7,8%, abaixo do F0 8,3%): a soma do delta "por cima" interfere nos
   w1/w2/w3 treinados. Variante `prop` é pior que `sum` (o re-escalonamento
   atenua exatamente o que foi aprendido). Linha viva: aprender o
   conhecimento JÁ NO espaço do instruct (braço A) dominou o desvio via base.
5. **K2V/GRPO**: +0 p.p. sobre o F0 em closed-book com menos dados (1.222 QAs
   fill-blank vs 5.5k QAs do funil) e muito mais custo (RL + juiz por step).
   RLVR verificável não injetou conhecimento novo neste tamanho/duração; o
   smoke→corrida foi viabilizado via TRL-GRPO porque **vllm/sglang não
   suportam o LFM3** (erro registrado no README).

\* Métricas da melhor época por braço; análise por vocabulário disponível nos
parquets (`data/eval/`).

## Onde está cada coisa

- Parquets canônicos: `data/eval/lfm3-<checkpoint>-<célula>.parquet` (46
  células, inclui baseline retroativo no schema estendido).
- Curvas e dados brutos de treino: wandb offline (Spark
  `~/cemig-poc/lfm3_f2/wandb`, H100 `/workspace/cemig/wandb`; `wandb sync`).
- Dataset sintético + relatório de diversidade: `data/synth/`
  (`diversity_report.md`: entropia 11,0 bits, 0 contaminação).
- Checkpoints: Spark `~/cemig-poc/lfm3_f2/runs/` (A/B LoRA por época,
  task vectors, C-k2v-grpo). Não versionados (pesos).
