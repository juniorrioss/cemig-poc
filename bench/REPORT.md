# Relatório vivo — Exploração LFM3-1.7B (F0 → F5)

> **Documento INCREMENTAL.** Fonte canônica: este `bench/REPORT.md`; o
> `bench/REPORT.html` é gerado dele (`python3 bench/report_html.py`). A cada
> fase nova: (1) reescrever o Resumo executivo, (2) atualizar a Tabela-mestra,
> (3) acrescentar a seção da fase com data, (4) estender o Apêndice de
> artefatos. Números SEMPRE copiados dos `RESULTS.md`/parquets reais de cada
> fase — nunca de memória.
>
> **Última atualização**: 2026-11-04 (F5: alavancas de retrieval + compressão de contexto).
> Branches-fonte (exploração, NÃO mesclados no main por decisão do capitão):
> `main` (F0, F2) · `fm/lfm3-f3-robustez` (F3) · `fm/lfm3-f3b-sweep` (F3b) ·
> `fm/lfm3-f4-tools` (F4) · `fm/lfm3-f5-retrieval` (F5).

---

## 1. Resumo executivo

**O que foi explorado.** Uma linha completa de adaptação do
`LiquidAI/LFM3-1.7B` para o assistente de NRs da POC, medida sempre pela
mesma régua honesta (benchmark `UFGCEMIGONA/benchmark_mobile`, 1536 queries =
384 perguntas × 4 vocabulários, 4 células closed-book/oráculo/ranking/
distrator, juiz 27B, cobertura de fatos com citação fora do gate):

- **F0** — baseline as-is do modelo (a régua do "antes").
- **F2** — injeção de conhecimento nos pesos (LoRA no instruct, task vector
  via base, K2V/GRPO).
- **F3** — robustez de uso de contexto com distratores do retriever REAL.
- **F3b** — sweep one-factor-at-a-time sobre a receita F3 (o que domina?).
- **F4** — tool-calling multiturno (o modelo decide buscar) sobre o campeão
  F3b.
- **F5** — alavancas de retrieval no v4 (gate, vizinhos, manuais) medidas
  numa régua DUPLA (151 da POC + benchmark) e **compressão de contexto**
  por relevância (hipótese do capitão) medida fim-a-fim com o `f4-r32-ep2`.

**Candidato final: `f4-r32-ep2`** (LoRA r32 MLP sobre o merged
`data2x-ep4`). Números-chave na bateria completa 1536 vs F0 baseline:

| métrica | F0 | f4-r32-ep2 | Δ |
|---|--:|--:|--:|
| ranking-1-de-5 (aprovação) | 65,8% | **79,6%** | +13,8 p.p. |
| oráculo (aprovação) | 83,7% | **87,8%** | +4,1 p.p. |
| distrator: recusa correta | 30,5% | **66,6%** | +36,1 p.p. (2,2×) |
| distrator: alucinação | 28,3% | 24,2% | −4,1 p.p. |
| recusa indevida (ranking) | 6,2% | 6,6% | ≈ estável |
| truncamento (distrator) | 35,4% | 0% (desde F3) | ruminação eliminada |
| probe de capacidade geral (1–5) | 3,92 | 3,38 | −0,54 (custo real) |
| decisão de tool (F1) | 0 (nunca chama) | **97,2** | habilitada |
| sintaxe XML de tool | — | **100%** (0 falhas/285 chamadas) | perfeita |

**As lições centrais da exploração:**

1. **Dado > hiperparâmetro.** No sweep F3b, os três eixos que dominam são
   todos de DADO (tamanho 2× → ranking; dose de recusa → alucinação; âncora
   geral → probe) e são ortogonais; rank, alvos LoRA e LR são neutros.
2. **Conhecimento nos pesos tem teto baixo.** A injeção da F2 funciona mas é
   modesta (closed-book tratadas 8,3→17,2% no melhor caso) e época extra vira
   alucinação, não conhecimento; task vector restaura comportamento mas não
   preserva o injetado; RLVR (K2V/GRPO) não move a agulha neste tamanho.
3. **Robustez + tools pagam.** F3/F3b compram +11 p.p. de ranking e 2× de
   recusa correta; o treino de tool da F4 CRUZA todas as metas de
   comportamento (decisão/sintaxe/reuso/troca ≥90%) **com regressão estática
   ZERO** — na verdade melhora o modo sem ferramenta (+2,6 p.p. ranking,
   +2,5 p.p. oráculo, probe +0,2). O ponto fraco que resta é o multihop
   (7,1% fim-a-fim, teto de argumento do modelo pequeno).
4. **Dominâncias não somam.** Combinar eixos vencedores sem dado novo tem
   interação NEGATIVA: `combo` derruba ranking; `combo3` recupera ranking e
   alucinação mas colapsa a probe (2,88) — empilhar exige escalar a âncora em
   valor absoluto, não em fração.
5. **Menos contexto, mais denso, não piora — se o orçamento for reinvestido
   em recall (F5).** Buscar top-5 e comprimir a 50% dos tokens do controle
   (seleção de unidades normativas pelo próprio EmbeddingGemma, id do item
   preservado) dá aprovação ≥ top-2 cru (57,5 vs 55,6%, 2 rodadas, recorte
   344, dentro do ruído), cita o item certo +8 p.p. e corta 33% do prompt;
   comprimir só o top-2 PIORA. No retrieval, o único ajuste de gate que passa
   nas duas réguas é limiar 0,7 + boost 2× (ganho modesto); tirar o
   classificador é mascaramento (benchmark +3, POC −11 p.p. R@2).

---

## 2. Tabela-mestra de evolução (bateria completa 1536, juiz 27B)

Todas as colunas na bateria oficial de 1536 queries; a F2 foi avaliada apenas
no **recorte 344** (6 normas) e por isso entra como nota, não como coluna.

| métrica | F0 baseline | f3-r32-ep4 (F3) | f3b-data2x-ep4 (campeão F3b) | **f4-r32-ep2 (candidato)** |
|---|--:|--:|--:|--:|
| ranking aprovação | 65,8% | 75,2% | 77,0% | **79,6%** |
| oráculo aprovação | 83,7% | 83,5% | 85,4% | **87,8%** |
| closed-book aprovação | 5,9% | 6,2% | 7,4% | 6,7% |
| distrator: alucinação | 28,3% | 22,4% | 24,9% | 24,2% |
| distrator: recusa correta | 30,5% | 59,1% | 55,6% | **66,6%** |
| recusa indevida (ranking) | 6,2% | 5,9% | 8,3% | 6,6% |
| recusa indevida (oráculo) | 5,9% | 3,1% | — (não reportada) | — (não reportada) |
| truncamento (distrator) | 35,4% | 0,0% | 0,0% | — (não reportado; 0% desde F3) |
| probe geral (1–5) | 3,92 | 3,18 | 3,17 | 3,38 |
| tokens médios/resposta | ~700–1500 (thinking) | ~110 | — | — |

**Nota F2 (recorte 344, closed-book)**: F0 tratadas 8,3% / controle 5,5% →
melhor braço A-r32-ep2 **17,2% tratadas** (+8,9 p.p., 2,1×) com controle
estável 7,9%. Linha abandonada (teto baixo; ver §3.2).

**Divergência registrada (recorte vs 1536)**: no F3b, o ganho do `data2x`
medido no recorte 344 (+2,6 p.p. de ranking) **encolhe para +1,8 p.p. na
bateria 1536** — o recorte, por conter só as 6 NRs do F2, favorece o funil
novo. Pelo mesmo motivo, as referências F0 diferem entre fontes
(ranking 65,8% na 1536 vs 67,7% no recorte; distrator recusa-correta 30,5%
vs 22,4%). Sempre comparar números do MESMO recorte.

---

## 3. Seções por fase

### 3.1 F0 — Baseline as-is (2026-10-28) · `main:bench/lfm3_f0`

- **Objetivo**: régua do "antes": o LFM3-1.7B bf16 sem qualquer adaptação,
  nas 4 células × 1536, thinking-ON (default do template), max_new 3072.
- **Desenho**: régua honesta portada de `bench/regua` (gabarito de fatos por
  `base_qa_id` extraído de answer+evidence pelo juiz 27B; aprovado =
  cobertura ≥0,5 ∧ ¬tautologia ∧ ¬alucinação ∧ ¬recusa ∧ ¬truncada;
  truncamento NUNCA conta como recusa). Embaralhamento determinístico por id.
- **Resultado (all, 1536)**: closed-book **5,9%**, oráculo **83,7%**,
  ranking **65,8%**, distrator recusa-correta **30,5%** (alucinação 28,3%,
  truncou ruminando 35,4%). Gradiente de vocabulário consistente:
  clean/technical > operational/lay em todas as células (lay: closed-book
  3,6%, ranking 57,6%, alucinação distrator 42,2%).
- **Veredito honesto**: o modelo quase não carrega as NRs no peso
  (closed-book ≈ piso), sintetiza bem com a passagem certa (oráculo alto),
  perde 18 p.p. com 4 distratores na janela e tem o distrator-puro como
  calcanhar (prefere responder a recusar). O abismo lexical da POC reaparece
  na síntese.
- **Custo**: geração ~14 h na GB10 (6144 gerações, bf16, batch 24);
  julgamento local ~40 min (16 workers).

### 3.2 F2 — Injeção de conhecimento (2026-10-30) · `main:bench/lfm3_f2`

- **Objetivo**: "simular pré-treino" em 3 normas (tratadas NR-10/18/34;
  controle NR-13/31/33 nunca vistas) e subir o closed-book sem quebrar o
  resto. Avaliação no recorte 344 (6 normas × 4 vocab).
- **Desenho — 3 braços** sobre o mesmo dataset sintético (funil front-loaded
  no 27B: 842 unidades → 6.536 QAs → crítico 27B → 5.526 aprovados; entropia
  11,0 bits, 0 contaminação):
  - **A** — LoRA no INSTRUCT (w1/w2/w3, 72 módulos MLP), r8 × r32,
    12 épocas, LR 5e-5;
  - **B** — task vector (hipótese do capitão): LoRA no BASE, merge,
    re-aplica delta instruct−base (variantes `sum`/`prop`);
  - **C** — K2V via TRL-GRPO (1.222 QAs fill-blank verificáveis + checklist
    27B; vllm/sglang não suportam LFM3 — rollout só via transformers).
- **Resultado (closed-book, recorte; F0 = trat 8,3% / ctrl 5,5%)**:
  - melhor célula **A-r32-ep2: 17,2% tratadas** (ctrl 7,9%, haluc 36,1%);
    A-r8-ep6 15,0%. r32 colapsa cedo (haluc 36→72% até ep12); r8 satura ~ep6.
  - **B task vector**: restaura comportamento (haluc 83–90% do LoRA-no-base
    puro → 22–36%) mas NÃO preserva o conhecimento (trat ≤7,8% < F0);
    `prop` pior que `sum`.
  - **C K2V/GRPO**: 7,2% tratadas = +0 p.p. sobre o F0, com muito mais custo.
  - Guardas: oráculo/ranking caem 4–7 p.p. no braço A; probe geral
    3,92 → 3,48 (r8-ep6) / 3,78 (r32-ep2).
- **Veredito honesto**: a injeção FUNCIONA mas é modesta e específica
  (controle nunca acompanha); **o que cresce com época é ALUCINAÇÃO, não
  acerto**; nenhum braço passa limpo pelas guardas. Linha abandonada em favor
  da robustez de contexto (F3).
- **Custo**: treinos A/B na Spark (GB10) + braço C e task vector na H100;
  horas de GPU não consolidadas nos RESULTS (curvas nos runs wandb
  `cemig-lfm3-f2`, offline). Corrida C: 400 steps GRPO (pico de reward
  ~step 200).

### 3.3 F3 — Robustez de uso de contexto (2026-10-31) · `fm/lfm3-f3-robustez:bench/lfm3_f3`

- **Objetivo**: fechar o gap recuperável F0 (oráculo 83,7 vs ranking 65,8) com
  distratores do retriever REAL v4 da POC. Alvos: ranking ≥78%, alucinação
  distrator <10%, guardas estáveis.
- **Desenho**: dataset 6.972 exemplos ancorado no retriever real (casos gold
  ~60% / absent ~20% / partial ~10% / closed ~10%; k variável 3/5/8; alvos
  gerados pelo 27B vendo APENAS a âncora), LoRA r32 MLP w1/w2/w3 sobre o
  instruct LIMPO, `assistant_only_loss`, 8 épocas, grade completa
  8 × 4 células × 1536.
- **Resultado (melhor equilíbrio ep4; 1536)**: ranking **75,2%** (+9,4 p.p.),
  alucinação distrator **22,4%**, recusa correta **59,1%** (2× o F0),
  truncamento 35,4→**0%**, oráculo 83,5% e closed-book 6,2% (guardas
  intactas), recusa-indevida MENOR que o F0. Probe: **3,92 → 3,18** (ep4) /
  3,31 (ep6) — há forgetting real.
- **Veredito honesto**: ganho grande e barato em comportamento, mas os 2
  alvos numéricos NÃO cruzados (platô 74–76% / 21–26%; eval_loss mínima em
  ep3). O resíduo concentra-se em operational/lay (mesmo abismo lexical). O
  trade-off de recusa-indevida da sft_v3 NÃO ocorreu. Alavanca seguinte é
  DADO, não época/rank → F3b.
- **Custo**: 1 treino de 8 épocas na H100 + geração das células ~20–40
  min/célula (Spark/H100, batch 48; vs ~14 h do F0 na GB10). wandb
  `lfm3-1.7b-cemig/lfm3-f3-robustez-r32` (kde9xyam ep1-4, rcavhk37 ep5-8).

### 3.4 F3b — Sweep de dominância (2026-11-02) · `fm/lfm3-f3b-sweep:bench/lfm3_f3b`

- **Objetivo**: um fator por vez sobre a receita F3 congelada — qual eixo
  domina? Avaliação rápida no recorte 344 + probe; campeão na bateria 1536.
- **Desenho — 6 eixos**: dose de recusa 20→35% (`recusa35`), tamanho do
  dataset 2× (`data2x`, 12 NRs novas), rank r16/r64, alvos MLP+atenção, LR
  5e-5, âncora geral 10→25% (`anchor25`); + combinações `combo` e `combo3`.
- **Tabela de dominância (recorte 344; melhor época por eixo)**:

  | eixo | célula | ranking | haluc distrator | recusa-correta | rec-indevida | probe-g | veredito |
  |---|---|--:|--:|--:|--:|--:|---|
  | — | F0-baseline | 67,7% | 27,0% | 22,4% | 3,8% | 3,92 | ref |
  | — | F3 (f3-r32-ep4) | 77,0% | 23,0% | 53,8% | 3,8% | 3,18 | ref |
  | tamanho 2× | data2x-ep4 | **79,7%** (ep3 80,2) | 22,4% | 51,2% | 3,8% | 3,17 | **DOMINA ranking** |
  | recusa 35% | recusa35-ep3 | 76,2% | **13,4%** (ep4 11,9) | 73,3% | 7,0% | 3,37 | **DOMINA alucinação** |
  | âncora 25% | anchor25-ep4 | 76,2% | 21,8% | 51,7% | 3,2% | **3,65** | **DOMINA probe** |
  | rank r16 | r16-ep4 | 77,0% | 21,2% | 55,8% | 3,2% | 3,35 | neutro |
  | rank r64 | r64-ep4 | 75,0% | 18,3% | 58,4% | 4,4% | 3,39 | neutro/leve |
  | MLP+atenção | mlpattn-ep4 | 74,4% | 18,6% | 56,7% | 4,4% | 3,30 | neutro |
  | LR metade | lr5e5-ep4 | 75,9% | 24,1% | 54,1% | 4,1% | 3,25 | neutro/pior |

- **Pareto honesto**: `data2x-ep4` (campeão formal; única célula que melhora
  ranking sem violar guardas), `recusa35-ep3` (se a prioridade virar
  alucinação), `combo3-ep3` (ranking 78,8 + haluc 15,4, mas probe 2,88 —
  só para assistente-NR puro).
- **Campeão na 1536**: ranking **77,0%** (+1,8 p.p.; alvo 78 a 1 p.p.),
  oráculo **85,4%**↑, closed 7,4%, probe 3,17, recusa-indevida 8,3%↑ (custo
  real leve). Alucinação <10% NÃO atingível por época/rank/LR.
- **Veredito honesto**: **o que domina é DADO, não hiperparâmetro**; as 3
  dominâncias são ortogonais e NÃO somam de graça (combos com interação
  negativa; empilhar exige âncora em valor absoluto ≥3× — hipótese F3c, não
  testada).
- **Custo**: 9 treinos de 4 épocas (fila na H100, `queue_runner.sh`) +
  avaliação rápida 344/célula; horas de GPU não consolidadas nos RESULTS
  (runs wandb `lfm3-1.7b-cemig/lfm3-f3b-*`).

### 3.5 F4 — Tool-calling multiturno (2026-11-03) · `fm/lfm3-f4-tools:bench/lfm3_f4`

- **Objetivo**: o próprio modelo decide chamar a busca, sintetiza do retorno,
  recusa com contexto errado e re-busca (multihop) — sem regredir o modo
  estático do campeão F3b. Metas: decisão/sintaxe ≥90%, multihop ≥60%,
  regressão estática ≤2 p.p.
- **Desenho**: base = merged `data2x-ep4` (md5 `ae5f304d…`); formato de tool
  NATIVO do LFM3 (XML `<tool_call>` do próprio `apply_chat_template`; parser
  inverso determinístico com round-trip provado; `enable_thinking=False`
  sempre); dataset 4.624 diálogos em 6 famílias (chamada_simples, recusa,
  troca, reuso, multihop, sem_ferramenta; argumento da 1ª chamada = FALA
  CRUA); avaliação com o retriever v4 REAL respondendo ao vivo; variante
  `mhx3` (dose multihop 9→24% ×3 réplicas) para a curva dado×métrica.
- **Resultado — estáticas (1536)**: ZERO regressão; o treino de tool
  MELHOROU o modo sem ferramenta: ranking 77,0→**80,5/79,6/78,3** (ep1/2/3),
  oráculo 85,4→**87,8–88,4**, recusa-indevida 8,3→6,2–7,4, probe
  3,17→**3,38–3,45**.
- **Resultado — métricas de tool (suites disjuntas; base sem treino nunca
  chama a tool, F1 = 0)**:

  | métrica | f4-ep2 | f4-ep3 | mhx3-ep1 | mhx3-ep2 | alvo |
  |---|--:|--:|--:|--:|---|
  | decisão F1 | 97,2 | **97,6** | 98,4 | 98,0 | ≥90 ✓ |
  | sintaxe XML válida | **100** (0/285) | 100 (0/285) | 100 (0/232) | 100 (0/311) | ≥90 ✓ |
  | reuso correto (não chamar) | **100** | 100 | — | 100 | ✓ |
  | troca: nova chamada / acha NR nova | 100 / 96,4 | 100 / 96,4 | — | 98,2 / 96,4 | ✓ |
  | recusa pós-busca correta | 50,0 | 51,7 | 45,0 | 38,3 | trade-off ↓ |
  | multihop: 2ª chamada | 2,4 | 2,4 | 7,1 | **33,3** | |
  | multihop: cadeia completa | 0 | 0 | 0 | **7,1** | ≥60 ✗ |

- **Curva multihop (dado×métrica, ordem do brief)**: dose 411 (9%) →
  1233 (24%, ×3 réplicas): 2ª chamada 2,4 → 7,1 (ep1) → **33,3%** (ep2); ouro
  no hop2 2,4 → 11,9%; cadeia completa 0 → **7,1%**; recusa pós-busca paga o
  ganho (50 → 38,3). Gargalos em série: (a) a 2ª consulta escrita pelo 1.7B
  recupera o ouro bem menos que a consulta-2 validada do 27B (11,9% vs 100%
  no dataset) — mesmo teto de argumento do tools_v1; (b) quem re-busca passa
  a responder com o que veio, certo ou não.
- **Veredito honesto**: decisão/sintaxe/reuso/troca CRUZARAM as metas já na
  1ª época (saturação instantânea, consistente com tools_v1); **multihop
  ≥60% NÃO atingido** (7,1% no melhor ponto) e dobrar o dataset multihop NÃO
  é a alavanca — o gargalo é a qualidade da 2ª consulta, não a vontade de
  chamar. **Candidato: `f4-r32-ep2`** (equilíbrio F1 97,2 / recusa 50 /
  estáticas no topo / probe 3,38); mhx3-ep2 é referência da curva, mas paga
  12 p.p. de recusa por 7% de cadeia.
- **Custo**: 2 treinos (3 épocas r32 + 2 épocas mhx3) na H100 + bateria
  estática 3 épocas × 4 células × 1536 + 5 suites de tool ao vivo; horas de
  GPU não consolidadas nos RESULTS (wandb `lfm3-f4-tools-r32` r22hqe1h e
  `lfm3-f4-tools-mhx3-r32`). Pipeline autônomo no container para
  intermitência de rede (juiz via gateway docker `172.17.0.1:8005`).

### 3.6 F5 — Alavancas de retrieval + compressão de contexto (2026-11-04) · `fm/lfm3-f5-retrieval:bench/lfm3_f5`

- **Objetivo**: as 3 alavancas do smoke de retrieval (gate do classificador,
  vizinhos ±1, manuais como aba separada) + a hipótese do capitão: comprimir
  o contexto entregue ao LFM com um modelo pequeno de relevância (estilo SCR
  do MobileRAG+) melhora ou mantém a resposta com menos tokens. Trava
  anti-mascaramento: só conta o que sobe numa régua sem derrubar a outra
  (151 da POC × benchmark_mobile 1536).
- **Harness**: reimplementado do apêndice do smoke e agora COMMITADO
  (`align.py`/`metrics.py`); paridade 151 = 51,7/65,6. Override de NR
  explícita portado para `bench/ctx_topk/retrieval_v4.py` (espelho = app);
  2ª divergência espelho×app achada (top-1 `nenhuma`).
- **Retrieval (controle `v4app`: 151 R@2 52,3 / R@5 66,2; benchmark norma
  hit@2 0,762 / hit@5 0,879)**:

  | alavanca | 151 R@2 / R@5 | norma hit@2 / hit@5 | manual hit@5 | trava |
  |---|--:|--:|--:|:-:|
  | gate limiar 0,7 + boost 2× | 52,3 / 66,9 | 0,775 / 0,886 | 0 | ✓ |
  | sem classificador (só NR explícita) | 41,1 / 60,3 | 0,791 / 0,898 | 0 | ✗ (mascarada) |
  | top-2 + vizinhos ±1 (conjunto) | 56,3 | 0,807 | — | ✓ (só c/ compressão) |
  | manuais no RRF w=1,25 | 52,3 / 66,2 | 0,762 / 0,877 | 0,220 | ≈ (−0,2 norma) |
  | manuais no RRF w=2 | 50,3 / 62,9 | 0,730 / 0,847 | 0,782 | ✗ |

- **Compressão fim-a-fim** (`f4-r32-ep2`, recorte 344, formato do app com a
  busca injetada no turno tool; régua honesta, juiz 27B; aprovação = média de
  2 rodadas onde houve réplica; TTFT estimado = prompt ÷ 116 t/s, não medido
  em aparelho):

  | célula | aprov. | alucin. | cita item-ouro | tok prompt | TTFT~ |
  |---|--:|--:|--:|--:|--:|
  | (a) top-2 cru (app) | 55,6 | 13,8 | 65,2 | 1635 | 14,1 s |
  | (b) top-2 → 50% | 52,3 | 12,2 | 67,0 | 1096 | 9,5 s |
  | (c) top-2 + ±1 → orçamento de (a) | 57,4 | 12,9 | 71,7 | 1547 | 13,4 s |
  | **(d) top-5 → 50%** | **57,5** | **12,5** | **73,2** | **1100** | **9,5 s** |

- **Veredito**: hipótese confirmada na forma (d): mesma ou melhor aprovação
  (ganho dentro do ruído, sinal pareado p≈0,46), citação de item +8 p.p. e
  −33% de prompt. Comprimir só o top-2 piora (−2 a −3 p.p., recusa sobe).
  A lição do `trim.py` foi resolvida: o `(item X)` ancorado em cada unidade faz
  todas as células comprimidas citarem o item MELHOR que o cru. **Pendente
  antes de embarcar**: medir (d) nas 151 da POC (o proxy sem LLM das falas
  coloquiais perde fatos: 0,70→0,49 no contexto).
- **Custo**: retrieval ~1 min/variante (5070 local + CPU); geração E2E 2408 +
  1032 (réplica) respostas na Spark (batch 8; batch 48 causou OOM da memória
  unificada); julgamento local ~7 min por rodada.

---

## 4. Decisões de produto em aberto

1. **Multihop híbrido via app** (caminho (i) do RESULTS da F4): a 2ª consulta
   vem do app/27B em vez do 1.7B — réplica do desenho `tools_oraculo` da POC,
   que já provou que fala-crua+retriever supera a reescrita do modelo
   pequeno. Alternativas listadas e não executadas: consultas-2 múltiplas por
   item de treino (variância de argumento) e relaxar o gate do hop2 para
   top-8.
2. **Escolha final de checkpoint**: candidato técnico é `f4-r32-ep2`;
   `f4-r32-ep3` tem F1 0,4 maior e probe 3,45, `mhx3-ep2` maximiza multihop
   mas sacrifica recusa. Decisão do capitão pendente.
3. **Condição de embarque (esquecimento)**: a probe geral segue abaixo do F0
   (3,38–3,45 vs 3,92). Aceitável para assistente-NR puro; para uso geral,
   exigiria mitigação (a âncora `anchor25` da F3b provou mover a probe
   +0,47, mas não foi combinada ao candidato — a interação negativa dos
   combos exige âncora em valor absoluto).
4. **Quantização pendente**: toda a exploração foi em bf16 em GPU (ordem do
   capitão na F0). Q4/GGUF, latência on-device e RAM do 1.7B não foram
   medidos — bloqueia qualquer decisão de embarque.
5. **Retrieval/compressão no app (F5)**: (i) gate limiar 0,7 + boost 2× no
   `HybridRetriever` (2 constantes); (ii) compressão top-5→50% após validar
   nas 151 (custa ~20 MB int8 de índice de janelas); (iii) peso dos manuais
   no RRF é escolha de produto (w=1,25 sem custo para as normas; gate por
   margem densa a w=2 como equilíbrio).

---

## 5. Apêndice

### 5.1 Mapa de artefatos (branch → diretório → artefatos)

| fase | branch | diretório | parquets canônicos | checkpoints | wandb (`lfm3-1.7b-cemig`, salvo indicação) |
|---|---|---|---|---|---|
| F0 | `main` | `bench/lfm3_f0` | via retrofit no schema F2 (baseline 1536 × 4 células em `bench/lfm3_f2/data/eval/`) | — (modelo as-is) | — |
| F2 | `main` | `bench/lfm3_f2` | `data/eval/` — 46 parquets (checkpoint×célula, inclui baseline retroativo) | Spark `~/cemig-poc/lfm3_f2/runs/` (A/B por época, task vectors, C-k2v) | projeto `cemig-lfm3-f2` (offline); braço C `ixu0jg3k` |
| F3 | `fm/lfm3-f3-robustez` | `bench/lfm3_f3` | `data/eval/` — 32 parquets (8 épocas × 4 células) | H100 `checkpoints/lfm3-f3-robustez-r32/checkpoint-{218…1744}`; Spark ep1-6 | `lfm3-f3-robustez-r32` (kde9xyam), `-ext8` (rcavhk37) |
| F3b | `fm/lfm3-f3b-sweep` | `bench/lfm3_f3b` | `data/eval/` — 4 parquets do campeão; dominância completa em `data/dominance.json` | H100 `checkpoints/lfm3-f3b-data-2x/checkpoint-1396` (campeão) + demais células `lfm3-f3b-*` | runs `lfm3-f3b-*` |
| F4 | `fm/lfm3-f4-tools` | `bench/lfm3_f4` | `data/eval/` — **35 parquets** (estáticas 3 épocas+mhx3 + 5 suites de tool × 5 tags) + suites/traces jsonl | H100 `checkpoints/lfm3-f4-tools-r32/checkpoint-{142,284,426}` e `lfm3-f4-tools-mhx3-r32/checkpoint-{157,314}`; merged base `/workspace/cemig/f4/merged-rebuilt` (md5 `ae5f304d91bb47402d9125d96e766a15`) | `lfm3-f4-tools-r32` (r22hqe1h), `lfm3-f4-tools-mhx3-r32` |
| F5 | `fm/lfm3-f5-retrieval` | `bench/lfm3_f5` | — (retrieval: `data/retrieval/*.json`; E2E: `data/e2e/{gen,scored,summary}_f4-r32-ep2*`) | usa `f4-r32-ep2` (checkpoint-284, md5 adapter `18a40d6c…`) | — |

RESULTS por fase: `bench/lfm3_f{0,2,3,3b,4,5}/RESULTS.md`; consolidações
numéricas: `lfm3_f3/data/consolidation.json`, `lfm3_f3b/data/dominance.json`,
`lfm3_f4/data/consolidation.json`, `lfm3_f5/data/consolidation.json`.

### 5.2 Infra usada

- **Spark GB10** (`walcyrios@spark-b431` / tailscale `100.79.169.101`,
  `~/cemig-poc/`): geração batched (bs48 ≈ 26× bs1) e treinos leves; memória
  unificada — NUNCA 2 treinos simultâneos (OOM derruba o host; r32 exige
  bs4/ga8).
- **H100 GPU 2** (`aluno_walcy@10.100.0.111`, container `cemig_mobile_walcy`;
  artefatos em `/raid/aluno_walcy/cemig_mobile` = `/workspace/cemig` no
  container): treinos F3/F3b/F4 (fila `queue_runner.sh`), braço C e task
  vector da F2, pipeline autônomo de avaliação da F4 (juiz via gateway
  docker `172.17.0.1:8005`). H100→Spark NÃO roteia: adapters via host local.
- **Juiz**: vLLM `Qwen/Qwen3.8-27B-FP8` em `http://10.100.0.111:8005/v1`;
  julgamento local (WSL, `classifier/.venv`).
- **Restrições de stack**: vllm/sglang não suportam o LFM3 (camada conv);
  tokenizer novo exige transformers 5.x; LoRA sempre MLP `w1/w2/w3`
  (72 módulos; `gate_proj` é gate da ATENÇÃO — nunca incluir).

### 5.3 Schema dos parquets canônicos

Um parquet por **checkpoint × célula**
(`lfm3-<checkpoint>-<célula>.parquet`); definição autoritativa em
`bench/lfm3_f2/data/eval/README.md`:

| grupo | colunas |
|---|---|
| benchmark | id, base_qa_id, question, answer_gold, question_type, source_id, vocabulary_level, query_features, source_file, evidence_text |
| saída do modelo | answer_model, raw_output (integral: thinking+resposta), thinking, n_gen_tokens, stopped_eos |
| veredito (régua honesta) | coverage, n_facts, tautology, hallucination, refused, truncated, refusal_correct, refusal_undue, citation_present, approved, answered_correct_anyway, judge_note |
| protocolo | context_given, context_type (none/oracle/ranking_1de5/distractor_only; F4 adiciona tool_loop/multihop), system_prompt_sha + system_prompt_ref, checkpoint, cell, gen_params |

`context_given` é reconstruído deterministicamente (seed
`sha256(f"42:{id}")`, idêntico ao da geração —
`bench/lfm3_f0/common_f0.py:build_cell_context`).

### 5.4 Divergências conhecidas entre fontes

- **Recorte 344 vs bateria 1536**: números do recorte favorecem as NRs do
  funil (F3b: +2,6 p.p. no recorte → +1,8 p.p. na 1536). As referências F0
  mudam junto (ranking 67,7 vs 65,8; recusa-correta distrator 22,4 vs 30,5).
- **Probe da F3 vs F3b**: a linha F3 da tabela de dominância usa a probe
  medida NA FASE F3 (3,18/3,31), marcada com asterisco no RESULTS da F3b.
- **"Reuso 100%" do base sem treino na F4 é vacuoso** (nunca chama nada) e a
  "recusa 60%" do base vem do estilo herdado do F3b — comparações de tool só
  valem entre checkpoints treinados.
