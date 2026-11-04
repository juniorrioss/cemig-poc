# RESULTS — F6: manuais nos índices (ponto de operação + fim-a-fim)

Ordem do capitão: *"junte os manuais nos índices então, nem que seja um experimento
diferente"*. Motivo: no fim-a-fim da F5 a busca em NORMAS acerta 85–92%, mas 35% do
recorte (26% das 1536) tem o gabarito em MANUAL não indexado. Acerto 0% por construção.

Base reaproveitada da F5, sem reconstruir: `cache/manuals.db` (9 manuais, 1092 chunks,
offsets nativos), denso próprio (EmbeddingGemma), 2 sinais extras no RRF
(`retriever_f5.Cfg.manual_w`), compressor SCR-Gemma (`compress.py`), modelo
`f4-r32-ep2`, régua honesta com juiz 27B e formato do app (busca injetada no turno tool).
Tabela-mestra: `data/consolidation.json`.

## Resumo

| | antes (F5 d50, sem manuais) | **F6 ponto escolhido** | Δ |
|---|--:|--:|--:|
| recorte 344: aprovação global | 57,5 | **68,0** | **+10,5 p.p.** |
| recorte: aprovação, ouro em MANUAL (120) | 18,3 | **53,3** | **+35 p.p. (2,9×)** |
| recorte: aprovação, ouro em NORMA (224) | 78,4 | 75,9 | −2,5 observado / **−0,3 causal** |
| recorte: recusa | 27,5 | 15,4 | −12,1 |
| recorte: alucinação | 11,5 | 13,2 | +1,7 |
| recorte: tokens de prompt | 1100 | 1100 | = |
| 151: aprovação (régua honesta) | 13,2 (d50) / 16,9 (top-2 cru) | 15,2 | ruído (p≥0,3) |

**Ponto de operação: `thr0.7_f2+man1.75+margin`**: gate limiar 0,7 + boost 2× (F5) +
manuais com peso 1,75 no RRF, ligados só quando o melhor trecho denso de manual for pelo
menos tão parecido com a fala quanto o melhor de norma (gate por margem densa, m=0).

## 1. Escolha do ponto de operação (retrieval, 2 réguas + 400 de manual)

Critério do brief: maximizar **manual hit@5** com **norma hit@5 ≥ −0,5 p.p.** e
**151 R@2 intacto**. Δ medidos contra o controle SEM manuais com o MESMO gate.
Grade de 49 variantes em `data/retrieval/`. A tabela abaixo mostra a fronteira:

| variante | 151 R@2 | 151 R@5 | norma hit@2 | norma hit@5 (Δ) | **manual hit@5** | todas hit@5 | trava |
|---|--:|--:|--:|--:|--:|--:|:-:|
| `v4app` (app, sem manuais) | 52,3 | 66,2 | 0,762 | 0,879 | 0,000 | 0,650 | — |
| `thr0.7_f2` (gate F5, sem manuais) = **controle** | 52,3 | 66,9 | 0,775 | 0,886 | 0,000 | 0,656 | — |
| v4app + w=1,25 (candidato do brief) | 52,3 | 66,2 | 0,762 | 0,877 (−0,26) | 0,220 | 0,706 | ✓ |
| v4app + w=1,5 (candidato do brief) | 52,3 | 66,2 | 0,761 | 0,868 (−1,14) | 0,472 | 0,765 | ✗ |
| v4app + margem w=2 (candidato do brief) | 52,3 | 65,6 | 0,754 | 0,868 (−1,14) | 0,540 | 0,783 | ✗ |
| v4app + margem w=1,45 (melhor ✓ no gate v4app) | 52,3 | 66,2 | 0,761 | 0,875 (−0,44) | 0,325 | 0,732 | ✓ |
| thr0.7_f2 + w=1,4 | 52,3 | 66,9 | 0,775 | 0,883 (−0,35) | 0,365 | 0,748 | ✓ |
| **thr0.7_f2 + margem w=1,75** ⇐ | **52,3** | **66,9** | **0,770** | **0,882 (−0,44)** | **0,468** | **0,774** | **✓** |
| thr0.7_f2 + margem w=1,8 / 1,9 | 52,3 | 66,9 | 0,770 | 0,881 / 0,880 (−0,52 / −0,61) | 0,482 / 0,515 | 0,777 / 0,785 | ✗ (limite) |
| thr0.7_f2 + w=1,6 (pareto "mid") | 52,3 | 67,5 | 0,772 | 0,877 (−0,96) | 0,537 | 0,788 | ✗ |
| thr0.7_f2 + margem w=2 | 52,3 | 66,2 | 0,768 | 0,877 (−0,96) | 0,537 | 0,788 | ✗ |
| thr0.7_f2 + w=1,8 | 51,0 | 66,9 | 0,767 | 0,872 (−1,49) | 0,685 | 0,823 | ✗ (151) |
| thr0.7_f2 + w=2 (pareto "hi") | 50,3 | 63,6 | 0,749 | 0,861 (−2,55) | 0,780 | 0,840 | ✗ |

Por que esse ponto:
- **A 1,25 e a 1,5 o dial é íngreme.** Manual hit@5 sobe ~1 p.p. a cada 0,01 de peso, e
  a norma passa de −0,3 para −1,1 entre 1,4 e 1,5. Os três candidatos do brief ficam em
  dois extremos: w=1,25 deixa manual a 22%, e w=1,5 e margem w=2 quebram a trava (−1,1).
- **Gate da F5 + margem densa dobra o manual dentro da trava.** O gate limiar 0,7 +
  boost 2× (frente 1 da F5, que já embarca) sobe a norma +0,7 p.p. Essa folga absorve
  parte da diluição. A margem densa liga os manuais só nas falas "de manual", o que
  protege a norma (−0,44 contra −0,96 sem gate no mesmo nível de manual). Resultado:
  **manual 22% → 46,8%** (2,1× o w=1,25) com 151 R@2 e R@5 iguais ao controle e
  norma −0,44 p.p.
- Contra o app de hoje (`v4app`) o ponto escolhido fica **acima em tudo**: norma
  hit@5 +0,26, hit@2 +0,8, 151 R@5 +0,7, manual 0 → 46,8%, todas hit@5 0,650 → 0,774.
- **Robustez local**: a vizinhança (w 1,7–1,9, margem −0,005–0) fica entre −0,44 e
  −0,61 de norma, com manual de 46 a 52%. O platô é estável e o ponto está na borda da
  tolerância, não num pico isolado.

## 2. Fim-a-fim: recorte 344 (f4-r32-ep2, juiz 27B, formato do app)

Pipeline: retriever → top-5 → compressão a 50% → `f4-r32-ep2`. O compressor ganhou as
**janelas dos 1092 chunks de manual**: `compress_f6.py`, 10.125 unidades com a mesma
segmentação, janela-3 e prompt de documento, spans realinhados ao texto nativo do manual
(só 4 chunks com cobertura de alinhamento abaixo de 0,9). O orçamento é **50% dos tokens do top-2 cru do
`v4app` da mesma query**, igual à d50 da F5. Chunk de manual (~1700 chars) não infla o
prompt. **Paridade**: a d50 `v4app` remontada com o compressor estendido sai **idêntica
byte a byte à da F5 em 344/344** (`data/e2e/parity_d50_f5.json`).

3 rodadas para ctrl/op e 2 para mid/hi (sampling 0,1/50/1,1, batch 8). Os números são
médias das rodadas:

| célula | retrieval | aprov. | **norma** | **manual** | cobertura | alucin. | recusa | cita NR | cita item-ouro | tok ctx / prompt | TTFT~ | ouro no ctx |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F5 a (app, top-2 cru) | v4app | 55,6 | 75,7 | 18,3 | 0,62 | 13,8 | 24,7 | 99 | 65,2 | 983 / 1635 | 14,1 s | 56 |
| F5 d50 | v4app | 57,5 | 77,9 | 19,2 | 0,63 | 12,5 | 25,9 | 99 | 73,2 | 448 / 1100 | 9,5 s | 60 |
| **ctrl_d50** (controle F6) | thr0.7_f2 | 57,5 | 78,4 | 18,3 | 0,626 | 11,5 | 27,5 | 99,5 | 62,9 | 448 / 1100 | 9,5 s | 60,2 |
| **op_d50** ⇐ escolhido | thr0.7_f2+man1.75+margin | **68,0** | 75,9 | **53,3** | 0,745 | 13,2 | 15,4 | 99,2 | 68,1 | 448 / 1100 | 9,5 s | 76,2 |
| mid_d50 (pareto) | thr0.7_f2+man1.6 | 67,5 | 77,9 | 48,0 | 0,740 | 14,8 | 15,6 | 99,4 | 67,5 | 448 / 1100 | 9,5 s | 74,1 |
| hi_d50 (pareto) | thr0.7_f2+man2 | **73,0** | 75,2 | **68,8** | 0,788 | 13,2 | 12,3 | 98,8 | 70,3 | 447 / 1099 | 9,5 s | 82,8 |

(F5 a/d50: médias das 2 rodadas da F5. "ouro no ctx" = a evidência do benchmark está
nos spans de fonte do contexto entregue. A coluna "cita item-ouro" mede o código de
item; nas perguntas de manual o "item-ouro" só existe quando o manual comenta um item.)

**Pareado vs ctrl_d50** (por rodada; ganhos/perdas de aprovação na mesma query):
op 46/12, 46/12, 54/13 (manual 42/1, 42/1, 47/3; sinal p<1e-6); hi 71/20, 70/16;
mid 46/15, 45/9.

### 2.1 Por vocabulário (aprovação, média das rodadas)

| célula | clean | technical | operational | lay |
|---|--:|--:|--:|--:|
| F5 a (app) | 63,4 | 57,6 | **50,6** | 51,2 |
| F5 d50 | 63,4 | 58,7 | **54,7** | 52,9 |
| ctrl_d50 | 64,0 | 58,5 | 55,4 | 51,9 |
| **op_d50** | **72,9** | **75,2** | **65,1** | **58,9** |
| hi_d50 | 77,3 | 76,2 | 73,9 | 64,6 |

| op_d50 por tipo × vocabulário | clean | technical | operational | lay |
|---|--:|--:|--:|--:|
| norma (ctrl → op) | 85,7 → 83,9 | 79,2 → 78,6 | 76,8 → 73,8 | 72,0 → 67,3 |
| manual (ctrl → op) | 23,3 → 52,2 | 20,0 → 68,9 | 15,6 → 48,9 | 14,4 → 43,3 |

No operational, que é o usuário real do app, a aprovação vai de **50,6% (app atual) e
54,7% (F5 d50) para 65,1%**, sem nenhum token a mais.

### 2.2 A norma caiu? Decomposição honesta

A norma observada caiu de 78,4 para 75,9 (−2,5 p.p.). Mas em **208 das 224** perguntas de
norma o turno tool do op_d50 é **byte a byte idêntico** ao do controle: a margem densa
não ligou os manuais, ou ligou e nenhum chunk de manual entrou no top-5. Nessas
perguntas o efeito real é zero por construção, e a diferença vem só da amostragem
(temp 0,1). Em contexto idêntico, ctrl e op deram respostas iguais só em 188 de 624
pares, e em 0 delas o veredito do juiz divergiu.

| | queries de norma | pareado w/l (soma das rodadas) | Δ em p.p. de norma |
|---|--:|--:|--:|
| op: contexto ALTERADO (efeito real) | 16 | 3 / 5 (p=0,73) | **−0,30** |
| op: contexto idêntico (só amostragem) | 208 | 12 / 27 | −2,2 (ruído) |
| mid: contexto alterado | 54 | 8 / 12 | −0,89 |
| hi: contexto alterado | 104 | 8 / 23 | **−3,35** |

- **No op o efeito causal sobre a norma é −0,3 p.p.**, igual ao do retrieval (−0,44 de
  hit@5). O resto da queda observada é sorteio da amostragem.
- Ressalva honesta: as perdas em contexto idêntico se repetem com sinal desfavorável nas
  3 rodadas (9/3, 10/3, 8/6 só no op). Não achamos mecanismo: as mesmas perguntas
  raramente perdem em mais de uma rodada (2 em comum entre r1 e r2), e o mesmo teste no
  mid e no hi dá 8/8 e 8/7. Tratamos como ruído com n pequeno. Um teste definitivo
  exige geração determinística (greedy), que muda o regime da série LFM3.
- **No hi (w=2) a queda de norma é real** (−3,4 p.p. causal, 23 perdas contra 8 ganhos
  onde o contexto mudou). É o trade-off do pareto.

### 2.3 Leitura

- **Manual 18% → 53%** no ponto escolhido, e **69%** no hi. A expectativa do capitão se
  confirma. Onde a evidência entra no contexto, o modelo converte 81–83% (op: ouro no ctx
  → aprovado 81,7/81,3/82,8%), o mesmo patamar da norma (ctrl 84%). **O leitor não piora
  com manual: o teto é recall.** Sem a evidência, a aprovação sobe de 18 para 23–24%,
  porque o chunk de manual vizinho ainda ajuda a responder.
- A **recusa cai pela metade** (27,5 → 15,4), quase toda nas perguntas de manual
  (61 → 28%). O modelo deixa de dizer "a norma não traz isso" quando o manual traz.
- A **alucinação sobe +1,7 p.p.** (11,5 → 13,2). Sai do mesmo lugar: responder mais
  perguntas de manual com contexto parcial (manual 12,5 → 14,4%).
- **Custo zero de prompt**: o mesmo orçamento de 448 tokens de contexto, 1100 de prompt
  e TTFT estimado de 9,5 s. Ligar manuais não mexe na latência do LFM, só na busca: dois
  sinais extras, FTS5 próprio e um KNN 1092×768.
- **mid × op**: com manual quase igual (48 contra 53%), o op tem menos alucinação (13,2
  contra 14,8) e a mesma aprovação. O gate por margem vale a pena.

## 3. Ressalva pendente da F5: d50 nas 151 coloquiais (régua honesta)

Prompts das 151 (`corpus/qa_pairs_v2.jsonl`) no mesmo formato do app, com o gabarito
`bench/regua/data/gabarito_151.jsonl`. O juiz checa alucinação contra o trecho-ouro
(`rejudge.load_gold_texts`). Mesma definição de aprovado do recorte: inclui ¬recusa. São
2 rodadas.

| célula | aprov. (r1 / r2 / média) | cobertura | alucin. | recusa | ouro no ctx (R@k) | ouro no ctx → aprovado | tok prompt | TTFT~ | pareado vs a (somado) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| a top-2 cru (app) | 16,6 / 17,2 / **16,9** | 0,239 | 22,2 | 53,6 | 52,3 | 27,2 | 1654 | 14,3 s | — |
| **d50** (F5, sem manuais) | 14,6 / 11,9 / **13,2** | 0,221 | 24,5 | 53,0 | 66,2 | 17,5 | 1112 | 9,6 s | 21 / 32 (p=0,17) |
| op_d50 (F6) | 15,9 / 14,6 / **15,2** | 0,233 | 23,9 | 53,3 | 66,9 | 20,8 | 1111 | 9,6 s | 27 / 32 (p=0,60) |

- **A trava dupla da síntese NÃO fecha a favor da compressão nas 151.** A d50 fica
  −3,7 p.p. abaixo do top-2 cru (média de 2 rodadas). O pareado não é significativo
  (p=0,17), mas o sinal é desfavorável nas duas rodadas. É o que o proxy da F5 previa:
  a fala coloquial longa não casa as frases e a compressão perde fatos (fatos no
  contexto 0,70 → 0,49). Aqui isso aparece como **conversão menor quando o ouro está no
  contexto (27 → 17,5%)**, apesar de o ouro entrar em mais queries (52 → 66%).
- **Os manuais não pioram as 151**: op_d50 ≥ d50 (16/10 pareado, p=0,33). Só 4 das 151
  recebem chunk de manual, e a margem densa praticamente não dispara nessas falas.
- **O nível absoluto das 151 é baixo em todas as células** (13–17%, recusa ~53%). O
  `f4-r32-ep2` recusa muito com as falas longas da POC, que pedem ações e dados que
  muitas vezes estão fora do trecho-ouro. Isso não muda entre células e não é efeito da
  F6.
- Conclusão para o embarque da compressão: no benchmark (falas que citam o item) a d50
  ≥ top-2 cru. Na fala coloquial longa ela tende a perder (−2 a −5 p.p. por rodada). **A
  compressão continua candidata, mas não fecha as duas réguas.** Opção de baixo risco:
  orçamento 70% (proxy da F5: 0,58 de fatos nas 151) ou top-2 cru + manuais. Essa
  célula ainda não foi medida e fica como próximo passo.

## 4. Pareto e recomendação

| ponto | manual hit@5 | norma hit@5 (Δ) | 151 R@2 | E2E global | E2E norma (causal) | E2E manual | alucin. |
|---|--:|--:|--:|--:|--:|--:|--:|
| sem manuais (ctrl) | 0 | 0,886 | 52,3 | 57,5 | — | 18,3 | 11,5 |
| **op: margem w=1,75** | 46,8 | 0,882 (−0,44) | 52,3 | **68,0** | **−0,3** | 53,3 | 13,2 |
| mid: w=1,6 | 53,7 | 0,877 (−0,96) | 52,3 | 67,5 | −0,9 | 48,0 | 14,8 |
| hi: w=2 | 78,0 | 0,861 (−2,55) | 50,3 | **73,0** | −3,4 | **68,8** | 13,2 |

- **Recomendação técnica: o ponto escolhido (`thr0.7_f2+man1.75+margin`)**. Cumpre a
  trava das duas réguas, entrega **+10,5 p.p. de aprovação global** (+35 p.p. no manual)
  com custo causal de norma de −0,3 p.p. e zero token a mais. Em código são 3 constantes
  além do gate da F5: peso 1,75, margem 0 e o sinal de manual ligado.
- **O hi (w=2) é a opção agressiva**: mais +5 p.p. global e +15 no manual, pagos com
  ~−3,4 p.p. de norma e −2 p.p. de 151 R@2, o que quebra a trava. Se o cliente pesa os
  manuais tanto quanto as normas (26% das 1536), o hi maximiza o global. Se a norma é o
  contrato, fica o op. **Decisão de produto do capitão.**
- Para ter manual alto sem custo de norma, o próximo passo é um **classificador de
  intenção treinado** (norma × manual) no lugar da margem densa. A margem já é melhor que
  o regex, mas só separa por similaridade.

## Reprodução

```bash
make -C bench/lfm3_f5 align manuals units          # caches da F5 (se ausentes)
make -C bench/lfm3_f6 retrieval units prompts       # fronteira da grade (as 49 var. = eval_retrieval_f6.py --variants <nomes em data/retrieval>) + janelas manuais + prompts
# Spark (setsid nohup; BATCH 8, max-new 768 — lição de OOM da F5), venv ~/cemig-poc/lfm3_f2/.venv:
#   python gen_e2e.py --model ~/cemig-poc/lfm3_f4/lfm3-1.7b-data2x-ep4-merged \
#     --adapter ~/cemig-poc/lfm3_f5/checkpoint-284 --tag f6-recorte --prompts data/e2e/prompts_recorte.jsonl --batch 8 --max-new 768
#   (idem --tag f6-recorte-rep2; f6-recorte-rep3 --cells ctrl_d50,op_d50; f6-151 / f6-151-rep2 com prompts_151.jsonl)
#   gen_e2e.py = bench/lfm3_f5/gen_e2e.py (sem mudança). ~17 min/1376 prompts a 1,35 it/s, ~11 GB.
make -C bench/lfm3_f6 score consolidate export
```

Artefatos: `data/retrieval/*.json` (49 variantes), `data/e2e/{prompts_*.jsonl.gz, gen_*,
scored_*, summary_*}`, `data/e2e/parity_d50_f5.json`, `data/consolidation.json`,
`data/eval/*.parquet` (7 parquets checkpoint × suíte × célula, schema estendido da F2/F4,
com `context_type` ∈ {`retrieved_top2_raw`, `retrieved_top5_compressed`,
`retrieved_top5_compressed_manuals`}, `context_given` exato, `ctx_ids`, `n_manual_ctx`,
`retrieval_variant` e `gold_kind`). Caches reprodutíveis (gitignored):
`cache/units_manual{.json,_win3.npz}` (10.125 janelas) e checkpoints de julgamento.
Os caches da F5 (`bench/lfm3_f5/cache/`) também são gitignored.
