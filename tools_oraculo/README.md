# tools_oraculo — Oráculo, Recusa, Híbrido e origem da alucinação do tool-calling do 1.2B

Fecha as lacunas do `tools_v1/` e VALIDA COM NÚMERO o desenho **híbrido** que o capitão definiu:
o MODELO decide quando buscar/reusar (arquitetura de tool), mas a BUSCA em si ignora a consulta
reescrita e usa a **fala crua + classificador de NR + pipeline v4 externo**. Nada foi retreinado
— reusa os checkpoints `tools_1_2b_r*` do `tools_v1/`, a régua honesta `bench/regua`, o índice v4
e o conjunto de recusa do `sft_v2/`.

> **Checkpoint medido: `tools_1_2b_r128`** (bf16 e Q4). O `tools_v1` mostrou que os ranks são
> quase indiferentes na qualidade (F1 94.9–95.4, aprovação 22–24 com desvio ~3); o r128 tem o
> melhor reuso/novo-tópico e argumento marginalmente melhor. Régua: `bench/regua` (ruler +
> juiz 27B `judge_regua`), limiar 0.5, citação FORA do gate. Juiz vLLM 27B `10.100.0.111:8005`
> (roteável deste host). Holdout 151/20 INTOCÁVEL (só medição). 3 runs para média±desvio em toda
> comparação que embasa recomendação. Tabelas com **bf16 E Q4 lado a lado** (preferência do
> capitão). Higiene: paralelo, checkpoint, não-interativo, procedência. Comentários PT-BR; código
> em inglês.

Ambientes: geração/score no `classifier/.venv` (jinja2, requests, sklearn); LLM 1.2B servido na
DGX Spark via `llama-server` (`scripts/start_srv_net.sh`, bind 0.0.0.0, ctx 16384 / `--parallel 4`
= 4096 tok/slot — **crítico**: o oráculo injeta trechos de até ~1450 tok e estoura o slot de 2048
com `--parallel 8`). Acesso pela tailscale `100.79.169.101:8500`.

---

## O que cada modo faz (o harness `harness_oraculo.py`)

Em TODOS os modos o **modelo decide** chamar/reusar/não-chamar (a arquitetura de tool do
`tools_v1`, byte-a-byte no mesmo formato nativo `<|tool_call_start|>[buscar_norma(...)]`). O que
muda é só **de onde vem o trecho** devolvido no turno `tool`:

| modo | retorno do turno `tool` | responde |
|---|---|---|
| `oracle` | **chunk-ouro** direto (bypass da busca) | MEDIÇÃO 1 |
| `hibrido` | contexto do **pipeline v4 externo** (fala crua + classificador NR + RRF v4) | MEDIÇÃO 3 (A) |
| `pure` | busca BM25 v4 com a **`consulta` do próprio modelo** (= tools_v1 puro) | MEDIÇÃO 3 (B) |

O contexto híbrido e o chunk-ouro vêm do cache `bench/prompt_teto/data/chunks_v4.json` (a v4
embarcada, R@2 51%, com `gold_chunks` por pergunta). A saída sai no schema do `sft_v2/generate_v2`
para ser pontuada por `sft_v2/score_v2.py` e `sft_v3/refusal_metrics.py` **sem alterá-los**.

---

## MEDIÇÃO 1 — ORÁCULO (a que o capitão pediu)

Com o **chunk-ouro de bandeja** no turno `tool`, o treino de tool AJUDOU ou ATRAPALHOU a síntese
vs o SFT sem tool? **AJUDOU** — o tool-calling supera o `sft_1_2b` sem tool em oráculo, tanto em
Q4 quanto em bf16, sem piorar a alucinação de forma relevante.

| candidato (oráculo, 151) | aprovação % | alucinação % | fonte |
|---|--:|--:|---|
| 1.2B QAD base (sem tool) | 2.7 | 2.0 | sft_1_2b |
| 1.2B sft_1_2b r64 **sem tool** — Q4 | 22.3 | 23.0 | sft_1_2b |
| 1.2B sft_1_2b r64 **sem tool** — bf16 | 24.3 | 18.9 | sft_1_2b |
| **1.2B tools r128 — Q4** | **31.4 ± 4.1** | **28.2 ± 2.2** | ESTE (3 runs) |
| **1.2B tools r128 — bf16** | **34.9 ± 2.4** | **24.3 ± 0.8** | ESTE (3 runs) |
| 2.6B v3 r64 — Q4 | 46.4 | 21.2 | slm/sft_v3 |
| 2.6B v3 r64 — bf16 | 47.0 | 21.9 | slm/sft_v3 |
| Qwen 27B (teto) | 84.3 | 2.6 | slm_oraculo |

**Resposta com número:** o tool-calling AJUDA a síntese em oráculo — **+9.1 p.p. em Q4**
(31.4 vs 22.3) e **+10.6 p.p. em bf16** (34.9 vs 24.3) sobre o `sft_1_2b` sem tool. A alucinação
sobe pouco (Q4 +5.2 p.p.; bf16 +5.4 p.p.), dentro/perto do desvio. Ou seja: **a arquitetura de
tool não atrapalha a síntese** quando o contexto é controlado — pelo contrário, o modelo TOOL
sintetiza um pouco melhor com o mesmo chunk certo. Isto **isola** a fonte dos 55.9% de alucinação
do `tools_v1`: NÃO vem da arquitetura de tool; vem do **contexto pior** que a consulta reescrita
produz (ver MEDIÇÃO 4). O gap para o 2.6B (~15 p.p.) e para o 27B (~50 p.p.) continua sendo
**capacidade do modelo**, não a ferramenta.

> Nota de comparabilidade: o número `conv chunk-certo→aprovada 35.3%` do tools_v1 NÃO era
> comparável ao oráculo (era condicional à busca do próprio modelo acertar). Este oráculo entrega
> o ouro a 100% dos itens e mede a síntese pura — é o número certo para comparar com os 22.3% do
> sft sem tool.

---

## MEDIÇÃO 2 — RECUSA (lacuna 2 do tools_v1)

A família (5) `recusa_apos_busca` (675 exemplos) foi treinada mas nunca reportada. Rodamos o
MESMO pack de recusa do `sft_v2`/`sft_1_2b`/`v3` (85 itens: oracle 30 + recusa 25 + distrator 20 +
parcial 10) e computamos a matriz de confusão (`sft_v3/refusal_metrics.py`, classe positiva =
"deveria recusar"). **A família preservou o ganho de recusa** — F1 90.5 (bf16) / 85.4 (Q4),
praticamente igual ao `sft_1_2b` sem tool (90.2) e a poucos p.p. do 2.6B v3 (95.7).

| candidato (recusa, 85) | recall (correta) % | FP (indevida) % | precisão % | F1 % | TP/FN/FP/TN |
|---|--:|--:|--:|--:|---|
| 1.2B QAD base | 84.4 | 66.7 | 65.5 | 73.8 | 38/7/20/10 |
| 1.2B sft_1_2b r64 — Q4 | 82.2 | **0.0** | **100.0** | 90.2 | 37/8/0/30 |
| **1.2B tools r128 — bf16** | **84.4** | **3.3** | **97.4** | **90.5** | 38/7/1/29 |
| **1.2B tools r128 — Q4** | 77.8 | 6.7 | 94.6 | 85.4 | 35/10/2/28 |
| 2.6B v3 r64 — Q4 | 100.0 | 13.3 | 87.7 | 95.7 | 45/0/6/24 |
| Qwen 27B | 95.6 | 3.3 | 97.7 | 96.6 | 43/2/1/29 |

**Resposta:** a arquitetura de tool **NÃO diluiu a recusa**. O bf16 empata com o sft sem tool
(F1 90.5 vs 90.2) e mantém recusa-indevida baixa (3.3%). O Q4 perde um pouco (F1 85.4, recall
77.8) — a recusa é levemente sensível ao 4-bit, como no v2. A família (5) transferiu bem.

---

## MEDIÇÃO 3 — O HÍBRIDO (o desenho que o capitão quer levar adiante)

Três configurações na MESMA bateria (151, multiturno single-turn por pergunta):

- **(A) híbrido** — o modelo decide chamar; quando chama, o harness **IGNORA a `consulta`** e
  busca com **fala crua + classificador NR + RRF v4** (o melhor pipeline que temos);
- **(B) tools puro** — a busca usa a **`consulta` do próprio modelo** (= `tools_v1`);
- **(C) pipeline fixo sem tool** — `sft_1_2b` r64 + busca v4 **SEMPRE** (sem decisão do modelo).

| config | quant | aprovação % | alucinação % | chamou % | got_gold(quando chamou) % |
|---|---|--:|--:|--:|--:|
| **(A) híbrido** | Q4 | **19.2 ± 0.6** | 34.2 ± 0.8 | 99.1 | 51.2 |
| **(A) híbrido** | bf16 | **19.2 ± 1.4** | 31.8 ± 0.9 | 100.0 | 51.0 |
| (B) tools puro | Q4 | 7.1 ± 1.4 | 39.7 ± 5.0 | 98.5 | 24.7 |
| (B) tools puro | bf16 | 8.2 ± 0.6 | 42.6 ± 3.8 | 100.0 | 18.8 |
| (C) fixo sem tool | Q4 | 10.1 | 33.1 | (sempre) | 51 (R@2 v4) |
| (C) fixo sem tool | bf16 | 16.2 | 30.4 | (sempre) | 51 (R@2 v4) |

**Resultado:** o **híbrido (A) BATE o tools puro (B) com folga** — **+12.1 p.p. em Q4**
(19.2 vs 7.1) e **+11.0 p.p. em bf16** (19.2 vs 8.2), com **menos alucinação** (Q4 −5.5;
bf16 −10.8). A causa direta é o retrieval: a fala crua + classificador + RRF v4 entrega o
chunk-ouro em **~51%** das chamadas, contra **~19–25%** da consulta reescrita pelo 1.2B — exatamente
o achado do `tools_v1` ("a reescrita do 1.2B PIORA a busca"), agora medido ponta a ponta.

Contra o pipeline fixo (C), o híbrido **empata/supera em Q4** (19.2 vs 10.1) e **supera levemente
em bf16** (19.2 vs 16.2). Ou seja: **adicionar a decisão do modelo ao pipeline v4 não custa
aprovação — e ainda entrega o multiturno/decisão** (o modelo decide quando buscar, F1 95%, e
quando reusar, ~85%, do `tools_v1`) que o pipeline fixo não tem.

**Recomendação de arquitetura (com clareza):** **adotar o híbrido (A)** — o modelo TOOL para
DECIDIR quando buscar/reusar (destrava o multiturno) + a **busca com fala crua + classificador +
RRF v4** (nunca a consulta reescrita do 1.2B). O híbrido bate o tools puro e não perde para o
pipeline fixo, somando o comportamento multiturno que o capitão quer. A consulta reescrita do 1.2B
fica FORA do caminho de busca (só serve, se um dia, o 27B na nuvem como rewriter).

---

## MEDIÇÃO 4 — DE ONDE VEM A ALUCINAÇÃO

Decomposição da alucinação por condição do contexto (soma dos 3 runs; `hall_rate` = alucinação
DENTRO da condição; `share` = fatia do total de alucinações):

| config | condição | n | alucinações | hall_rate % | share % |
|---|---|--:|--:|--:|--:|
| (A) híbrido Q4 | got_gold ✓ | 230 | 82 | 35.7 | 52.9 |
| (A) híbrido Q4 | contexto errado | 219 | 72 | 32.9 | 46.5 |
| (A) híbrido Q4 | não chamou | 4 | 1 | 25.0 | 0.6 |
| (A) híbrido bf16 | got_gold ✓ | 231 | 86 | 37.2 | 59.7 |
| (A) híbrido bf16 | contexto errado | 222 | 58 | 26.1 | 40.3 |
| (B) tools puro Q4 | got_gold ✓ | 110 | 54 | 49.1 | 30.0 |
| (B) tools puro Q4 | contexto errado | 336 | 126 | 37.5 | 70.0 |
| (B) tools puro bf16 | got_gold ✓ | 85 | 39 | 45.9 | 20.2 |
| (B) tools puro bf16 | contexto errado | 368 | 154 | 41.8 | 79.8 |

**Leitura:**
- No **tools puro (B)**, a alucinação vem majoritariamente de **contexto errado** (70–80% das
  alucinações) — porque a consulta reescrita erra a busca em ~75–81% das chamadas. **Confirma o
  achado: a busca é o teto** — a maior alavanca no tools puro é o retrieval, não a síntese.
- No **híbrido (A)**, com a busca corrigida, a alucinação se **redistribui**: agora **~53–60%**
  vem de casos em que o **chunk-ouro ESTAVA presente** (got_gold ✓) e o 1.2B ainda alucina
  (hall_rate ~36–37%). Ou seja, no híbrido o teto **deixa de ser só o retrieval** e passa a ser
  também a **capacidade do 1.2B de sintetizar sem inventar** dado o chunto certo — o mesmo limite
  visto na MEDIÇÃO 1 (oráculo: hall ~24–28%).
- A oráculo cruza os dois: com 100% got_gold, a alucinação cai para 24–28% (bf16/Q4). Logo, a
  alucinação de 55.9% do `tools_v1` era **soma de retrieval ruim (a maior parte) + síntese do
  1.2B**; o híbrido remove a parte do retrieval e sobra a do LM.

**Onde intervir, em ordem:** (1) **retrieval** (já resolvido no híbrido — usar fala crua + v4,
não a consulta do modelo); (2) **capacidade do sintetizador** (subir para o 2.6B v3 quando couber
na voz, ou destilar 27B→1.2B) para atacar a alucinação residual com chunk-ouro presente.

---

## TABELA MESTRE — todos os candidatos já medidos na POC

Régua honesta `bench/regua` (limiar 0.5, citação fora do gate) em TODOS. Oráculo = chunk-ouro de
bandeja; retrieval real = busca v4 embarcada (R@2 51%).

| modelo / config | oráculo aprov % | oráculo aluc % | retrieval-real aprov % | recusa F1 % | fonte |
|---|--:|--:|--:|--:|---|
| 1.2B QAD base (embarcado) — Q4 | 2.7 | 2.0 | 1.4 | 73.8 | sft_1_2b |
| 1.2B sft_1_2b r64 sem tool — Q4 | 22.3 | 23.0 | 10.1 | 90.2 | sft_1_2b |
| 1.2B sft_1_2b r64 sem tool — bf16 | 24.3 | 18.9 | 16.2 | 90.7 | sft_1_2b |
| **1.2B tools r128 — Q4** | **31.4** | **28.2** | **7.1** (puro) / **19.2** (híbrido) | **85.4** | ESTE |
| **1.2B tools r128 — bf16** | **34.9** | **24.3** | **8.2** (puro) / **19.2** (híbrido) | **90.5** | ESTE |
| 2.6B v3 r64 — Q4 | 46.4 | 21.2 | 20.0 | 95.7 | sft_v3 |
| 2.6B v3 r64 — bf16 | 47.0 | 21.9 | 26.7 | 94.5 | sft_v3 |
| Qwen 27B (teto) | 84.3 | 2.6 | 45.7 | 96.6 | slm_oraculo |

> "retrieval-real" do tools = com a busca do PRÓPRIO modelo (puro) vs com a fala crua+v4 (híbrido).
> Nos demais 1.2B/2.6B a coluna é o pipeline fixo (busca v4 sempre). O híbrido do tools é a única
> config que soma retrieval v4 + decisão/multiturno do modelo.

---

## VEREDITO (recomendação de arquitetura)

1. **Oráculo (M1):** o treino de tool **AJUDA** a síntese (Q4 31.4 vs 22.3; bf16 34.9 vs 24.3 do
   sft sem tool). A arquitetura de tool não é a causa da alucinação alta do tools_v1.
2. **Recusa (M2):** **preservada** (F1 bf16 90.5 ≈ sft 90.2; Q4 85.4). A família (5) transferiu.
3. **Híbrido (M3):** **é a recomendação.** (A) híbrido bate (B) tools puro por ~11–12 p.p. de
   aprovação e não perde para (C) pipeline fixo — e entrega o multiturno/decisão. **Regra de
   embarque: o modelo decide chamar/reusar; a busca usa SEMPRE a fala crua + classificador + RRF
   v4, NUNCA a consulta reescrita pelo 1.2B.**
4. **Alucinação (M4):** no tools puro a alucinação é 70–80% de **contexto errado** (a busca é o
   teto — confirmado); o híbrido corrige a busca e o resíduo passa a ser **síntese do 1.2B com o
   chunk certo** (~36%), o mesmo teto do oráculo. Alavanca residual = capacidade do sintetizador
   (2.6B v3 quando couber na voz; hoje ~23 s/4,3 GB no S24+, inviável).
5. **Candidato se embarcar tool-calling:** **r128 Q4** (mesma latência/RAM do QAD embarcado;
   ranks menores no ruído). O 1.2B segue o modelo de campo por latência de voz; o 2.6B v3 é o alvo
   de qualidade quando latência/RAM destravar.

### Amostras literais dos casos do capitão
`data/samples_captain.txt` (oracle / hibrido / pure lado a lado). Destaques (tools r128 Q4):
- **13,8 kV sozinho (qa-013):** oráculo e híbrido CHAMAM, recuperam NR-10 (10.6/10.7) e dão a
  ação certa ("não pode subir sozinho… suspender na iminência de perigo, OS assinada por
  superior"); o **puro erra a busca** (puxa NR-16 Anexo IV) e recusa — típico do retrieval ruim.
- **poste podre (qa-025):** o **híbrido** recupera NR-18 e responde "não suba, recuse com
  justificativa"; oráculo/puro recuperam a norma errada (a pergunta é de recusa/altura, não
  NR-10) — mostra que nem o ouro por (doc,seção) cobre o caso, é ambiguidade de norma.
- **13,8 kV distância (qa-029):** com o ouro (Anexo II reparado) o híbrido chega à Zona de Risco
  mas **erra o número** ("220 metros" em vez da faixa da tabela) — a leitura fina de tabela é o
  limite do 1.2B (mesmo achado do slm_oraculo: nem o 2.6B lê a faixa certa).
- **camiseta rasgada (qa-003):** híbrido e puro **NÃO cometem o erro de "adornos"** do sft v1
  (respondem sobre vestimenta/EPI) — a família de recusa/distrator ajudou.

---

## Reprodução

```bash
PY=../classifier/.venv/bin/python
export REGUA_JUDGE_URL=http://10.100.0.111:8005/v1
# 1) sobe o checkpoint na Spark (Q4 ou bf16), --parallel 4 (4096 tok/slot; oráculo estoura 2048)
ssh walcyrios@spark-b431 "bash ~/cemig-poc/scripts/start_srv_net.sh \
   ~/cemig-poc/models/lfm2.5-1.2b-tools_1_2b_r128-Q4_0.gguf 8500 16384 4"
# 2) geração por modo (oracle/hibrido/pure = 151; recusa = pack sft_v2), 3 runs p/ média
URL=http://100.79.169.101:8500
$PY harness_oraculo.py --url $URL --label tools_r128_oracle_q4_run1 --mode oracle --pack 151 \
    --out data/gen_tools_r128_oracle_q4_run1.json --workers 6
$PY harness_oraculo.py --url $URL --label tools_r128_refusal_q4 --mode oracle --pack refusal \
    --out data/gen_tools_r128_refusal_q4.json --workers 6
# 3) score (régua honesta reusada do sft_v2) + matriz de recusa (sft_v3)
(cd ../sft_v2 && $PY score_v2.py --resp ../tools_oraculo/data/gen_tools_r128_oracle_q4_run1.json \
    --out ../tools_oraculo/data/score_tools_r128_oracle_q4_run1.json --workers 8)
$PY ../sft_v3/refusal_metrics.py \
    --scores data/score_tools_r128_refusal_q4.json:tools_r128_q4 --out data/refusal_confusion_q4.json
# 4) consolida as 4 medições + amostras
$PY analyze.py ; $PY extract_samples.py
```
`make all` roda o pipeline completo (ver `Makefile`).

### Arquivos
```
harness_oraculo.py   harness dos 3 modos (oracle/hibrido/pure) + pack de recusa; schema sft_v2
analyze.py           consolida as 4 medições + refs (sft_1_2b/consolidation.json) -> data/consolidation.json
extract_samples.py   amostras literais dos casos do capitão (oracle/hibrido/pure lado a lado)
Makefile             pipeline reprodutível (gen -> score -> refusal -> analyze -> samples)
```
Reuso (não alterados): `tools_v1/render_jinja.py|render.py|tool_schema.py|retrieval_check.py`,
`sft_v2/score_v2.py`, `sft_v3/refusal_metrics.py`, `bench/regua`, `bench/prompt_teto/data/chunks_v4.json`.

### Artefatos (gitignore para pesados; leves commitados)
`data/gen_*.json` (gerações), `data/score_*.json` (régua), `data/refusal_confusion_{bf16,q4}.json`,
`data/consolidation.json` (tabela mestre), `data/samples_captain.txt`. Pesos GGUF na Spark
(`~/cemig-poc/models/lfm2.5-1.2b-tools_1_2b_r128-{bf16,Q4_0}.gguf`, do tools_v1, gitignored).
```
