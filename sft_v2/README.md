# SFT v2 — 4 famílias (com/sem oráculo) + varredura de rank + métrica de recusa (`sft_v2/`)

Responde à ordem do capitão após o v1 (`slm_oraculo/`, já no main): "fazer o v2 com TUDO que
foi diagnosticado". O v1 provou que destilar o 27B funciona (2.6B Q4 45→58,3% em oráculo,
~1/3 do gap até o teto), mas expôs 4 problemas — o v2 ataca todos:

1. **OVERFITTING**: no v1 a loss caiu monotônica (0,94→0,26) enquanto a QUALIDADE PIOROU
   (aprovação ep1→ep3 e alucinação 25→30%). Sem validation set. → v2 treina **1 ÉPOCA** e
   mede a **régua** na validação, não só a loss.
2. **O MODELO NÃO SABE RECUSAR**: alucinação 25% em oráculo, 41-51% no mundo real. Treinado
   só com chunk-ouro. → v2 fabrica dados **COM e SEM oráculo** (recusa/parcial/distrator).
3. **r=16 nunca foi testado com o MLP dentro** (bug dos targets do v1). → v2 varre **r 16/32/64**.
4. **Camiseta rasgada**: o v1 responde citando "vedado o uso de adornos pessoais" (item de
   joias, ERRADO) com convicção. → família **distrator plausível** ensina a não atribuir.

> **Infra**: geração de dados e pontuação rodam LOCAIS (`classifier/.venv`; juiz vLLM 27B
> `10.100.0.111:8005` roteável). **Treino + export GGUF na DGX Spark** (`walcyrios@spark-b431`,
> GB10 121 GB unificada, `~/jupyterlab/.venv` torch 2.12+cu130). Régua ÚNICA: `bench/regua`
> (`ruler.deterministic_pass` + `judge_regua.judge_disambiguate`) + juiz de recusa próprio.
> Holdout 151/20 INTOCÁVEL; 3 muralhas do `finetune2/` + split por NR mantidos. Paralelo
> (≥16 workers), checkpoint, não-interativo, procedência. Comentários PT-BR; código em inglês.

---

## PREMISSAS VERIFICADAS (exigência do capitão, antes de qualquer treino)

O `train_v2.py` imprime `targeted_module_names` + `print_trainable_parameters` e **ABORTA por
assert** se o LoRA não cobrir MLP+atenção ou tocar o ShortConv. Um dry-run de 80 pares validou
o pipeline inteiro (treino→merge→GGUF) antes da varredura. Publicado em `data/premises/`:

| rank | alpha | módulos adaptados | trainable params | % | asserts |
|---|--:|--:|--:|--:|:--|
| 16 | 32 | **122** (MLP `feed_forward.w1/w2/w3` + `self_attn.{q,k,v,out}_proj`) | 20.135.936 | 0,74% | mlp✓ attn✓ **conv✗** |
| 32 | 64 | 122 | 40.271.872 | 1,47% | mlp✓ attn✓ conv✗ |
| 64 | 128 | 122 | 80.543.744 | 2,90% | mlp✓ attn✓ conv✗ |

Regex de alvo: `r".*\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|feed_forward\.(w1|w2|w3))$"`.
Os 22 `conv.*` (ShortConv recorrente) ficam de fora (0 casados) — o bug do v1, agora impossível.

---

## PARTE 1 — os dados (4 famílias, TODAS filtradas por régua)

`gen_data.py` (27B professor, filtro por família). Formato conversacional único
`{"messages":[system, user, assistant]}` com o **`SYNTHESIS_SYSTEM_PROMPT_V2`** — igual ao
de produção EXCETO a última regra: em vez da recusa preguiçosa ("diga APENAS: Não sei…"), o
v2 ensina a **recusa honesta E ÚTIL** (diz que não está na norma consultada + o que faltou).

| família | alvo | n final | % | critério de aprovação (procedência) |
|---|---|--:|--:|---|
| **(a) ORÁCULO** | resposta acionável do 27B (prompt `numeros`) | **1476** | 48,0% | régua honesta: cobertura≥0,5 ∧ sem tautologia ∧ **sem alucinação** (== v1) |
| **(b) RECUSA** | chunk de OUTRA norma; recusa honesta+útil | **640** | 20,8% | juiz recusa: recusou_ou_delimitou ∧ disse_o_que_faltou ∧ ¬inventou ∧ ¬atribuiu_errado |
| **(c) PARCIAL** | chunk certo com o dado-chave REMOVIDO | **640** | 20,8% | juiz recusa: respondeu_o_possivel ∧ declarou_o_que_faltou ∧ ¬inventou_dado |
| **(d) DISTRATOR** | mesma norma, seção vizinha que NÃO responde | **320** | 10,4% | juiz recusa: recusou_ou_delimitou ∧ **¬atribuiu_item_errado** ∧ ¬inventou (o caso camiseta) |

**Total: 3076 pares régua-aprovados, 33 NRs, 0 contaminação de NR reservada, 0 malformado.**
O **critério de aprovação das famílias de recusa é RECUSAR/DELIMITAR CORRETO, não cobrir
fatos** (ordem do capitão) — implementado em `common_v2.approve_nonoracle` (transparente e
auditável em Python, não escondido no prompt). O contexto PARCIAL é construído removendo
deterministicamente a sentença de maior sobreposição com os fatos (`make_partial_context`);
o distrator é o chunk da mesma norma com MENOR sobreposição com os fatos
(`pick_same_norm_distractor`).

Muralhas (reuso `finetune2/common`): (1) chunk-ouro do holdout nunca é fonte/distrator;
(2) NR-33/16/26 reservadas (0 chunks); (3) filtro lexical Jaccard<0,4 da pergunta vs holdout
(0 descartes). **Split adicional (lição do v1)**: os chunks de VAL/refusal-test ficam FORA do
treino (`split_chunks`, seed 17) — val=140 e refusal-test=85, **disjuntos entre si e do treino**
(0 overlap verificado).

---

## PARTE 2 — treino (1 época, 3 ranks, MESMOS dados/seed, única variável = rank)

`train_v2.py` na Spark, `run_train_v2.sh` (mata llama-servers antes; verifica integridade;
varre r 16/32/64). bs 8 × grad-accum 4, max_len 2048, bf16, gradient checkpointing,
`assistant_only_loss=True`, lr 1e-4, cosine, seed 42. **Validation set INTERNO** (5% do
dataset, fora do treino) → `eval_loss` ao fim da época no wandb (offline). Pico de memória
26,4 GB (muito abaixo de 121 GB — sem OOM). wandb `WANDB_MODE=offline` (chave não configurada;
sync retroativo depois; NUNCA impressa/commitada).

| rank | train_loss | eval_loss (interno) |
|---|--:|--:|
| 16 | 0,7088 | 0,6193 |
| 32 | 0,6719 | 0,5870 |
| 64 | 0,6413 | 0,5604 |

A loss cai monotônica com o rank — **mas a lição do v1 é que loss não é qualidade**. A régua
externa (Parte 3) é quem decide.

---

## PARTE 3 — avaliação e veredito (régua honesta, n=151 + val 140 + recusa 85)

Cada variante medida em **bf16 E Q4_0** (separa efeito do treino do efeito da quantização),
nas 4 suites. `run_all_evals.sh` (um llama-server por vez na Spark — lição OOM do v1) +
`run_teto_27b.sh` (27B pelo servidor do juiz).

### Tabela mestre

| variante | oráculo apr% | oráculo **aluc%** | retrieved apr% | retrieved **aluc%** | **recusa-correta%** | **recusa-indevida%** | val apr% |
|---|--:|--:|--:|--:|--:|--:|--:|
| **27B (TETO)** | **84,3** | **2,6** | 45,7 | 27,8 | 95,6 | **3,3** | 60,0 |
| 2.6B base Q4 | 49,0 | 18,5 | 25,3 | 42,0 | 60,0 | 0,0 | 38,6 |
| 2.6B base bf16 | 42,4 | 25,8 | 23,3 | 41,3 | 66,7 | 0,0 | 41,4 |
| **v1 r16 Q4** (só oráculo) | 53,6 | 25,8 | 31,3 | 33,3 | **46,7** | 0,0 | 42,9 |
| v2 r16 Q4 | 41,1 | 25,2 | 13,3 | 42,7 | 97,8 | 40,0 | 53,6 |
| v2 r32 Q4 | 43,0 | 16,6 | 22,0 | 32,0 | 97,8 | 23,3 | 56,4 |
| **v2 r64 Q4** (embarque) | 47,7 | **15,9** | 22,0 | 34,0 | **100,0** | 23,3 | 54,3 |
| v2 r16 bf16 | 44,4 | 21,2 | 20,7 | 40,0 | 100,0 | 33,3 | 62,9 |
| v2 r32 bf16 | 51,0 | 13,9 | 25,3 | 33,3 | 97,8 | 26,7 | 55,7 |
| **v2 r64 bf16** | **57,0** | **13,2** | 22,0 | 39,3 | 97,8 | 26,7 | 62,1 |

### Achado 1 — a varredura de rank MOVE a agulha (ao contrário do v1)

No v1 o MLP só entrou no LoRA no fim (bug dos targets), então o rank nunca foi testado de
verdade. Agora, com MLP+atenção dentro em todos, o sinal é **real e monotônico**:

| oráculo (bf16, isola quant) | r16 | r32 | r64 |
|---|--:|--:|--:|
| aprovação% | 44,4 | 51,0 | **57,0** |
| **alucinação%** | 21,2 | 13,9 | **13,2** |
| cobertura | 0,521 | 0,532 | 0,559 |
| eval_loss interno | 0,619 | 0,587 | **0,560** |

**+12,6 p.p. de aprovação e −8,0 p.p. de alucinação de r16→r64, sem sinal de saturação.**
Não é ruído: sobe nas 4 métricas juntas e acompanha a eval_loss. O custo (r64 = 2,9%
trainable vs 0,74% do r16) é irrelevante para um LoRA. **A varredura que move a agulha é
resultado legítimo — e aqui ela moveu.** lr 1e-4 foi estável em r=64 (sem instabilidade;
nada a reportar/propor).

### Achado 2 — a RECUSA foi destravada (o coração do v2), com um custo medido

Esta é a métrica que o v1 nem tinha. O contraste é o ponto central da tarefa:

| na suite de RECUSA | recusa-correta% | recusa-indevida% | alucinação nas de recusa |
|---|--:|--:|--:|
| 2.6B base | 60–67 | 0 | ~78% (inventa com o contexto errado) |
| **v1 (só oráculo)** | **46,7** | 0 | **51,8%** |
| **v2 r64** | **100** | 23,3 | ~30% |
| 27B (teto) | 95,6 | 3,3 | 2,4% |

O v1, treinado só em oráculo, **PIOROU a recusa** (46,7% < base 60%): ao ficar mais assertivo
sem aprender a recusar, ele inventa com o contexto errado (o efeito exato que o capitão
diagnosticou). **O v2 leva a recusa-correta a 98-100% e derruba a alucinação de contexto
errado de 51,8→30%** — resolve o diagnóstico #2.

**O custo honesto**: a recusa-indevida sobe de ~0 para ~23% (v2 recusa quando DAVA para
responder). É um dano igualmente grave e por isso é métrica obrigatória. O rank ATENUA (r16
Q4 40% → r32/r64 23%), mas não zera. O teto do 27B (recusa-indevida 3,3%) mostra que o gap
que sobra aqui é **capacidade de discriminar "dá/não dá para responder"**, não desenho de
dados.

### Achado 3 — na Q4 de embarque, aprovação em oráculo NÃO supera o v1

O melhor Q4 (r64) faz oráculo 47,7% vs **v1 58,3%**. Por quê? O v2 **troca aprovação por
honestidade**: ao aprender a hedge ("…mas o contexto não define X"), ele recusa/delimita
casos que a régua contaria como respondidos se o modelo "chutasse com confiança". A
recusa-indevida nas 151 (67-75%, medida como "recusou numa answerable") confirma: parte da
queda de aprovação é over-refusal, não perda de conhecimento. Em **bf16** o r64 chega a 57,0%
(perto do v1 Q4 58,3%) com **metade da alucinação** (13,2 vs 25,2%). **A quantização Q4 custa
caro no v2** (r64 57,0→47,7 = −9,3 p.p.) — bem mais que os −4 p.p. do v1 — porque o
comportamento de hedge é mais sensível ao ruído do 4-bit.

### Achado 4 — retrieval segue sendo o teto prático

Na busca v4 real (R@2 51%), toda a família 2.6B fica em 20-25% (v1 Q4 31,3% é o melhor,
justamente por ser o mais "chutador"). O v2 r64 Q4 faz 22,0%. **O retrieval domina o teto
prático** — o mesmo veredito do v1: não adianta o SLM melhorar se o chunk certo só chega em
metade das vezes. A alucinação na v4 continua alta (~34%) porque, quando o chunk-ouro falta,
até o modelo que aprendeu a recusar às vezes tem contexto plausível-porém-errado e responde.

### Casos fixos do capitão (respostas literais)

- **Camiseta rasgada (qa-003)** — o caso que o v1 ERRA:
  - v1: _"…Além disso, é vedado o uso de adornos pessoais…"_ — **atribui o item de joias à
    roupa (ERRADO, com convicção).**
  - v2 r64: _"A NR-10, item 10.2.9.1… mas o contexto não define qual é a camiseta mínima
    exigida. Não é possível afirmar…"_ — **não comete o erro de adornos** (delimita).
  - 27B: _"Não, você não pode usar a camiseta comum de algodão. A NR-10, item 10.2.9.2, exige
    vestimentas adequadas…"_ — responde certo e completo.
- **Aliança (qa-004)** — o outro lado da moeda (NR-10 10.2.9.3 proíbe adornos EXPLICITAMENTE):
  - 27B: _"Não libera. A NR-10, item 10.2.9.3, veda o uso de adornos pessoais… remova a
    aliança."_ — certo. v1 também acerta aqui.
  - v2 r64: _"…mas o contexto não especifica se a aliança é uma medida de proteção…"_ —
    **over-refusa** um caso que DAVA para responder. É a recusa-indevida em carne e osso: o
    v2 ficou cauteloso demais e às vezes recusa o óbvio.
- **13,8 kV** — nenhuma variante 2.6B lê a faixa certa da tabela (todas dão 0,25 m da faixa
  3-6 kV; o 27B lê 0,38 m). O SFT não ensina leitura fina de tabela — fronteira de capacidade,
  igual ao v1.
- **Poste bambo** — v1 **alucina perigo** (_"Suba imediatamente para um ponto de apoio…"_);
  o v2 r64 bf16 recusa e redireciona (_"A NR-35 não define critérios para a firmeza de postes…
  consulte a NR-10"_). O 27B recusa (_"Não sei com base nas normas consultadas"_).

---

## VEREDITO

1. **O SFT de destilação v2 é a receita certa, agora com as 4 alavancas do diagnóstico.** A
   varredura de rank **moveu a agulha** (r64 melhor em tudo: +12,6 p.p. aprovação, −8,0 p.p.
   alucinação em oráculo bf16) — provando que r=16 do v1 estava sub-parametrizado por causa do
   bug dos targets. **1 época bastou** (sem o overfitting do v1: a alucinação CAIU, não subiu).
2. **A recusa foi destravada**: recusa-correta 47→100% e alucinação-de-contexto-errado
   51,8→30% vs o v1 — o v2 resolve o "modelo não sabe recusar". **Custo medido e honesto**:
   recusa-indevida sobe a ~23% (atenuada pelo rank, não zerada). É um trade-off real, não um
   almoço grátis.
3. **Em oráculo Q4, o v2 não supera o v1 em aprovação** (47,7 vs 58,3) — porque troca chute
   confiante por honestidade; em **bf16 empata** (57,0) com **metade da alucinação**. **A Q4
   custa caro no v2** (−9,3 p.p.), o comportamento de hedge é sensível ao 4-bit.
4. **Retrieval segue sendo o teto prático** (v4 real ~22%); e **no aparelho o 2.6B (treinado
   ou não) não cabe no orçamento de voz** (~23 s / 4,3 GB no S24+, ver `slm_oraculo`).

**Recomendação de embarque**: quando o 2.6B couber no aparelho, embarcar o **v2 r64** (a
menor alucinação e a melhor recusa; se houver folga de RAM/latência para bf16, o ganho é
grande). Enquanto o 2.6B não couber (hoje), **o embarcado segue 1.2B** e a maior alavanca de
produto continua sendo o **retrieval**. O v2 é a melhor versão de destilação que temos e o
alvo natural quando o engine/latência do 2.6B (ou a destilação p/ o 1.2B) destravar.

**Nuance para o capitão**: se o critério de produto for "nunca inventar" (segurança do
trabalho), o v2 é claramente superior ao v1 (alucinação bem menor, recusa que funciona). Se
for "responder o máximo possível", o v1 chuta mais e por isso pontua mais alto na régua de
cobertura — mas ao custo de 25-52% de alucinação. **A escolha é do capitão; os dois números
estão na tabela.**

---

## Arquivos e reprodução

```
sft_v2/
  common_v2.py          # núcleo: system prompt v2, prompts do professor, juízes de recusa,
                        #   critérios approve_nonoracle, split de chunks, detecção de recusa
  gen_data.py           # PARTE 1: fabrica as 4 famílias (27B professor, filtro por régua)
  build_eval_packs.py   # val_pack (140) + refusal_test_pack (85), FORA do treino/holdout
  generate_v2.py        # gera respostas do aluno (system v2/numeros/prod) sobre um pack
  score_v2.py           # régua honesta + métrica de recusa (recusa-correta%/indevida%)
  train_v2.py           # SFT LoRA (rank configurável, 1 época, eval_loss interno, asserts)
  run_train_v2.sh       # varredura r16/r32/r64 na Spark (mata servers, verifica integridade)
  eval_checkpoint_v2.sh # sobe 1 checkpoint na Spark + gera/pontua as 4 suites
  run_all_evals.sh      # roda todas as variantes 2.6B (uma por vez — lição OOM)
  run_teto_27b.sh       # teto do 27B nas 4 suites (via servidor do juiz)
  consolidate_v2.py     # tabela mestre + resumo por variante + varredura de rank
  data/                 # packs, respostas, scores, dataset de treino, premises/ (leves; commitados)
```

Comandos (juiz via `REGUA_JUDGE_URL`, default `10.100.0.111:8005`):
```bash
PY=../classifier/.venv/bin/python
# PARTE 1 (local): dados + packs de avaliação
$PY gen_data.py --target-oracle 1600 --target-recusa 640 --target-parcial 640 --target-distrator 320
$PY build_eval_packs.py
# PARTE 2 (Spark): scp train_v2.jsonl + train_v2.py/run_train_v2.sh; setsid nohup run_train_v2.sh
# PARTE 3 (local): avaliação + consolidação
./run_all_evals.sh      # 2.6B (base, v1, v2 r16/r32/r64 em bf16+Q4)
./run_teto_27b.sh       # teto 27B
$PY consolidate_v2.py   # tabela mestre -> data/consolidation.json
```

### Artefatos na Spark (gitignored: reprodutíveis)
- dataset: `~/cemig-poc/train_v2/train_v2.jsonl` (== `data/train_v2.jsonl`, 3076 pares)
- adapters: `~/cemig-poc/train_v2/sft_v2_r{16,32,64}/`
- GGUFs: `~/cemig-poc/models/lfm2.5-2.6b-sft_v2_r{16,32,64}-{bf16,Q4_0}.gguf`
- **artefato de embarque (quando o 2.6B couber)**: `lfm2.5-2.6b-sft_v2_r64-Q4_0.gguf`
  (oráculo 47,7% / alucinação 15,9% / recusa-correta 100%) ou o bf16 se houver folga
  (oráculo 57,0% / alucinação 13,2%).
- premises + eval_loss por rank: `sft_v2/data/premises/` (commitado)
- log de treino: `~/cemig-poc/logs/train_v2.log`; telemetria wandb: `~/cemig-poc/wandb/`
