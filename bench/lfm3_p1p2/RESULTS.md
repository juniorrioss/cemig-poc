# RESULTS — P1 (filtro de aprendibilidade) + P2 (mini-RLVR de groundedness) sobre o f4-r32-ep2

Ordem do capitão, depois da revisão do paper Rufus-Air (arXiv 2609.29421): **P1** filtra o
treino pelo que a política atual ainda NÃO domina; **P2** é um RLVR curto com reward binário
verificável por regra, sem juiz no loop, sobre o `f4-r32-ep2`. Avaliação no recorte 344
(normas 10/18/34/13/31/33), 4 células padrão + e2e do app (op_d50) + probe geral, juiz 27B
(régua honesta). Tabela-mestra: `data/consolidation.json`. Relatório de bandas: `data/p1_report.json`.

## Resumo

- **P1 entregue como ferramenta** (`p1_pipeline.py`, 7 subcomandos reentrantes). Num universo
  de 11.313 itens (pool RLVR completo + 5.776 do SFT data2x/F4), **49,2% são descartados**:
  39,5% fáceis (acc ≥ 0,8, n=4) e 9,7% mal especificados (acc = 0 e o 27B também falha). Ficam
  45,0% na banda produtiva e 5,8% difíceis que o 27B resolve.
- **P2 melhora o modo app e o ranking. A alucinação no distrator NÃO cai: sobe.** No melhor
  ponto (`v2s400`): e2e do app **68,3 → 73,8** (pareado 30/11, p = 0,004), cita item-ouro
  **69,0 → 74,0%**, alucinação no app 14,5 → 12,8, ranking **77,6 → 83,7** (p = 0,004),
  recusa indevida 3,2 → 0,9, oráculo 89,0 → 91,3, probe 3,38 → 3,37 (dentro de 1 p.p.).
  Mas o distrator estático (4 passagens erradas, system F0 sem ferramenta) piora: alucinação
  **22,4 → 28,5** e recusa correta 61,0 → 46,8 (pareado 1/50).
- **Metas do brief**: citação de item > 73% ✓ (74,0 no app); alucinação no distrator < 24,9% ✗
  (28,5). Ranking, oráculo e probe sem regressão ✓, mas **o distrator regride bem além da
  tolerância de 1 p.p.** Na régua da ordem, o P2 **não passa** como substituto do f4-r32-ep2.
  O ganho real fica no modo app.
- **Concisão**: tokens médios de resposta 86,2 → 91,8 (+6,5%). O reward binário não encurta;
  fica para o P3, com termo explícito.

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
- **Cobertura parcial (honesta)**: o rollout do SFT foi cortado pelas janelas de treino P2.
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

**Runs** (wandb `lfm3-1.7b-cemig`):

| run | pool | contrato | wandb |
|---|---|---|---|
| `lfm3-p2-rlvr` (s200/s400) | RLVR produtiva+difícil, recusa ×2 (19%) | v1 | `c5cepsnk` |
| `lfm3-p2-rlvr-bal` (bal200/bal400) | + todas as recusas fáceis (36%) | v1 | — |
| `lfm3-p2-rlvr-v2` (v2s200/300/400) | âncoras de recusa até 30% | **v2** | `5h5m9jt4` |
| `lfm3-p2-rlvr-v3` | v2 + itens data2x no formato estático | v2 | `o4kcs0vz`* |

Reward de treino (média por quartil, run v2): answer 0,65 → 0,76; refuse 0,87 → 0,94.

### Avaliação (recorte 344, juiz 27B)

| tag | ranking | rec-indev. | **distrator aluc.** | distr. recusa | oráculo | closed | **e2e app** | **cita item (app)** | aluc. app | probe | tok |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **f4-r32-ep2 (base)** | 77,6 | 3,2 | **22,4** | 61,0 | 89,0 | 7,8 | 68,3 | 69,0 | 14,5 | 3,38 | 86,2 |
| s200 | 84,6 | 0,9 | 28,8 | 50,6 | 91,9 | 8,4 | 72,7 | 70,7 | 14,5 | 3,34 | 89,1 |
| s400 | 83,7 | 1,5 | 26,2 | 48,0 | 91,6 | 8,1 | 73,3 | 73,0 | **11,6** | 3,32 | 91,2 |
| bal200 | 84,3 | 1,5 | 25,9 | 50,3 | 91,3 | 8,7 | 70,9 | 70,7 | 14,8 | 3,34 | 87,2 |
| bal400 | 83,4 | 1,2 | 28,2 | 45,1 | 91,0 | 7,0 | 73,5 | 72,7 | 12,5 | 3,31 | 90,1 |
| v2s200 | 84,0 | 0,9 | 25,3 | 51,5 | 90,1 | 8,4 | 72,7 | 71,0 | 13,1 | 3,33 | 87,7 |
| v2s300 | **84,9** | 0,9 | 25,9 | 48,8 | 90,7 | 5,8 | 70,9 | 72,0 | 15,4 | 3,29 | 90,2 |
| **v2s400** | 83,7 | **0,9** | 28,5 | 46,8 | 91,3 | 7,0 | **73,8** | **74,0** | 12,8 | **3,37** | 91,8 |

Pareado vs base (aprovação; teste do sinal): e2e v2s400 30/11 (p = 0,004), s400 29/12 (0,012);
ranking s400 36/15 (0,005), v2s400 35/14 (0,004); oráculo +1 a +3 sem significância (exceto s200,
p = 0,04); distrator **1/50 (p < 1e-6)** em v2s400. Alucinação no distrator pior/melhor:
v2s400 43/22 (p = 0,013); s400 35/22 (p = 0,11).

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
desaprende a recusar. O v3 (pool com itens data2x no formato estático, recusa ~35%) testa
essa hipótese.

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
