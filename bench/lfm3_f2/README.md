# bench/lfm3_f2 — F2: injeção de conhecimento no LFM3-1.7B (LoRA / task-vector / K2V)

Pergunta do capitão: dá para "simular pré-treino" num conjunto de normas via
matriz de rank baixo (muitas iterações, LR baixo) e **aumentar o conhecimento
closed-book sem quebrar o conhecimento prévio**?

Régua do "antes": `bench/lfm3_f0` (closed-book 5,9%, oráculo 83,7%, ranking
65,8%, recusa-distrator 30,5%). Resultados comparativos: `RESULTS.md`.

## Desenho

- **Normas**: sorteio estratificado (cobertura no benchmark × tamanho do doc,
  pareado por estrato, seed 20251028, `select_norms.py`) →
  **tratadas NR-10/18/34, controle NR-13/31/33** (`data/norm_selection.json`).
- **Braços** (mesmos dados, mesmas normas):
  - **A** — LoRA no INSTRUCT (`LiquidAI/LFM3-1.7B`), target w1/w2/w3 (MLP;
    gate_proj é gate da ATENÇÃO no LFM3, fora do alvo). Assert: 72 lora_A em
    `*.feed_forward.*`, 0 em `*.self_attn.*`. Células r8 × r32, 12 épocas,
    LR 5e-5 cosine, checkpoint/época.
  - **B** — task vector: delta = instruct − base; mesmo LoRA sobre o BASE,
    merge, re-aplica delta (variantes `sum` e `prop`; fórmulas em
    `train/task_vector.py`).
  - **C** — método K2V (SeedScientist/K2V): corpus → grafo → QA fill-blank
    verificável (PT-BR) + checklist por questão (27B) → RL. **Implementação:
    TRL GRPOTrainer** (reward answer-gated: acerto da entidade mascarada +
    0.5×pass-rate do checklist julgado pelo 27B, só quando a resposta está
    certa). *Limitação conhecida*: o rollout vllm/sglang do verl NÃO suporta
    o LFM3 (`ValueError: Layer 0 (conv) does not dispatch through the
    Transformers attention interface`, vllm 0.32.0) — decisão registrada de
    manter o método com outra lib de RL.

## Geração sintética (funil front-loaded no 27B)

`synth/` — estágios com artefato auditável + **gate de validação por estágio**
(`validate_stage.py`: style×referência <2%, entropia prefixo-3 ≥6 bits,
top-prefixo <5%, near-dup <10%, rastreabilidade 100%, schema):

1. `s1_decompose.py` — 842 unidades de conhecimento com ref exata.
2. `s2_expand.py` — 3 personas × 2 cenários × {com_ref, sem_ref} por unidade
   (aberturas proibidas rotativas → anti-mesmice). 6.536 QAs.
3. `s3_socratic.py` — cadeias aplicação→por-quê→e-se→limite nas unidades
   densas (style rotulado POR DETECÇÃO; 3 iterações de funil até passar o
   gate — ver histórico no git).
4. `s4_critic.py` — 27B como crítico (fiel/autossuficiente/respondível):
   5.526 aprovados (~76%).
5. `s5_paraphrase.py` — reescritas multi-registro do texto cru.
6. `s6_mix_report.py` — dedup anticontaminação (Jaccard ≥0,5 vs benchmark:
   0 casos) + near-dup intra (≥0,8: 6) + **relatório de diversidade**
   (`data/synth/diversity_report.md`: entropia n=3 **11,0 bits** vs 7,7 do
   benchmark; cobertura por unidade ~94%; estilos 50/50).

Mistura de treino (`train/prep_train_data.py`): raw 49 + paráfrases 97 + QA
5.518 ("Pergunta:/Resposta:" texto corrido, estilo pré-treino), split 98/2.

## Avaliação

- Harness: `eval/gen_f2.py` (recorte 344 queries = 6 normas × 4 vocab;
  batched generate bs48 — bs1→bs48 dá ~26× em tok/s; medição em
  `bench_gen.py` na Spark) + `eval/score_f2.py` (régua honesta do F0) +
  `eval/export_parquet.py`.
- **`data/eval/` é o diretório canônico** (ver `data/eval/README.md`):
  1 parquet por checkpoint×célula com schema estendido de transparência
  (context_given/context_type/system_prompt/raw_output/gen_params) —
  baseline F0 re-exportado retroativamente no mesmo schema.
- Guardas: oráculo/ranking/distrator (eps selecionadas) + probe de capacidade
  geral (`probe/run_probe.py`, réguas OOD congeladas da POC).
- Decisão registrada: curva closed-book A-r8 satura ~ep6 → demais runs
  avaliados nas épocas 2,4,6,8,12.

## Infra / reprodução

- **Spark** (`walcyrios@spark-b431` / tailscale `100.79.169.101`,
  `~/cemig-poc/lfm3_f2/`): venv `.venv` (transformers 5.21, peft, trl, wandb).
  Treinos: `run_all_trains.sh` (sequenciais; **r32 com bs8 derrubou a máquina
  por OOM — usar bs4/ga8**). Grade: `run_eval_grid*.sh`.
- **H100** (`aluno_walcy@10.100.0.111`, container `cemig_mobile_walcy`,
  GPU 2 apenas, artefatos em `/raid/aluno_walcy/cemig_mobile` =
  `/workspace/cemig`): braço C + task vector (worker dedicado).
- **wandb**: `WANDB_MODE=offline`, runs em `~/cemig-poc/lfm3_f2/wandb`
  (Spark) e `/workspace/cemig/wandb` (H100), projeto `cemig-lfm3-f2`, nomes
  `lfm3-f2-{A,B,C}-...`. Sync posterior: `wandb sync <dir do run>`.
- Julgamento local (WSL): `classifier/.venv`, juiz `10.100.0.111:8005`.
- K2V: `k2v/setup_k2v.sh` (clone pinado + `patches.diff`: thinking-OFF no
  cliente e saída PT-BR nos templates). Dados: `k2v/data/k2v_qa_checklist.jsonl`
  (1.222 QAs fill-blank com checklist).

## NÃO fazer

- Benchmark NUNCA vira treino (dedup anticontaminação ativo).
- Não regenerar as réguas OOD congeladas (`probe/ood_*.json`).
- Não rodar 2 treinos simultâneos na Spark (memória unificada; OOM derruba o
  host — já aconteceu).
