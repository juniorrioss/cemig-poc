# Worker H100 — braço C (K2V/GRPO) + task vector (F2 do LFM3-1.7B)

Trabalho da GPU 2 da H100 (`dgx-H100-01`, `aluno_walcy@10.100.0.111`), TODO
dentro do container `cemig_mobile_walcy` (pytorch 2.9.0+cu128; GPU 2 do nó =
`cuda:0` interno; artefatos em `/raid/aluno_walcy/cemig_mobile` montado como
`/workspace/cemig`). O worker da Spark (task lfm3-f2-inject) é dono da grade
de avaliação; este worker produz checkpoints e devolve via scp.

## Arquivos

| Arquivo | O quê |
|---|---|
| `train_grpo_k2v_h100.py` | braço C: K2V via TRL GRPOTrainer com LoRA ff (72 matrizes), judge paralelo no 27B local, thinking-OFF + few-shot (ver abaixo) |
| `task_vector.py` | cópia fiel do script da Spark (fonte: worker lfm3-f2-inject); roda CPU/fp32 |
| `run_task_vectors.sh` | lança as 4 variantes sum/prop × r8/r32 |
| `TASK_VECTOR_README.md` | fórmulas exatas de cada variante |

## Sharp edges do container (não re-descobrir)

1. **Download HF quebrado por default**: `httpx2` do container falha com
   `TypeError: process() takes no keyword arguments` ao descomprimir.
   Fix: `pip install -U zstandard brotli` dentro do container.
2. **LFM3-1.7B é modelo de RACIOCÍNIO (thinking ON por default)**: o
   chat_template fecha com `<|im_start|>assistant\n<think>`; com ON ele gasta
   TODO o budget de completion pensando em inglês, nunca emite EOS nem chega
   a 'Resposta:' ⇒ **reward 0 em todos os steps** (clipped_ratio=1). Fix:
   `chat_template_kwargs={"enable_thinking": False}` no `GRPOConfig`
   (suportado no TRL 1.17; o template injeta `<think></think>` vazio).
3. **Thinking-OFF sozinho NÃO basta**: acerto base ~0-4% nas fill-blank K2V
   (o 1.7B diverge/divaga) ⇒ sem variância de reward ⇒ sem gradiente GRPO.
   Com 1 exemplo few-shot + a dica dos `adjacent_nodes` do grafo (campo que
   já existe no dataset K2V) o acerto base medido é ~10% @ temp 1.0 ⇒ P(≥1
   acerto no grupo de 8) ≈ 57%. max_completion 512, temperature 1.0.
4. **wandb**: `.netrc` da Spark copiado p/ `/root/.netrc` do container ⇒
   online no projeto `lfm3-1.7b-cemig` (forge.coreweave.com, user walcyrios).
5. **GRPOTrainer + LoRA**: `peft_config=` direto no construtor; asserts
   idênticos aos braços A/B (72 lora_A em `feed_forward.{w1,w2,w3}`, 0 em
   `self_attn` — `gate_proj` no LFM3 é gate da ATENÇÃO).
6. Processos longos: `docker exec -d` + nohup, log+PID em
   `/workspace/cemig/checkpoints/<run>/`; monitorar por `tail` via ssh.

## Resultados do smoke (50 steps)

- v1 (prompt original da Spark): reward 0/50 steps — causa: thinking ON (edge 2).
- v2 (thinking-OFF): completions terminam mas reward 0 — causa: acerto base ~0% (edge 3).
- v3 (few-shot + adjacent nodes): **PASSOU** — reward médio 0.066, steps com
  reward até 0.82, grupos com 4/8 acertos no dump; judge 27B parse ok
  (pass-rate 0.83 na amostra de calibração); ~11.5 s/step; checkpoint-50
  salvo e recarregável (PeftModel 72 lora_A).

## Corridas

- Smoke: wandb run `lfm3-f2-C-k2v-smoke` (gcuhdbfd).
- Completa: `lfm3-f2-C-k2v-grpo` (wandb ixu0jg3k), 400 steps, lr 1e-6,
  batch 8 × 8 gerações, 82 min (12.3 s/step).
  **Curva de reward por quartil de steps: 0.080 → 0.142 → 0.116 → 0.084** —
  sobe até ~step 200 e DECAI depois (possível over-optimização/lr decay);
  por isso foram enviados p/ a grade `final` (step 400) E `checkpoint-150`/
  `checkpoint-200` (região do pico) — a grade da Spark decide qual vale.
  Entregues em `~/cemig-poc/lfm3_f2/runs/lfm3-f2-C-k2v-grpo/` (+ premises e
  reward_dump.jsonl com 1 amostra de completion por step).

## Task vector (item 3 da fila)

4 variantes construídas na H100 (CPU), sanity-check de geração ok, enviadas
para a Spark em `~/cemig-poc/lfm3_f2/runs/taskvec-{sum,prop}-{r8,r32}/`.
Escalas da `prop` ficaram ≈0.999 (o LoRA quase não mudou a magnitude média
dos w1/w2/w3) ⇒ espera-se prop ≈ sum na avaliação. Fórmulas:
`TASK_VECTOR_README.md`.

## Células closed_book rodadas na H100 (Spark saturada, cláusula do brief)

Com a GPU 2 livre após a corrida C e a Spark a 94%, as gerações closed_book
dos 6 artefatos novos rodaram aqui (`run_eval_h100.sh`, harness `gen_f2.py`
da Spark INALTERADO, batch 48, max_new 3072, normas 10/18/34/13/31/33,
344 queries/célula, ~7-8 min/célula):
taskvec-{sum,prop}-{r8,r32} (modelo HF direto) + C-k2v `final` e
`checkpoint-200` (adapter on-the-fly sobre o instruct). Saídas entregues em
`~/cemig-poc/lfm3_f2/data/gen_f2-{taskvec-*,C-k2v-*}_closed_book.jsonl` na
Spark — o julgamento (27B) é do worker da Spark.
