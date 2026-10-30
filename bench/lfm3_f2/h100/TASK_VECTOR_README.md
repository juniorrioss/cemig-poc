# Task vector F2 (experimento do capitão) — executado na H100 (GPU 2)

Pipeline: o LoRA do braço B (treinado SOBRE o `LiquidAI/LFM3-1.7B-Base`, CLM de
injeção de conhecimento, módulos `feed_forward.{w1,w2,w3}`) é mergeado no base
e o **task vector do post-training da Liquid** é re-aplicado por cima.

```
delta        = W_instruct − W_base          (por tensor; shapes verificados 1:1)
W_trained    = merge_and_unload(base, LoRA) (base_treinado)
```

## Variantes (fórmula exata por tensor `t`)

| Variante | Fórmula | Racional |
|---|---|---|
| `sum`  | `W_final_t = W_trained_t + delta_t` | Soma direta. Nos tensores NÃO tocados pelo LoRA, `W_trained_t = W_base_t` ⇒ `W_final_t = W_instruct_t` (instruct puro). Nos `w1/w2/w3`: instruct + contribuição LoRA (`B·A·α/r`). |
| `prop` | tensores treinados: `s_t = mean(|W_base_t|) / mean(|W_trained_t|)`; `W_final_t = W_trained_t + s_t·delta_t`. Não treinados: `s_t = 1` (igual a `sum`). | Ideia literal do capitão ("proporção da média"): re-escala o delta para compensar a mudança de magnitude média que o merge do LoRA introduziu no base. |

Cada saída é um modelo HF completo em bf16 com o esqueleto do **instruct**
(config + chat template do instruct), + `task_vector_stats.json` com as escalas
`s_t` por tensor e a contagem de tensores alterados.

## Execução

- Script: `run_task_vectors.sh` (dentro do container `cemig_mobile_walcy`,
  artefatos em `/workspace/cemig/checkpoints/taskvec-{sum,prop}-{r8,r32}/`).
- Adapters de origem: Spark `~/cemig-poc/lfm3_f2/runs/lfm3-f2-B-base-{r8,r32}/final`
  (trazidos por tar-over-ssh para `/workspace/cemig/checkpoints/spark_adapters/`).
- `task_vector.py` é o script da Spark INALTERADO (commitado em
  `bench/lfm3_f2/task_vector.py` no branch fm/lfm3-f2-inject); roda em CPU/fp32.
- Retorno: variantes empacotadas e enviadas de volta para a Spark em
  `~/cemig-poc/lfm3_f2/runs/<nome>/` para a grade de avaliação (worker da Spark).
