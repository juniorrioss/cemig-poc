# bench/lfm3_f3b — sweep one-factor-at-a-time sobre a receita F3

Ordem do capitão: "variar e ver o que domina" — dataset (dose de recusa,
tamanho), rank, alvos LoRA, LR e âncora anti-esquecimento, mantendo TODO o
resto igual à receita F3 (dataset 6.972, r32, LoRA MLP w1/w2/w3, lr 1e-4
cosine, bs8×ga4, max_len 6144, assistant_only_loss, 4 épocas ckpt/época).
Objetivo: descobrir qual eixo domina e se dá para cruzar os alvos
(ranking ≥78%, alucinação <10%) reduzindo/mantendo o esquecimento
(probe ≥3,18).

## Protocolo

- **Avaliação rápida padronizada** (ranqueia células): ranking + distrator no
  RECORTE das 6 normas do F2 (10/18/34 tratadas + 13/31/33 controle = 344
  queries/célula) via `eval/run_eval_f3b.sh` (gen_f2.py da F2 com `--norms`),
  julgamento `eval/score_f3b.py` (shim do score_f2 = régua honesta do F0),
  probe de capacidade (130 itens, `bench/lfm3_f2/probe/run_probe.py`) nos
  checkpoints relevantes. Referências F0/F3 recomputadas NO MESMO recorte
  pelo `analyze_f3b.py` (não comparar com os números de 1536 do RESULTS da F3).
- **Campeão final**: bateria completa 1536 nas 4 células
  (`bench/lfm3_f3/eval/run_eval_grid_f3.sh`) + parquets schema estendido.
- Critério do campeão: max(ranking) sujeito a alucinação ≤ F3 (21,3% em 1536),
  recusa-indevida não pior, probe ≥ 3,18.

## Eixos / células

| eixo | célula | o que muda | dataset |
|---|---|---|---|
| 1 dose de recusa | `recusa35` | absent 20%→35%, distratores DUROS (same-NR seção errada; top-8 real + hard-negatives do índice v4 com containment<0,30) | `build_variants.py --variant recusa35` |
| 2 tamanho | `data2x` | ~2x QAs (funil NRs novas 4/7/9/11/15/19/20/29/30/32/36/37 — fora das usadas E fora do controle 13/31/33) | `synth/run_funnel_f3b.py` + retrieve + `--variant data2x` |
| 3 rank | `rank16`, `rank64` | r16 / r64 (alpha=2r) | F3 congelado |
| 4 alvos LoRA | `mlpattn` | MLP + q/k/v/o_proj (116 lora_A; NUNCA self_attn.gate_proj — gate sigmoide da atenção; assert aborta) | F3 congelado |
| 5 LR | `lr5e5` | lr 1e-4 → 5e-5 | F3 congelado |
| 6 âncora | `anchor25` | âncora (closed+geral) 10%→25% com pares gerais PT-BR do 27B (`gen_general.py`, anticontaminação Jaccard≥0,5 vs probe CONGELADO) | `--variant anchor25` |

## Infra

- Treinos: fila na H100 (`train/queue_runner.sh` consome
  `/workspace/cemig/f3b/queue/*.job`; container `cemig_mobile_walcy`, GPU 2).
  Motor: `train/train_f3b.py` (generalização do train_f3 com --targets/--lr;
  asserts de módulos obrigatórios: mlp=72, mlp_attn=116, 0 gate_proj/conv).
- Geração de avaliação: Spark (`~/cemig-poc/lfm3_f2`) e H100 em paralelo;
  adapters H100→local→Spark (H100→Spark não roteia).
- wandb `lfm3-1.7b-cemig`, runs `lfm3-f3b-<eixo>-<valor>`.

## Resultados

Ver `RESULTS.md` + `data/dominance.json` (tabela de dominância).
