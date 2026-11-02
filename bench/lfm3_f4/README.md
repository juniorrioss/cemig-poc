# bench/lfm3_f4 — F4: tool-calling multiturno sobre o campeão data2x-ep4

Fase 2 do plano aprovado: o PRÓPRIO MODELO decide chamar a busca, sintetiza
do retorno, recusa quando o contexto está errado e faz **MULTIHOP** (re-chama
com outro argumento quando a 1ª busca não basta — transforma a alucinação de
24,9% em segunda tentativa de busca).

Checkpoint-base: **merge do LoRA data2x-ep4 (F3b ckpt-1396) no LFM3-1.7B**
(`merge_base.py`; md5 agregado em `data/merged_base_md5.json`). O LoRA novo
(r32 MLP, receita F3) treina SOBRE esse merged.

## PASSO 0 — formato nativo (provado em `verify_format.py`)

- Tool-calling do LFM3 é **XML `<tool_call>`** emitido pelo próprio
  `apply_chat_template` (`tools=...`); render de referência em
  `data/passo0_render.txt`. Zero regex na RENDERIZAÇÃO (receita tools_v1:
  27B gera CONTEÚDO; formato é do template).
- `tool_parser.py` é o inverso determinístico do render — round-trip provado
  (acentos, vírgula decimal, `null`, multi-linha, 2 calls por conversa).
- **Paridade BAKED byte a byte**: tools embutidas no system
  (`make_baked_system_message`) == `system simples + tools=[...]` — é assim
  que o TRL treina (sem `tools=`).
- `{% generation %}` cobre DECISÃO + ARGUMENTOS + SÍNTESE e **exclui o turno
  `tool`** → `assistant_only_loss` nativo.
- `enable_thinking=False` SEMPRE (treino e inferência; achado do GRPO).

## Dataset (`gen_dialogs_f4.py`, 3 estágios)

Reuso: assignments/targets/units do **data2x** (F3b; contexts do retriever
REAL v4, k variável 3/5/8, rótulos gold/partial/distractor por containment),
templates de recusa da F3, cliente 27B da F2. Argumento da 1ª chamada =
**fala crua** (evidência tools_v1: reescrita do modelo PIORA a busca);
`nr` = detecção determinística de menção explícita.

| família | alvo | ensina |
|---|--:|---|
| chamada_simples | 1600 | user → tool_call(fala crua) → tool(top-k real) → síntese 27B citando item |
| recusa | 1050 (~22%) | tool retorna contexto errado (erros GENUÍNOS do retriever) → recusa honesta |
| troca | ~600 | 2º turno muda de NR → nova chamada |
| reuso | 700 | follow-up respondível com o contexto presente → responde SEM chamar |
| multihop | ≤500 | hop1 parcial → 2ª chamada com argumento DIFERENTE → síntese das 2 evidências |
| sem_ferramenta | 250 | saudação/fora de escopo → responde sem chamar |

Gates mecânicos por exemplo: round-trip do render; dedup intra por pergunta;
anticontaminação Jaccard≥0,5 vs as 1536 do benchmark; multihop exige
2º argumento ≠ 1º **E** ouro de verdade no top-5 do retriever real (a
consulta-2 do 27B é validada EXECUTANDO a busca; pré-filtro: só partials cujo
ouro existe no corpus — ~24% não têm). Gate final: `validate_dataset_f4.py`.

Partição EVAL **antes** do treino, disjunta por qa_id (240 gold, 60 absent,
80 partial) → suites `data/eval/suite_{decision,reuse,switch,refusal,multihop}.jsonl`.

## Treino (H100, container cemig_mobile_walcy, GPU 2)

`train/train_f4.py` = receita F3b com base no MERGED e max_len 4096
(diálogos de 2 ciclos de tool; p99 medido no validate). LoRA r32 alpha 64
MLP w1/w2/w3 — ASSERT 72 lora_A, 0 attn/gate/conv. 3 épocas ckpt/época,
wandb `lfm3-1.7b-cemig` run `lfm3-f4-tools-r32`. `train/run_train_f4.sh`
(desacoplado + polling barato).

## Avaliação

1. **Estática (regressão)**: 4 células × 1536 (`gen_f2.py` + `score_f4.py`
   shim da régua honesta) — ranking ≥75%, oráculo ≥83%, closed-book,
   distrator; probe de capacidade (`lfm3_f2/probe`). As células estáticas
   usam o SYSTEM do F0 SEM tools (o modo sem ferramenta não pode regredir).
2. **Tool-calling** (`eval/run_dialog_harness.py`): simulador de tool =
   retriever v4 REAL respondendo ao vivo (não mock). Métricas
   (`eval/score_tools.py`): F1 de decisão, % sintaxe XML válida, reuso
   correto, troca de assunto, recusa correta/indevida e MULTIHOP em 4
   estágios (2ª chamada feita → argumento difere → ouro no hop2 → síntese usa
   as DUAS evidências, juiz 27B).
3. Parquets schema estendido com `context_type` ∈ {tool_loop, multihop}
   (`eval/export_parquet_f4.py`) em `data/eval/`.

Critérios de sucesso (brief): decisão/sintaxe ≥90%, multihop ≥60% (meta
modesta — família nova), regressão estática ≤2pp.

## Reprodução

```
make verify      # PASSO 0
make merge       # checkpoint-base (publica md5)
make assemble gen27b build validate   # dataset
# treino: rsync p/ H100 + bash train/run_train_f4.sh
make harness ADAPTER=... TAG=f4-r32-ep2
make score-tools TAG=f4-r32-ep2
make parquet TAG=f4-r32-ep2
```

Resultados: `RESULTS.md` (comparativo data2x-ep4 sem tools vs F4 com tools).
