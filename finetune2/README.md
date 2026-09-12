# finetune2 — Treino de síntese do LFM2.5-2.6B (SFT + DPO) — RESULTADO NEGATIVO HONESTO

Responde à intenção do capitão: elevar a **conversão chunk-certo → resposta-aprovada** do
sintetizador (2.6B thinking-OFF: ~35-37% no `bench/sintese2/`) via **treino**, já que a engenharia
de prompt **saturou** no 2.6B. Desenho aprovado: **FASE A (SFT leve)** ensina o formato; **FASE B
(DPO)** com `rejected` = saída real ruim do 2.6B e `chosen` = edição mínima do 27B.

## TL;DR — veredito

**Nem o SFT nem o SFT+DPO melhoraram; ambos REGRIDIRAM no holdout e o painel os MATA.** A régua de
regressão (qualquer camada caindo além do ruído mata o candidato) reprovou as duas variantes.

| Variante (holdout 151, mesmo juiz 27B) | gate% | conv% | global | fac | fid | cit | pt | toks |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| **base f16** (não treinado, alta precisão) | **23.2** | **36.4** | **2.513** | 2.23 | 2.19 | 2.17 | 4.27 | 184 |
| base Q4_0 (referência de embarque) | 21.2 | 36.4 | 2.495 | 2.24 | 2.20 | 2.09 | 4.23 | 174 |
| dpo f16 (treinado, alta precisão) | 11.9 | 22.1 | 2.132 | 1.73 | 1.82 | 1.74 | 4.24 | 133 |
| dpo Q4_0 (treinado + requantizado) | 9.3 | 18.2 | 2.077 | 1.78 | 1.76 | 1.61 | 4.02 | 114 |
| sft Q4_0 | 7.9 | 13.0 | 2.019 | 1.68 | 1.68 | 1.56 | 4.11 | 113 |

**Recomendação de embarque: NÃO embarcar as variantes treinadas. Manter o sintetizador atual**
(a alavanca de qualidade segue sendo retrieval + destravar o thinking-OFF do 2.6B no engine
nativo, conforme `bench/sintese2/` e `android/README_CONSOLIDACAO.md`). O embarque real depende do
engine-upgrade que corre em paralelo — aqui **não** integramos no app (por ordem do brief).

## Por que regrediu (diagnóstico)

1. **É o TREINO, não a quantização** (célula pedida pelo capitão, ponto 3 do inbox). A regressão
   já aparece em **alta precisão**: base f16 **23.2%** → dpo f16 **11.9%** de gate (−11.3 p.p.). A
   requantização 4-bit adiciona só uma queda pequena e simétrica dos dois lados: base f16→Q4_0
   perde ~2 p.p. de gate (perda de quantização da própria Liquid); dpo f16→Q4_0 perde ~2.6 p.p.
   Ou seja, **o dano do fine-tuning (≈11 p.p.) é ~5× maior que o da quantização (≈2 p.p.)** e o
   treinado-requantizado **não** supera o base Q4_0. Isso responde à dúvida central: o ganho não
   sobrevive porque **não houve ganho** — o treino piorou o modelo antes de qualquer 4-bit.
2. **Over-fitting de um base forte a uma distribuição sintética mais estreita.** O 2.6B base já é um
   bom sintetizador RAG; 1 época de SFT sobre 1500 pares sintéticos + DPO sobre 122 pares
   **encolheu a resposta** (184→133→114 tokens) e passou a **escolher o item errado do contexto**.
   Leitura humana das 10 primeiras (governança obrigatória, `bench/judge.py`): em qa-001 o base
   cita corretamente NR-10 10.2.8.2 (5/5) e o treinado passa a citar 10.3.1, item errado do mesmo
   chunk (2/1). Padrão repetido em 17 itens onde o base passava e o treinado falha, todos com
   retrieval-hit. **PT-BR não caiu** (4.2-4.3 em todas as células) — o dano é factual/citação.
3. **DPO > SFT, mas ambos abaixo do base.** O DPO recupera parte do que o SFT perdeu
   (gate 7.9→9.3% Q4_0; 22.1% conv no f16), coerente com as margens de recompensa subindo no treino
   (0.01→0.05, acc 0.575→1.0), mas **parte de um SFT já degradado** e não volta ao base.

## Os 122 pares DPO (honestidade — ponto 4 do inbox)

Alvo era 300-500; ficamos em **122**. Motivo registrado: dos 800 candidatos do pool limpo, só
**200** eram `rejected` válidos (retrieval-hit + reprovado no gate), e o **filtro de minimalidade**
(difflib ratio ≥ 0.6) descartou **70 edições-demais + 7 sem-mudança** — exatamente o que o capitão
pediu para garantir "edição mínima" de verdade. Os 122 têm ratio médio 0.89 (mediana 0.94), todos
≥ 0.6. **122 é pouco para a FASE B ter efeito grande**, e o resultado confirma: o DPO move o
ponteiro só um pouco acima do SFT — não o suficiente para cruzar o base. Não vendemos isso como
ganho; é ruído favorável dentro de um quadro de regressão.

## Camadas OOD (o painel completo do brief)

| Variante | OOD-estrutural gate% (NR 33/16/26) | geral qualidade | geral PT | comport. segurança |
| :-- | --: | --: | --: | --: |
| base | 22.9 | 3.93 | 4.85 | 4.00 |
| sft | 11.4 | 3.73 | 4.83 | 3.93 |
| dpo | 11.4 | 3.81 | 4.78 | 4.20 |

- **OOD estrutural** (normas cujos chunks nunca entraram no treino — MURALHA 2): treinados caem
  pela metade, confirmando que a degradação **generaliza** (não é só o holdout).
- **Capacidade geral** (100 itens FIXOS congelados): queda leve de qualidade (3.93→3.73/3.81); PT
  praticamente intacto. Não é colapso catastrófico, mas soma à regressão.
- **Comportamental** (30 itens FIXOS): estável; o DPO até melhora um tico a segurança (4.0→4.2) —
  único sinal positivo, mas irrelevante frente à queda factual.

## As TRÊS MURALHAS anti-contaminação (aplicadas e auditáveis)

1. **Holdout intocável** (151 `qa_pairs_v2` + 20 smoke): só tocado no `eval_panel.py` (juízo final).
   Nunca em geração/treino/ajuste. `common.py:load_holdout` / `holdout_gold_chunk_ids`.
2. **Split por NR**: NR-33/16/26 **reservadas** — 0 chunks delas no SFT (`common.RESERVED_NRS`,
   verificado: `reserved NRs present: 0` no `sft.jsonl`); viram o painel OOD estrutural.
3. **Filtro lexical automático**: toda fala gerada passa por Jaccard de 3-grams vs TODO o holdout;
   ≥ 0.4 é descartado e **registrado com procedência** (`data/sft_lexical_discards.json`,
   `data/dpo_questions_lexical_discards.json`). Resultado do SFT: **0 descartes**, max Jaccard
   observado **0.10** (média 0.02) — a geração não copiou o holdout.

> **Nota de integridade (importante):** as 604 respostas do 2.6B já julgadas em
> `bench/sintese2/` e `bench/judge151/` foram TODAS geradas sobre o holdout 151. Usá-las como
> fonte de DPO violaria a MURALHA 1 e contaminaria a avaliação final nas mesmas 151. Por isso o
> `gen_dpo.py` **gera respostas frescas do 2.6B sobre um pool CLEAN** (`data/dpo_questions.jsonl`,
> 800 perguntas de chunks não-holdout/não-reservados, Jaccard < 0.06 vs holdout), preservando
> exatamente a metodologia DPO do brief (rejected = saída real ruim; chosen = edição mínima do 27B).

## Conjunto OOD FIXO (congelado — ponto do capitão sobre a camada 3)

Sem benchmark externo (ENEM/OAB difícil de obter), montamos régua própria pequena e FIXA:
`data/ood_general.json` (**100 itens**, 4 categorias: instruções genéricas, resumo/reescrita, QA
factual, conversa banal) gerada 1x pelo 27B e **CONGELADA** (guarda anti-regeneração em
`gen_ood.py`: aborta se já existir, salvo `--force` com aviso) + `data/ood_behavioral.json` (**30
prompts** escritos à mão, determinísticos). É régua interna de regressão, não benchmark acadêmico.

## Pipeline e reprodução

Interpretadores (**recriáveis** — são gitignored): `classifier/.venv` (uv, py3.12, sklearn 1.9.1 —
busca/juiz/geração) e `.venv-train` (uv, py3.10, torch cu128 + trl/peft — treino). Ver comandos de
reccriação em `classifier/README.md` e AGENTS.md.

```bash
make -C finetune2 data          # gen_sft + gen_dpo_questions + gen_dpo + gen_ood (congelado)
make -C finetune2 train         # FASE A (SFT r=16, 1ep, lr1e-4) + FASE B (DPO beta0.1, lr5e-6, 1ep)
make -C finetune2 eval          # sobe servidores GGUF base/sft/dpo + painel 4 camadas + consolida
# ou orquestração completa:
./finetune2/run_all.sh all
```

Artefatos de treino ficam em `~/train-poc/` (FORA do repo): `sft_adapter/`, `dpo_adapter/`
(LoRA 12 MB cada), `merged_{sft,dpo}/*-Q4_0.gguf` (GGUF Q4_0 1.59 GB), `base-f16.gguf` e
`merged_dpo_f16/dpo-f16.gguf` (célula de requantização).

### Arquivos

```
finetune2/
  common.py            # 3 muralhas + vLLM com retry + holdout + chunks (fonte única)
  gen_sft.py           # FASE A: SFT por inversão de chunks (1 correto + 1 distrator)
  gen_dpo_questions.py # pool CLEAN de perguntas (NÃO holdout) p/ a FASE B
  gen_dpo.py           # FASE B: 2.6B real -> juiz 27B -> rejected + edição mínima 27B (chosen)
  gen_ood.py           # conjunto FIXO capacidade geral + comportamental (CONGELADO)
  train.py             # FASE A (SFT) + FASE B (DPO) + export merged/GGUF Q4 (QLoRA fallback OOM)
  eval_panel.py        # painel 4 camadas base/sft/dpo (PARALELO 8w) + requant + consolidação
  retrieval_local.py   # retriever v3 (RRF 3-sinais) robusto a caches gitignored (usa bin_*_rank)
  run_all.sh / Makefile
  data/
    sft.jsonl                       # 1500 pares SFT (0 contaminação, 33 NRs, procedência/linha)
    sft_lexical_discards.json       # descartes da MURALHA 3 (0)
    dpo_questions.jsonl             # 800 perguntas CLEAN (fonte DPO)
    dpo_raw.jsonl                   # 800 respostas cruas do 2.6B + juízo (procedência)
    dpo.jsonl                       # 122 pares DPO {prompt,chosen,rejected,meta(ratio,eixos)}
    ood_general.json / ood_behavioral.json   # régua FIXA congelada (100 + 30)
    panel_{base,base_f16,sft,dpo,dpo_f16}.json  # respostas+juízo+métricas por variante/camada
    panel_consolidation.json        # tabela + veredito de regressão (MATA sft e dpo)
```

## Higiene e notas de execução

- **Paralelização** (ordem do capitão, ponto 1): `eval_panel.py` roda 8 workers (juiz vLLM) +
  4 slots do llama-server (`--parallel 4`); cada variante caiu de ~40 min (serial) para **~2-3 min**.
- **vLLM com retry** (VPN oscila): backoff exponencial em `common.call_vllm`.
- **Checkpoint incremental** em toda geração e no painel (reaproveita itens já feitos).
- **Requantização**: `llama-quantize` foi buildado à parte em `~/llama.cpp/build-quant` (o
  build-cuda não linka por incompatibilidade brew-gcc/glibc); export usa Q4_0 (o Q4_K_M oficial da
  Liquid gera lixo no engine `434ddbb`, ver AGENTS.md).
- **Incidente de infraestrutura**: o worktree irmão que hospedava venvs/índices/caches densos
  gemma768 (gitignored) foi apagado no meio da tarefa. Reconstruí a partir de artefatos COMMITADOS:
  `classic_winner.pkl` (via `train_classic.py`, logreg top-2 69% — decisão já registrada em
  `classifier/README.md`, não re-decidida), `index_hf_36nr_exp.db` (via `build_expanded_index.py`)
  e os sinais densos via os caches commitados `retrieval3/results/bin_{text,exponly}_rank.json`
  (formato on-device DVEC1, a paridade real do app). O painel confirma R@2 52.3% no holdout,
  idêntico à entrega v3 — retrieval intacto.
- **Teste de aparelho (ponto 5 do inbox)**: condicionado a "se a variante treinada vencer".
  **Não venceu** → não medimos no S24+ (seria medir latência de um modelo pior). O caminho de
  aparelho segue o do `bench/device-liquid/` para o embarcado atual.

## Governança

O juiz LLM é **auxiliar**. A leitura humana das 10 primeiras (mandatória, `bench/judge.py`) foi
feita e **confirma o veredito automático**: nas perguntas com retrieval-hit o base converte melhor;
o treinado degrada acerto/citação mantendo o PT — regressão real, não artefato do juiz.
