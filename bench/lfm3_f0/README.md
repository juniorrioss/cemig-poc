# bench/lfm3_f0 — Baseline F0: LFM3-1.7B bf16 no benchmark_mobile

Régua de comparação de TODAS as fases seguintes da nova etapa da POC (LoRA de
contexto+tools e injeção de conhecimento). A célula **closed_book é a régua do
"antes"** da fase de injeção de conhecimento. Pedido literal do capitão:
*"rodar exatamente o modelo as is no benchmark e ver o resultado, ainda sem
qualquer etapa de quantização e afins, rodando em gpu mesmo."*

## O que é medido

- **Dataset**: `UFGCEMIGONA/benchmark_mobile`, config `normas_regulamentadoras`,
  split train (1536 queries = 384 base_qa_id × 4 vocabulários
  clean/technical/operational/lay). Sempre `chunks_512` (1 correct + 4 incorrect
  por query). Splits `data/*_informal.parquet` do repo IGNORADOS nesta fase.
- **Modelo**: `LiquidAI/LFM3-1.7B` **as is** — bf16, sem quantização,
  `trust_remote_code`, chat template oficial (thinking-ON, default do template),
  sampling do model card (temp 0.1, top_k 50, rep_penalty 1.1), max_new 3072.
- **Células** (todas com as 1536 queries):
  | célula | contexto | mede |
  |---|---|---|
  | closed_book | nenhum | conhecimento interno (régua do "antes" da injeção) |
  | oraculo | só a passagem correct | teto de síntese |
  | ranking | correct + 4 incorrect embaralhados | robustez a distrator |
  | distrator | só os 4 incorrect | recusa correta vs alucinação |

  Embaralhamento determinístico POR id: `random.Random(sha256(f"42:{id}"))`,
  ordem registrada em cada linha de saída (`shuffle_order`, `correct_position`).

- **Régua honesta** (porte de `bench/regua`, citação FORA do gate):
  - gabarito de fatos por base_qa_id (384) extraído de **answer+evidence** do
    dataset pelo juiz 27B (`data/gabarito_384.jsonl`);
  - cobertura de fatos (casamento barato + 1 desempate do juiz por resposta),
    antitautologia determinística, alucinação contra as evidências;
  - APROVADO (respondíveis) = cobertura ≥ 0.5 ∧ ¬tautologia ∧ ¬alucinação ∧
    ¬recusa ∧ ¬truncada; APROVADO (distrator) = recusa correta;
  - `truncated` (thinking estourou 3072 tokens sem resposta final) NUNCA conta
    como recusa — é coluna própria;
  - juiz = Qwen/Qwen3.8-27B-FP8 em `http://10.100.0.111:8005/v1`
    (`REGUA_JUDGE_URL`); calibração do juiz reutilizada da POC, não refeita.

## Topologia de execução (importante)

O juiz NÃO é roteável da Spark → **geração na Spark, julgamento local**:

1. **Spark** (`walcyrios@spark-b431`, `~/cemig-poc/lfm3_f0/`): venv próprio
   (`uv venv --python 3.12 --managed-python` — o python do sistema não tem
   headers e o triton JIT falha; transformers 5.20.2, torch cu130). Parquet
   copiado via scp (sem login HF lá). Modelo em `hf_cache/` local.
2. **Local (WSL)**: `classifier/.venv` para gabarito/score/analyze.

## Reprodução

```bash
# 0) Preflight (obrigatório antes da bateria; ver PREFLIGHT.md)
#    - sanidade do modelo: sanity_lfm3.py na Spark
#    - render de prompts: render_preflight.py na Spark -> data/preflight_prompts.json
#    - smoke: gen_f0.py --limit 10 por célula + score E2E (data/smoke/)

# 1) Gabarito (local, juiz direto)
../../classifier/.venv/bin/python build_gabarito_f0.py --workers 16

# 2) Geração (na Spark; detached via setsid nohup, NÃO tmux)
#    scp common_f0.py gen_f0.py normas_train.parquet -> ~/cemig-poc/lfm3_f0/
ssh walcyrios@spark-b431   # depois:
#    for c in oraculo ranking closed_book distrator; do
#      python gen_f0.py --cell $c --batch 24 --max-new 3072; done

# 3) Score + análise (local; puxar gen_*.jsonl da Spark antes)
make -C bench/lfm3_f0 pull score analyze
```

Os `data/gen_*.jsonl` commitados estão **gzipados** (contêm o thinking integral,
~86 MB plain); `gunzip -k` antes de rodar `score_f0.py` de novo.

Tabela-resumo: `RESULTS.md`; métricas completas (incl. corte por question_type):
`data/consolidation.json`; respostas cruas + vereditos: `data/gen_*.jsonl` /
`data/scored_*.jsonl` (JSONL por célula, com id, base_qa_id, vocabulary_level,
resposta crua e veredito, como pede o brief).

## Resultados (bateria completa 1536 × 4 células, juiz 27B)

Tabela completa em `RESULTS.md`; métricas por question_type em `data/consolidation.json`.

| célula | métrica-chave | all | clean | technical | operational | lay |
|---|---|---|---|---|---|---|
| closed_book | aprovação | **5,9%** | 7,0% | 8,6% | 4,2% | 3,6% |
| oraculo | aprovação | **83,7%** | 87,5% | 84,1% | 81,8% | 81,2% |
| ranking | aprovação | **65,8%** | 70,6% | 73,2% | 62,0% | 57,6% |
| distrator | recusa correta | **30,5%** | 39,6% | 43,2% | 18,5% | 20,8% |

Achados do baseline (régua do "antes"):

1. **Closed-book ≈ piso (5,9%)** — o LFM3-1.7B praticamente não carrega as NRs
   no peso: recusa indevida 57,5% e, quando arrisca, alucina (45,5%). É a régua
   honesta do "antes" da injeção de conhecimento (F2).
2. **Oráculo alto (83,7%)** — com a passagem certa de bandeja o modelo sintetiza
   bem (cobertura 0,965, citação 93,8%). O teto de síntese NÃO é o gargalo.
3. **Ranking perde ~18 p.p. vs oráculo (65,8%)** — 4 distratores na janela
   derrubam aprovação e dobram a alucinação (10→18%); trunca mais (10,4%)
   porque o thinking rumina sobre os trechos errados.
4. **Distrator é o calcanhar (30,5% recusa correta)** — com só contexto errado o
   modelo prefere responder assim mesmo: 28,3% alucina contradizendo a norma,
   20,2% acerta do conhecimento interno, 35,4% trunca ruminando.
5. **Gradiente de vocabulário consistente**: clean/technical > operational/lay
   em TODAS as células — fala de campo (lay) cai até 13 p.p. no ranking e
   dispara alucinação no closed-book (68,8%). Mesmo abismo lexical já visto na
   POC v2, agora medido também na síntese.
6. Custo: geração ~14 h na GB10 (6144 queries, bf16, max_new 3072, batch 24);
   julgamento local ~40 min (16 workers).

## Decisões registradas

- **System prompt fixo** em todas as células (ver PREFLIGHT.md) — pede PT-BR,
  citação quando houver contexto e honestidade quando o contexto não cobre.
- **Thinking-ON** = comportamento as-is do template oficial. O rascunho
  `<think>…</think>` é salvo (`thinking`) e só a resposta final é avaliada.
- **max_new 3072** calibrado no smoke (1024 truncava 60% do distrator).
- **Sem viés de frequência de norma**: nenhuma ponderação por norma em nenhuma
  etapa (ordem do capitão — ainda não temos essa informação).
- **Benchmark NUNCA vira dado de treino** — este diretório é só avaliação.
