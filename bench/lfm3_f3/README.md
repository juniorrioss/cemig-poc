# bench/lfm3_f3 — F3: robustez de uso de contexto (LFM3-1.7B + retriever real)

Linha conservadora aprovada pelo capitão após o veredito da F2: a injeção de
conhecimento tem teto baixo; o gap RECUPERÁVEL é o de USO de contexto —
oráculo 83,7% vs ranking-1-de-5 65,8% (−18 p.p. só por 4 distratores) e 28%
de alucinação em distrator-puro (números do F0). A F3 treina o modelo a
responder COM os conteúdos do retriever real (distratores de verdade), com
**k VARIÁVEL** (top-3/5/8, aprovado pelo capitão) para não decorar "sempre há
5 passagens".

Alvos: ranking-1-de-5 ≥78% | alucinação distrator-puro <10% | guardas
estáveis (oráculo ~83,7%, closed-book ~5,9%, probe de capacidade geral).

## Desenho do dataset (ancorado no retriever REAL)

1. **Perguntas**: 5.526 QAs validados da F2 (NR-10/18/34) + QAs novos das NRs
   extras **6, 12, 17, 22, 35** pelo MESMO funil da F2 (s1→s2→s4, módulos
   importados de `bench/lfm3_f2/synth`, saída em `data/synth/`; s3 socrático
   pulado — o alvo aqui é pergunta-âncora p/ o retriever, não cadeia densa).
   Subamostragem seedada p/ custo (180 unidades/NR, 4 QAs/unidade antes do
   crítico). Gates do `validate_stage.py` da F2 por estágio.
2. **Retriever real** (`retrieve_f3.py`): v4 EMBARCADO da POC (BM25
   gated-expandido índice do app + denso EmbeddingGemma texto + exp-only,
   RRF k=10 w=2,1,2 — espelho provado `bench/ctx_topk/retrieval_v4.py`) sobre
   o corpus COMPLETO (36 NRs). Top-8 com scores por pergunta.
   **Rótulo por passagem** (critério registrado):
   `containment = |tokens(fato_da_unidade) ∩ tokens(chunk)| / |tokens(fato)|`
   (tokens normalizados sem acento); `gold` se mesmo doc e ≥0,70; `partial`
   se mesmo doc e ≥0,30; senão `distractor`. Pool F2: **gold@8 = 80,5%**.
3. **Casos** (`build_dataset.py --assign`, seed 20260206, quotas do brief):
   - `gold` (~60%): ouro presente; posição ALEATÓRIA registrada
     (`gold_position`); alvo = 27B vendo APENAS a passagem ouro.
   - `absent` (~20%): contexto SÓ distratores; prioriza misses GENUÍNOS do
     retriever (sem ouro nem parcial no top-8), completa removendo o ouro
     (origem registrada em `origin_absent`); alvo = recusa curta padrão
     ("não encontrei nas normas recuperadas") + o que consultaria (o item da
     ref). 4 templates, todos disparam `looks_like_refusal` da régua.
   - `partial` (~10%): sem ouro, melhor passagem `partial` garantida no
     contexto; alvo = 27B vendo APENAS a parcial, responde o sustentado e
     sinaliza o limite.
   - `closed` (~10%): âncora closed-book da F2 (QA sem contexto,
     anti-esquecimento do conhecimento injetável).
   - k: ~50% k=5, ~25% k=3, ~25% k=8 (sorteio por exemplo; casos
     absent/partial podem ficar com k menor quando o pool não enche).
4. **Alvos 27B** (`synth/gen_targets.py`): thinking antes do JSON, parser do
   último bloco válido (reuso `common_synth` da F2). Fidelidade: o gerador
   NUNCA vê os distratores — só a âncora (ouro ou parcial).
5. **Render** (`build_dataset.py --render`): formato chat (messages) com
   system prompt IDÊNTICO ao F0/app (`common_f0.SYSTEM_PROMPT`) e contexto
   renderizado IDÊNTICO ao `render_user` do F0 (`[i] (fonte):\ntexto`).
   Split 98/2 estratificado por caso. Gate final: `validate_dataset.py`
   (schema, rótulo×caso, distribuição k/posições, dedup, entropia ≥6 bits,
   recusa coerente com a régua). Dedup anticontaminação vs benchmark
   (Jaccard ≥0,5, critério do s6 da F2): 0 remoções no pool F2.

## Treino (H100, container `cemig_mobile_walcy`, GPU 2)

Base = `LiquidAI/LFM3-1.7B` instruct LIMPO (decisão conservadora do capitão;
NÃO parte do A-r32-ep2). LoRA `target_modules=["w1","w2","w3"]` com ASSERT
obrigatório (72 lora_A em `*.feed_forward.*`, 0 em `*.self_attn.*` —
`gate_proj` no LFM3 é gate da ATENÇÃO), r32, alpha 64, **SFT chat com
`assistant_only_loss=True`** (o chat_template oficial do LFM3 tem
`{% generation %}`), 4 épocas com checkpoint por época (a F2 provou pico
cedo), lr 1e-4 cosine, bs8×ga4, max_len 6144 (k=8 com chunks de ~1,6k chars).
wandb projeto `lfm3-1.7b-cemig`, run `lfm3-f3-robustez-r32`.
`train/run_train_f3.sh` (docker exec -d + nohup; polling barato).

## Avaliação (harness existente, comparável ao F0)

Por checkpoint, 4 células do F0 (closed_book/oraculo/ranking/distrator,
chunks_512, **1536 queries completas**) com `gen_f2.py` da F2 INALTERADO
(batched bs48): `eval/run_eval_grid_f3.sh`. Julgamento local no 27B
(`eval/score_f3.py`, shim do score_f2 = régua honesta do F0). Parquets
canônicos schema estendido em `data/eval/` (`eval/export_parquet_f3.py`).
Probe de capacidade geral: `bench/lfm3_f2/probe/run_probe.py` (réguas OOD
congeladas). Consolidação vs F0: `analyze_f3.py` → `data/consolidation.json`
+ tabela no `RESULTS.md` (inclui recusa-correta e recusa-indevida — o
trade-off da sft_v3 é reportado, não aceito em silêncio).

## Reprodução

```bash
# 1a. funil NRs extras (27B)        # 1b. retriever real (llama-server :8399)
make funnel gates                    make retrieve
# 1c/1d. dataset
make assign targets render           # inclui validate_dataset.py (GATE)
# 2. treino (H100)
scp data/train/{train,val}.jsonl aluno_walcy@10.100.0.111:/raid/aluno_walcy/cemig_mobile/f3/
ssh aluno_walcy@10.100.0.111 'bash /raid/aluno_walcy/cemig_mobile/f3/run_train_f3.sh'
# 3. avaliação (Spark/H100) + julgamento local
bash eval/run_eval_grid_f3.sh <adapter> f3-r32-ep2
make score-all TAGS=f3-r32-ep2 && make export analyze
```

## Resultados

Ver `RESULTS.md` (preenchido ao fim da grade).
