# judge151 — Qualidade de RESPOSTA (E2E) do pipeline híbrido nas 151 reais

Mede a **qualidade da resposta final** (não só o retrieval) do pipeline híbrido completo que
está no main — classificador NR confiança-gated + BM25 boost + FTS5 corrigido + síntese — nas
**151 perguntas reais** (`corpus/qa_pairs_v2.jsonl`), com o juiz vLLM 27B nos 4 eixos.

Responde a pergunta central do capitão: **"quanto do gargalo agora é o LM?"** — comparando o
sintetizador embarcado (LFM2.5-1.2B-Instruct QAD-Q4_0) contra o LFM2.5-2.6B (com e sem thinking).

## TL;DR — Recomendação

- **O gargalo é DUPLO, mas o retrieval domina.** O retrieval intra-norma entrega o chunk-ouro no
  top-2 em apenas **29.1%** das 151 (44/151). Nessas 44, o LM embarcado (1.2B) converte só
  **11.4%** em resposta aprovada; o **2.6B converte 38.6%** — ou seja, **trocar o LM mais que
  triplica a conversão onde o retrieval já acertou**. Mas como 70.9% das perguntas nem recebem o
  chunk certo, o teto de qualidade continua preso ao retrieval.
- **O 2.6B é claramente melhor em qualidade** (global 2.20 vs 1.72; gate-pass 14.6% vs 3.3%;
  acerto factual +0.47) e **alucina menos** nas de retrieval ruim (26.2% vs 35.5% de erro grave).
- **Vale subir para o 2.6B no S24+? Com ressalvas, sim — SE aceitar o custo de tokens.** O
  2.6B **mantém decode acima da leitura humana** no S24+ (proj. ~18 tok/s » 4-7 tok/s de leitura),
  então "acompanha a leitura". O bloqueio real é **RAM (~3.0 GB projetados)**: inviável no S21
  (6 GB, coexistindo com Whisper) e apertado no S24+. **Thinking-ON gera verborragia
  (média 1430 tok, 29/151 truncam mesmo a 2048)** — inviável para voz. **Thinking-OFF (498 tok)**
  é o candidato realista, mas ainda **verboso** (112/151 respostas em markdown com títulos) e
  perde qualidade vs thinking-ON (gate 7.9% vs 14.6%).

**Veredito:** o maior ganho de qualidade por real ainda está em **consertar o retrieval
intra-norma** (rewriter de termos técnicos), não em trocar o LM. Se o capitão quiser o salto de
qualidade do LM mesmo assim, o caminho é **2.6B QAD-Q4_0 thinking-OFF com system prompt reforçado
para 2-4 frases**, e só no S24+/aparelhos ≥8 GB.

---

## Metodologia

Pipeline **idêntico ao app** para cada uma das 151 falas brutas:

1. **Estágio 1 (classificador gated)** — `NrClassifier` (paridade Kotlin↔sklearn provada,
   Δprob 2.3e-8) sobre a **fala bruta**: override de NR explícita → filtro duro; top-1=='nenhuma'
   → sem boost; prob≥0.5 → filtro duro na top-1; senão → boost suave 5x nas top-2.
2. **Estágio 2 (BM25 top-2)** — `index_hf_36nr.db`, `app_fts_query` (stem-6 + prefixo\* + OR,
   pesos 1.5/3/2/1). Distribuição de modos do estágio 1: soft 84, hard 61, none 4, explicit 2.
   Latência estágio 1+2: **mediana 3.2 ms / p95 5.1 ms**.
3. **Síntese** — `SYNTHESIS_SYSTEM_PROMPT` **idêntico** ao `AskPipeline.kt` de produção; contexto
   formatado igual (`[i] (doc - section - title): text`).

Reuso: `classifier/hybrid.py` (busca/boost), `corpus/eval_fino.app_fts_query` + `eval_retrieval.check_hit`
(paridade honesta com o device), `bench/judge.py` (juiz vLLM), `qa_pairs_v2.jsonl` (151).

**Sintetizadores** — todos na RTX 5070 (llama-server CUDA `-ngl 99`, sampling oficial Liquid
`temp=0.1 / top_k=50 / repeat_penalty=1.05`):

| Config | Modelo (GGUF) | Thinking |
| :-- | :-- | :-- |
| `lfm1.2b` | `LFM2.5-1.2B-Instruct-QAD-Q4_0` (embarcado) | n/a |
| `lfm2.6b` | `LFM2.5-2.6B-Q4_0` | **ON** (padrão) |
| `lfm2.6b_noth` | `LFM2.5-2.6B-Q4_0` | **OFF** (`--reasoning-budget 0`) |

**Como desligar o thinking do 2.6B:** o template LFM2.5 **sempre** injeta `<|im_start|>assistant\n<think>`
no prompt de geração — **não há flag `enable_thinking`**. O `llama-server` corta o thinking com
`--reasoning-budget 0` (encerra o `<think>` imediatamente). Confirmado: `reasoning_content` vazio e
resposta direta. Com thinking ON, o `reasoning_content` sai separado do `content`.

**REGRA DE VALIDADE (brief):** qualidade em GPU vale; **latência de GPU não vale** para o aparelho.
Marcamos `engine:cuda` e reportamos os **tokens gerados (thinking incluso)** para projetar a
latência mobile pela tabela do device-bench.

**Juiz:** vLLM 27B (`Qwen/Qwen3.8-27B-FP8`, `http://10.100.0.111:8005/v1`), 8 workers,
`max_tokens=2048`, parser último-JSON — 453 julgamentos (3 × 151).

### Nota de higiene / procedência (armadilha encontrada)

A **porta 8090 estava ocupada por um llama-server de OUTRA lane** (1.2B, alheio a esta tarefa).
A primeira rodada falhou o `bind` silenciosamente e o pipeline conversou com aquele servidor
alheio — respostas quase idênticas entre configs e 0 tokens de thinking. **Corrigido:** porta
dedicada alta (8397) + **guard duplo** em `run_all.sh` (aborta se a porta já estiver em uso e
verifica que o servidor vivo é o nosso). Os resultados abaixo são da rodada limpa.

---

## Tabela 1 — 4 eixos × global × gate-pass (ANTES vs AGORA; 1.2B vs 2.6B)

Notas 1-5. Global = 0.35·Fac + 0.30·Fid + 0.20·Cit + 0.15·PT. Gate-pass = fac≥3 ∧ fid≥3 ∧ cit≥3 ∧ global≥3.5.

| Config | Recall@top-2 | Fac | Fid | PT | Cit | **Global** | **Gate-pass%** |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **ANTES** — v2 pré-fix (1.2B Instruct, classic_rag, n=101) | 20.8% | 1.12 | 1.11 | 3.23 | 1.05 | 1.42 | 1.0% |
| **AGORA** `lfm1.2b` (embarcado, híbrido, n=151) | 29.1% | 1.44 | 1.38 | 3.71 | 1.23 | **1.72** | **3.3%** |
| **AGORA** `lfm2.6b` thinking-ON (n=151) | 29.1% | 1.91 | 2.02 | 3.82 | 1.77 | **2.20** | **14.6%** |
| **AGORA** `lfm2.6b_noth` thinking-OFF (n=151) | 29.1% | 1.88 | 1.72 | **4.09** | 1.64 | 2.12 | 7.9% |

**Leituras:**
- **Antes→agora no 1.2B (mesmo LM):** o fix do FTS5 + classificador elevou todos os eixos
  (global 1.42→1.72, gate 1.0%→3.3%, recall 20.8%→29.1%). O ganho é real mas modesto — confirma
  que **destravar o retrieval move a agulha**, mas o teto ainda é baixo com o 1.2B.
  *(Ressalva: "antes" é n=101 pré-fix; "agora" é n=151. Comparação de tendência, não pareada.)*
- **1.2B→2.6B agora (mesmo retrieval):** **+0.48 de global e gate-pass 4.4×** (3.3%→14.6%).
  O 2.6B thinking-ON é o melhor em Fac/Fid/Cit; o thinking-OFF ganha só em PT (mais fluente por
  não fragmentar), mas cai em fidelidade.

---

## Tabela 2 — Decomposição do gargalo ("quanto é o LM")

Separa as 151 em **retrieval OK** (chunk-ouro no top-2: 44) e **retrieval RUIM** (107).

| Config | Retrieval OK (n=44) pass-gate | acerto≥4 | Retrieval RUIM (n=107) pass-gate | Recusa correta ("não sei") | Alucinação grave (fac≤1) |
| :-- | :--: | :--: | :--: | :--: | :--: |
| `lfm1.2b` | **11.4%** | 13.6% | 0.0% | 45.8% (49) | 35.5% (38) |
| `lfm2.6b` (ON) | **38.6%** | 36.4% | 4.7% | 39.3% (42) | **26.2% (28)** |
| `lfm2.6b_noth` (OFF) | 27.3% | 27.3% | 0.0% | 26.2% (28) | 32.7% (35) |

**A métrica-chave que o capitão pediu:** *nas perguntas onde o retrieval acertou, qual % o LM
converte em resposta correta?*
- **1.2B: só 11.4%** (5 de 44). Ou seja, mesmo com o chunk certo no prompt, o LM embarcado
  desperdiça ~88% dos casos — **o LM É um gargalo pesado onde o retrieval acertou** (exatamente o
  sintoma do Q10 do capitão: chunk correto, síntese falha).
- **2.6B thinking-ON: 38.6%** (17 de 44) — **3.4× a conversão do 1.2B**. Confirma que **boa parte
  do gargalo pós-retrieval é o tamanho/capacidade do LM**, não o prompt.
- Nas de **retrieval ruim**, o 2.6B-ON **alucina menos** (26.2% vs 35.5% do 1.2B) — thinking ajuda
  a "perceber" que o contexto não responde. O thinking-OFF piora a recusa (26.2%) e volta a alucinar.

**Conclusão do gargalo:** é **duplo**. (a) O **retrieval intra-norma** é o teto dominante (só 29.1%
recebem o chunk certo). (b) Dado o chunk certo, o **LM 1.2B é gargalo real** (converte 11.4%); o
2.6B alivia muito isso (38.6%), mas não resolve o (a). O maior retorno por esforço continua no
**rewriter de termos técnicos** (Turno 1) para elevar o recall — achado já registrado no M3.

---

## Tabela 3 — Projeção de latência mobile (device-bench)

`engine:cuda` — latência de GPU **não** vale. Projeção pela tabela do `bench/device-liquid/`:
1.2B QAD-Q4_0 no S24+ = prefill 219 tok/s (pp800), decode real 39.9 tok/s, TTFT 3.65 s. O 2.6B
**não foi medido no device**; projetamos por escala de parâmetros (2.17×) sobre o 1.2B QAD e por
mmap de ~1.6 GB de pesos. Prompt RAG topk2 lean ≈ 850 tok. **S21 ≈ 2.25× mais lento.**

| Config | gen_tok médio | prefill t/s | decode t/s | TTFT S24+ | total S24+ | total S21 | RAM est. | decode > leitura? |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| `lfm1.2b` | 143 | 219 | 39.9 | 3.9 s | ~7.5 s | ~16.8 s | ~1.4 GB | **sim** |
| `lfm2.6b` (ON) | **1430** | ~101 | ~18.4 | 8.4 s | **~86 s** | ~194 s | **~3.0 GB** | sim |
| `lfm2.6b_noth` (OFF) | 498 | ~101 | ~18.4 | 8.4 s | ~35 s | ~80 s | ~3.0 GB | sim |

### O eixo do capitão: TTFT + decode tok/s vs leitura humana (não os 10 s)

Leitura adulta em PT ≈ **3-5 palavras/s ≈ 4-7 tok/s**. Se o decode superar isso com folga, a
resposta "acompanha a leitura" e o tempo total deixa de ser o gate duro.

- **1.2B:** decode ~40 tok/s no S24+ — **~6-10× a leitura humana**. Folga enorme. TTFT ~3.9 s.
- **2.6B (ambos os modos):** decode projetado **~18 tok/s no S24+** — ainda **~2.5-4× a leitura
  humana**, portanto **acompanha a leitura**. No S21 cairia para ~8 tok/s (no limite superior da
  leitura, mas ainda ≥ leitura). **TTFT ~8.4 s** no S24+ é o ponto fraco (o operário espera 8 s
  antes da primeira palavra) — e no S21 ~19 s, inaceitável.
- **Verborragia conta contra o 2.6B:** thinking-ON gera **1430 tok médios** (o operário não vê o
  thinking, mas paga o decode dele antes da 1ª palavra útil → TTFT efetivo explode). Mesmo
  thinking-OFF (498 tok) produz respostas longas em markdown (112/151), contra a diretriz de 2-4
  frases. **Isso precisa ser domado no prompt antes de qualquer embarque.**

**Veredito de latência:** pelo critério novo (decode vs leitura), o 2.6B **passa** — mantém decode
acima da leitura humana no S24+. Os bloqueios reais são **(1) RAM ~3.0 GB** (mata a coexistência
com Whisper no S21 e aperta o S24+) e **(2) TTFT alto + verborragia** do thinking. Se o capitão
aceita estender o teto de tempo em troca de qualidade, o 2.6B **thinking-OFF** é defensável **só no
S24+/≥8 GB**, com prompt reforçado para respostas curtas.

---

## Arquivos e reprodução

```
bench/judge151/
  retrieval.py            # espelho Python do HybridRetriever.kt (gated + BM25 top-2)
  pipeline.py             # E2E: fala -> estágio1+2 -> síntese (checkpoint incremental)
  run_all.sh              # orquestra as 3 configs no llama-server CUDA (porta 8397 + guard)
  build_judge_input.py    # empacota respostas no formato do bench/judge.py
  analyze.py              # tabelas 1-3 (gargalo + projeção mobile)
  data/
    responses_<cfg>.json        # respostas + tokens + decisão estágio 1 por config
    responses_lfm2.6b_cap1024.json  # evidência: thinking-ON truncou 120/151 a 1024 tok
    harness_for_judge.json      # entrada do juiz
    judge_evaluations.json      # 4 eixos por item (juiz 27B)
    judge_summary.csv           # resumo consolidado
    analysis.json               # métricas das tabelas 1-3
```

Reproduzir:
```bash
# 1. Gerar respostas E2E (RTX 5070; ~10-20 min por config)
bench/judge151/run_all.sh                       # todas | ou: run_all.sh lfm1.2b
#    obs.: 2.6b thinking-ON estoura budget 1024 -> rodar com --max-tokens 2048 (ver pipeline.py)

# 2. Empacotar p/ o juiz e julgar (vLLM 27B, 8 workers)
classifier/.venv/bin/python bench/judge151/build_judge_input.py
classifier/.venv/bin/python bench/judge.py \
  --input bench/judge151/data/harness_for_judge.json \
  --output-eval bench/judge151/data/judge_evaluations.json \
  --output-csv  bench/judge151/data/judge_summary.csv --no-calibration

# 3. Analisar
classifier/.venv/bin/python bench/judge151/analyze.py
```

**Interpretador:** `classifier/.venv` (tem scikit-learn 1.9 + requests + scipy). Não usar a
`.venv`/`.venv-train` da raiz.

## Governança

O LLM juiz é **auxiliar**. A leitura humana das 10 primeiras questões do vencedor é **mandatória**
antes de qualquer homologação/embarque em campo (disclaimer herdado de `bench/judge.py`).
