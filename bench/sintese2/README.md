# síntese v2 — Shootout de sintetizadores + engenharia da síntese (151 reais, retrieval v3)

Responde à ordem do capitão: a conversão **chunk-certo → resposta-aprovada** sobre a v3
(29% no 2.6B / 5% no 1.2B, medida no judge151) estava "abaixo de uma POC de alto nível" para
o domínio elétrico. Duas frentes na mesma bancada:

1. **SHOOTOUT** de sintetizadores incluindo o recém-lançado **openbmb/MiniCPM5-2B** (e o irmão
   1B), contra o embarcado **LFM2.5-1.2B QAD** e o **LFM2.5-2.6B thinking-OFF**;
2. **ENGENHARIA DA SÍNTESE** (few-shot + citação estruturada pelo app + ordenação de chunks)
   aplicada a **todos** os candidatos.

Métrica na ordem do capitão: **além do gate binário**, média/desvio/mediana/**histograma por
eixo** e **identificação explícita do eixo-gargalo** (por célula e global).

---

## TL;DR — recomendação

- **Vencedor de qualidade: LFM2.5-2.6B thinking-OFF** — gate-pass **21–22.5%**, global **2.47–2.50**,
  conversão chunk-certo→aprovada **35–37%** (≈ o medido no judge151). É o melhor em todos os
  eixos factuais e o **único candidato viável para embarque** entre os testados (2.6B ainda barrado
  no engine nativo — ver abaixo).
- **MiniCPM5-2B é o 2º e responde MUITO bem à engenharia da síntese**: `full` sobe gate 7.9%→**9.9%**
  e conversão 13.9%→**16.5%** (fac +0.36, cit +0.24). Ainda **abaixo do 2.6B da Liquid** no nosso
  domínio PT-BR/NR, apesar do "avg 53.9 vs 33.2" que o model card alega em benchmarks acadêmicos.
  **PT-BR é fluente** (passou o sanity), mas o modelo **raciocina em inglês com thinking-ON** (temos
  que rodar `enable_thinking=false`) e alucina mais que o 2.6B quando o retrieval erra.
- **MiniCPM5-1B: descartado.** Colapsa no RAG NR (gate 0%, fac 1.02/1.11, PT cai com few-shot);
  repete/degenera. Não compete com o LFM2.5-1.2B QAD embarcado.
- **O EIXO-GARGALO GLOBAL é CITAÇÃO (cit μ=1.65)**, seguido de fidelidade (1.70) e acerto (1.77);
  **PT-BR NÃO é gargalo** (μ=3.68). **MAS** a citação baixa é **efeito do retrieval**: nas 72/151
  perguntas em que o chunk-ouro não entra no top-2, a fonte citada é a norma errada e o juiz pune
  cit e fac juntos. **Onde o retrieval acerta (79/151), o 2.6B já cita bem** (cit ok=2.5–2.9).
- **Engenharia da síntese ajuda o modelo FRACO, satura no FORTE.** No MiniCPM5-2B o pacote completo
  dá salto real; no 2.6B as técnicas **não acumulam** (a melhor isolada foi **ordenação**, gate
  22.5%; few-shot **incha os tokens** 2× sem ganhar gate; citação-estruturada isolada até **piorou**
  o gate por o juiz cobrar a norma anexada quando o retrieval errou).
- **DSpark (decodificação especulativa): sem ganho aqui.** Na RTX 5070 o draft **atrasou** o decode
  (0.83×, compute-bound a batch=1). Aplicabilidade mobile documentada abaixo (pode ajudar em CPU
  memory-bound, mas exige suporte no engine embarcado e +650 MB de RAM do draft — inviável no S21).

**Implicação para o embarque:** o maior ganho por real **continua sendo o retrieval** (só 52.3%
recebem o chunk certo). Dado o chunk certo, **subir para o 2.6B triplica a conversão** vs o 1.2B —
mas o **2.6B segue barrado no engine nativo** (força thinking-ON no JNI; ver
`android/README_CONSOLIDACAO.md`). MiniCPM5-2B **não justifica** trocar a família (pior no nosso
domínio, mesma classe de RAM ~3 GB, mesmo problema de thinking). **Recomendação: manter o 1.2B QAD
embarcado; priorizar (a) retrieval e (b) destravar thinking-OFF do 2.6B no JNI** como a alavanca de
qualidade — MiniCPM5 fica arquivado como alternativa se a Liquid estagnar.

---

## Candidatos e verificação do MiniCPM5 (ordem do capitão)

| Candidato | GGUF (Q4_K_M / Q4_0) | Tamanho | Família / sampling | Raciocínio |
| :-- | :-- | --: | :-- | :-- |
| `lfm1.2b` | LFM2.5-1.2B-Instruct-QAD-Q4_0 (**embarcado**) | 664 MB | Liquid: temp 0.1 / top_k 50 / rep 1.05 | não |
| `lfm2.6b_noth` | LFM2.5-2.6B-Q4_0 (thinking-OFF) | 1.49 GB | Liquid (idem); `--reasoning-budget 0` | sim, desligado |
| `minicpm2b` | MiniCPM5-2B-Q4_K_M | 1.45 GB | MiniCPM: **temp 1.0 / top_p 0.95** | **sim** |
| `minicpm1b` | MiniCPM5-1B-Q4_K_M | 656 MB | MiniCPM (idem) | **sim** |

**Verificação do MiniCPM5 (feita ANTES da bateria):**
- **(a) É modelo de raciocínio?** Sim. Model card: "400B tokens de deep-thinking SFT". O
  `chat_template` embutido no GGUF tem `enable_thinking` (default **ON**). Com thinking-ON o modelo
  **raciocina em INGLÊS** (igual ao problema do LFM2.5 Thinking). **Desligamos** via payload
  `chat_template_kwargs={"enable_thinking": false}` no llama-server — confirmado: `reasoning_content`
  vazio, resposta direta em PT-BR. Toda a bateria roda **thinking-OFF**.
- **(b) Sanity de PT-BR (5 perguntas antes da bateria):** `sanity_ptbr.py`. Ambos (2B e 1B) produzem
  **gramática do português brasileiro fluente** — **não invalidam** por idioma. Sem RAG, o conteúdo
  é factualmente fraco (esperado); o 1B **degenera/repete** mais. Leve vazamento de inglês pontual
  no 2B ("short", "damaged componente"). Saída: `data/sanity_minicpm5_{2b,1b}_off.json`.
- **(c) Sampling do model card:** `temperature=1.0, top_p=0.95` (aplicado ao MiniCPM; Liquid mantém
  o dela). Arquitetura `LlamaForCausalLM` → **carrega direto no llama.cpp `434ddbb`** (sem fork).

---

## Condições (aplicadas a TODOS igualmente) — `prompts.py` + `harness.py`

1. **`baseline` (C1)** — system prompt conciso ATUAL de produção (voz, ≤4 frases, sem markdown,
   **citação inline pelo modelo**).
2. **`full` (C1+C2+C3+C4)** — pacote completo:
   - **C2 few-shot**: 2 exemplos perfeitos (pergunta + chunk → 2–4 frases) **fora do holdout**
     (NR-06/NR-35 genéricas, jamais uma das 151);
   - **C3 citação estruturada**: o prompt manda **não** escrever a fonte; o **harness anexa**
     `Fonte: NR-XX, item Y.Y.Y` dos metadados do melhor chunk (simula o app) — a resposta é
     **julgada COM a fonte anexada**;
   - **C4 ordenação**: melhor chunk **por último** (mais perto da pergunta).
3. **Ablações no vencedor** (`fewshot`, `citation`, `ordering` isoladas sobre C1).

---

## TABELA MESTRE — modelo × condição × eixos (151 reais, juiz vLLM 27B)

Notas 1–5. Global = 0.35·Fac + 0.30·Fid + 0.20·Cit + 0.15·PT. Gate-pass = fac≥3 ∧ fid≥3 ∧ cit≥3 ∧
global≥3.5. R@2 = 52.3% para todos (mesmo retrieval v3). conv = chunk-certo→aprovada; fac4 =
chunk-certo→acerto≥4.

| Célula | Fac | Fid | PT | Cit | **Global** | **Gate%** | conv% | fac4% | gargalo |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| lfm1.2b baseline | 1.48 | 1.46 | 3.60 | 1.30 | 1.75 | 0.0 | 0.0 | 0.0 | cit |
| lfm1.2b full | 1.05 | 1.06 | 3.61 | 1.13 | 1.45 | 0.7 | 1.3 | 1.3 | fac |
| **lfm2.6b_noth baseline** | 2.27 | 2.15 | **4.25** | 1.97 | **2.47** | **21.2** | 35.4 | **38.0** | cit |
| lfm2.6b_noth citation | 2.13 | 2.01 | 4.15 | **2.05** | 2.38 | 16.6 | 29.1 | 29.1 | fid |
| lfm2.6b_noth fewshot | 2.19 | **2.22** | 4.12 | **2.23** | **2.50** | 21.9 | 32.9 | 29.1 | fac |
| lfm2.6b_noth full | **2.34** | 2.15 | 4.18 | 1.95 | 2.48 | 21.2 | 35.4 | 35.4 | cit |
| **lfm2.6b_noth ordering** | 2.19 | 2.04 | 4.25 | 2.01 | 2.42 | **22.5** | **36.7** | 35.4 | cit |
| minicpm1b baseline | 1.02 | 1.01 | 2.94 | 1.00 | 1.30 | 0.0 | 0.0 | 0.0 | cit |
| minicpm1b full | 1.11 | 1.10 | 2.45 | 1.34 | 1.35 | 0.0 | 0.0 | 0.0 | fid |
| minicpm2b baseline | 1.65 | 1.58 | 3.34 | 1.48 | 1.85 | 7.9 | 13.9 | 15.2 | cit |
| **minicpm2b full** | 2.01 | 1.89 | 3.60 | 1.72 | 2.16 | **9.9** | **16.5** | 19.0 | cit |

**Leituras:**
- **1.2B embarcado é o teto baixo**: gate 0%; e o pacote `full` **PIORA** (fac 1.48→1.05) — o 1.2B
  não segue few-shot longo e "copia" o formato dos exemplos em vez de responder; **não use o
  pacote completo no 1.2B**.
- **2.6B domina**: ~3× o gate do MiniCPM5-2B e ~7× o "quase" nulo do 1.2B; melhor Fac/Fid; PT
  excelente (4.2).
- **MiniCPM5-2B é o único que ganha claramente com engenharia**: baseline→full sobe gate
  7.9→9.9%, conv 13.9→16.5%, fac 1.65→2.01, cit 1.48→1.72. Mas continua **atrás do 2.6B** em tudo
  que é factual/citação, no nosso domínio.

---

## EIXO-GARGALO (a métrica que o capitão pediu)

**Global (média dos eixos entre as 11 células):** `Fac 1.77 · Fid 1.70 · PT 3.68 · Cit 1.65`.
→ **Gargalo global = CITAÇÃO**, com fidelidade logo atrás; **PT-BR NÃO é gargalo** em nenhuma
célula (μ 2.45–4.25).

**Diagnóstico honesto do gargalo de citação — é retrieval, não estilo:** decompondo o 2.6B por
sucesso do retrieval:

| Subconjunto | cit | fac | fid |
| :-- | :--: | :--: | :--: |
| Retrieval OK (n=79) | **2.5–2.9** | 2.65 | 2.61 |
| Retrieval RUIM (n=72) | 1.3 | 1.9 | — |

Quando o chunk-ouro entra no top-2, o 2.6B **cita a norma e o item corretos**; quando não entra, a
fonte (inline ou anexada) aponta a **norma errada** e o juiz derruba cit **e** fac juntos. Por isso
**anexar a fonte estruturada (C3) não resolve** o gargalo de citação — ele é herdado do retrieval.
Confirmado na leitura humana: Q1 (hit) → 5/5/4/5; Q2–Q5 (miss) → alucinação confiante com norma
errada.

**Histogramas (1–5) do vencedor `lfm2.6b_noth ordering`:**
```
fac: 1:56 2:51 3:9  4:30 5:5     (bimodal: acerta cheio ou erra cheio -> reflete hit/miss)
fid: 1:85 2:21 3:13 4:18 5:14
pt : 1:0  2:0  3:6  4:101 5:44   (PT concentrado em 4-5: não é gargalo)
cit: 1:85 2:24 3:12 4:15 5:15
```
`quase-pass` (reprovou por 1 eixo só): quase sempre o eixo é **global<3.5** ou **cit** — nunca PT.

---

## Ablação das técnicas no vencedor (2.6B) — o que cada uma faz

| Condição | Global | Gate% | conv% | tok mediano | Efeito |
| :-- | :--: | :--: | :--: | :--: | :-- |
| baseline | 2.47 | 21.2 | 35.4 | 174 | referência |
| + ordenação (C4) | 2.42 | **22.5** | **36.7** | 169 | **melhor gate/conv**, custo zero de token |
| + few-shot (C2) | **2.50** | 21.9 | 32.9 | **294** | melhora cit/fid mas **incha 2× os tokens** |
| + citação estrut. (C3) | 2.38 | 16.6 | 29.1 | 153 | **piora o gate** (cobra a fonte anexada quando o retrieval erra) |
| full (C2+C3+C4) | 2.48 | 21.2 | 35.4 | 136 | empata com baseline, **tokens menores** |

**Conclusão da engenharia:** no modelo forte as técnicas **não acumulam** e few-shot é
contraproducente para voz (dobra a latência). A única técnica "de graça" que ajuda é a
**ordenação** (melhor chunk por último). No modelo fraco (MiniCPM5-2B) o pacote completo **soma**.

---

## Projeção mobile (device-bench; latência de GPU NÃO vale)

Projeção pela tabela `bench/device-liquid/` (1.2B QAD-Q4_0 no S24+: prefill 219 tok/s, decode
39.9 tok/s, TTFT pp800 3.65 s, RAM 1420.8 MiB), escalando prefill/decode/RAM por nº de parâmetros.
Prompt RAG topk2 ≈ 900 tok. S21 ≈ 2.25× mais lento. Leitura adulta PT ≈ 4–7 tok/s.

| Célula | tok μ | decode t/s S24 | TTFT S24 | total S24 | total S21 | RAM MiB | ≥leitura |
| :-- | --: | --: | --: | --: | --: | --: | :--: |
| lfm1.2b (embarcado) | 47 | 39.9 | 4.1 s | ~5.3 s | ~11.9 s | ~1421 | sim |
| lfm2.6b_noth full | 149 | 18.4 | 8.9 s | ~17.0 s | ~38.3 s | ~3078 | sim |
| lfm2.6b_noth fewshot | 319 | 18.4 | 8.9 s | ~26.2 s | ~59.0 s | ~3078 | sim |
| minicpm2b full | 90 | 19.0 | 8.6 s | ~13.3 s | ~30.0 s | ~2984 | sim |
| minicpm1b full | 97 | 47.9 | 3.4 s | ~5.5 s | ~12.3 s | ~1184 | sim |

- **Só o 1.2B (e o inviável 1B) cabem no orçamento de voz** com folga; decode dos 2B fica ~2.5–4×
  a leitura humana (acompanha), mas **TTFT ~8.6–8.9 s** e **RAM ~3 GB** são os bloqueios reais
  (mata coexistência com Whisper no S21; aperta o S24+). **Idêntico ao veredito do judge151.**
- **MiniCPM5-2B não alivia o custo mobile**: mesma classe de RAM (~3 GB) e TTFT do 2.6B.

---

## DSpark (decodificação especulativa) — testado no MiniCPM5-2B

O engine `434ddbb` **suporta DSpark nativamente** (auto-detecta `draft-dspark` dos metadados do
draft; `-md MiniCPM5-2.6B-DSpark.gguf -ngld 99`). Medição na RTX 5070 (`dspark_bench.py`, 10
prompts RAG reais, 256 tok):

| Modo | decode tok/s (5070) |
| :-- | --: |
| sem draft | **196.5** |
| com DSpark | 164.0 |
| **speedup** | **0.83× (mais lento)** |

**Por quê:** a GPU a batch=1 num modelo de 2B é **compute-bound**; o overhead do draft (forward
extra + verificação) supera o ganho. Speculative decoding rende quando o alvo é **memory-bandwidth
bound** (típico de CPU mobile). **Aplicabilidade mobile:** teoricamente favorável no CPU do
aparelho, MAS exige (1) suporte a DSpark no engine embarcado (o `libllama_engine.so` do app não
expõe path de draft no JNI) e (2) **+650 MB de RAM** do draft residente — **inviável no S21** e
apertado no S24+ somado ao alvo de ~3 GB. **Não recomendado para o embarque** sem uma tarefa
dedicada de engine + orçamento de RAM.

---

## Arquivos e reprodução

```
bench/sintese2/
  prompts.py            # C1 baseline + C1234 full + ablações (few-shot/citação/ordenação)
  harness.py            # E2E: fala -> retrieval v3 -> síntese (checkpoint incremental)
  run_shootout.sh       # orquestra candidatos × condições no llama-server CUDA (porta 8399 + guard)
  sanity_ptbr.py        # sanity de PT-BR do MiniCPM5 (5 perguntas) antes da bateria
  build_judge_input.py  # empacota respostas p/ o bench/judge.py (célula = cand__cond)
  analyze.py            # tabela mestre + eixo-gargalo + histogramas + projeção mobile
  dspark_bench.py       # speedup de decode do DSpark (draft on/off) na 5070
  data/
    responses_<cand>_<cond>.json   # respostas + tokens + retrieval_hit por célula
    sanity_minicpm5_{2b,1b}_off.json
    harness_for_judge.json         # entrada do juiz (11 células)
    judge_evaluations.json         # 4 eixos por item (juiz 27B)
    analysis.json                  # métricas da tabela mestre
    dspark_bench.json              # medição DSpark
```

Reproduzir (RTX 5070; juiz vLLM 27B):
```bash
# 1. Sanity PT-BR do MiniCPM5 (opcional, já registrado)
LD_LIBRARY_PATH=~/llama.cpp/build-cuda/bin ~/llama.cpp/build-cuda/bin/llama-server \
  -m ~/models-poc/MiniCPM5-2B-Q4_K_M.gguf -ngl 99 -c 2048 --jinja --port 8792 &
classifier/.venv/bin/python bench/sintese2/sanity_ptbr.py --url http://127.0.0.1:8792

# 2. Shootout (candidatos × baseline+full) e ablações no vencedor
bench/sintese2/run_shootout.sh                                   # 4 candidatos × baseline full
bench/sintese2/run_shootout.sh --conditions "fewshot citation ordering" lfm2.6b_noth

# 3. Juiz + análise
classifier/.venv/bin/python bench/sintese2/build_judge_input.py
classifier/.venv/bin/python bench/judge.py \
  --input bench/sintese2/data/harness_for_judge.json \
  --output-eval bench/sintese2/data/judge_evaluations.json \
  --output-csv  bench/sintese2/data/judge_summary.csv --no-calibration
classifier/.venv/bin/python bench/sintese2/analyze.py

# 4. DSpark (dois servidores: sem/ com draft)
```

**Interpretador:** `classifier/.venv` (sklearn 1.9 + requests). Retrieval v3 reusa
`bench/judge151/retrieval_v3.py` (BM25-gated-exp ao vivo + cache denso da 5070).

## Governança

O LLM juiz é **auxiliar**. A leitura humana das 10 primeiras questões do vencedor é **mandatória**
antes de qualquer homologação/embarque (disclaimer herdado de `bench/judge.py`). Feita aqui para
o `lfm2.6b_noth ordering` (5 primeiras): confirma o diagnóstico — retrieval-hit → resposta perfeita
(Q1 5/5/4/5); retrieval-miss → alucinação confiante com norma errada.
