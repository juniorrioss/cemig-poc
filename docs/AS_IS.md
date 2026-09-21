# AS_IS — A última entrega da POC CEMIG e o porquê de cada escolha

> **Documento canônico de handoff de engenharia.** Descreve **o que está vigente** (a última
> entrega) e **por que a arquitetura ficou assim**, com procedência de cada número (arquivo-fonte
> ao lado do valor). Para o mapa cronológico dos experimentos anteriores (o que foi superado e por
> quem), ver [`HISTORICO.md`](HISTORICO.md). Para a apresentação executiva, ver
> [`relatorio_poc.html`](relatorio_poc.html) — este arquivo serve à **continuidade** (engenheiro
> que vai retomar), o HTML serve à apresentação.
>
> **Regra de leitura**: cada afirmação está marcada como **[MEDIÇÃO]** (medido nas fontes),
> **[PROJEÇÃO]** (derivado/estimado, não medido diretamente) ou **[NÃO MEDIDO]** (lacuna
> declarada). Barra de erro típica em n=151: **~±3 p.p.** (amostral); onde uma decisão depende de
> diferença menor que isso, está escrito na própria linha.
>
> Divergências encontradas entre o brief da tarefa e as fontes estão listadas na
> [seção final](#divergências-entre-o-brief-e-as-fontes). **Vale a fonte.**

---

## PARTE A — O QUE É A ÚLTIMA ENTREGA (o vigente)

A última entrega integrada ao app é o **pipeline híbrido de tool-calling** (task `poc-app-tools`,
commit `c340cfa`), documentado em
[`android/README_TOOLS_HYBRID.md`](../android/README_TOOLS_HYBRID.md), que embarca o modelo
treinado em [`tools_v1/`](../tools_v1/README.md) com o desenho validado em
[`tools_oraculo/`](../tools_oraculo/README.md).

### A.1 — Stack de produção (no aparelho)

```
  fala (PT-BR, push-to-talk)
        │
        ▼
  ┌────────────────────┐  Nemotron 3.5 INT8 (sherpa-onnx), streaming 560 ms
  │        ASR         │  ~1,2 s no S24+, 91,9% de termos técnicos     [asr/, sft_v3/]
  └─────────┬──────────┘
            │ fala BRUTA (transcrição)
            ▼
  ┌────────────────────┐  LFM2.5-1.2B tool-trained (r128 Q4_0)
  │  DECISÃO do modelo │  o modelo decide: chamar buscar_norma(...) OU responder direto
  └─────────┬──────────┘  (fim da heurística Jaccard de reuso)          [tools_oraculo/]
            │
       chamou?├── NÃO → responde direto (reuso / saudação / fora de escopo), ZERO busca
            │ SIM
            ▼
  ┌────────────────────┐  IGNORA a `consulta` reescrita pelo modelo;
  │  BUSCA EXTERNA     │  busca com FALA BRUTA + classificador NR (gate) + RRF v4
  │  (fala crua + v4)  │  top-2 chunks                        [classifier/, retrieval4/]
  └─────────┬──────────┘
            │ 2 chunks + histórico + fala
            ▼
  ┌────────────────────┐  o MESMO 1.2B sintetiza a resposta (citação inline)
  │      SÍNTESE       │  n_ctx 2048, ≤1 busca/turno, prompt podado a 1700 tok
  └─────────┬──────────┘
            ▼
   resposta com fonte (ex.: "NR-10, Anexo II")
```

- **ASR = Nemotron 3.5 INT8** (`-Pcemig.asrEngine=nemotron`), substituiu o Whisper Base
  (ver Decisão 9). **[MEDIÇÃO]** `asr/README.md`, `sft_v3/README.md` (Parte 6).
- **Sintetizador = LFM2.5-1.2B tool-trained r128 Q4_0** servido como override externo sob o nome
  `LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf`. **[MEDIÇÃO]** `android/README_TOOLS_HYBRID.md`.
- **Renderização do ChatML de tool-calling é NOSSA e determinística** (`LfmToolRenderer.kt`), provada
  byte-a-byte contra o template oficial — o `llama_chat_apply_template` do JNI não renderiza `tools=`
  nem o papel `tool`. **[MEDIÇÃO]** `android/README_TOOLS_HYBRID.md` (Parte 1).
- **Treino**: 4.236 diálogos em **5 famílias**, dados régua-aprovados; famílias de recusa =
  **24,5%** do total (herdadas do rebalanceamento do `sft_v3`). **[MEDIÇÃO]** `tools_v1/README.md`
  (Passo 1+2), `sft_v3/README.md` (Parte 1). O conjunto está **versionado** em
  `tools_v1/data/train_full.jsonl` (sha256 `be60946f04237e78c159a4caab8d4a9cf3ade2c6ed33ce880d126999a94ae145`,
  procedência em `tools_v1/data/train_full.manifest.json`); reprodução exata na seção **REPRODUÇÃO**
  do README raiz.

### A.2 — Números do vigente (1.2B tool r128, régua honesta `bench/regua`, n=151)

| Métrica | Q4_0 (embarcado) | bf16 | Fonte |
|---|--:|--:|---|
| Oráculo — aprovação | **31,4 ± 4,1%** | 34,9 ± 2,4% | `tools_oraculo/README.md` (M1, 3 runs) |
| Oráculo — alucinação | 28,2 ± 2,2% | 24,3 ± 0,8% | `tools_oraculo/README.md` (M1) |
| Híbrido E2E — aprovação | **19,2 ± 0,6%** | 19,2 ± 1,4% | `tools_oraculo/README.md` (M3, config A) |
| Híbrido E2E — alucinação | 34,2 ± 0,8% | 31,8 ± 0,9% | `tools_oraculo/README.md` (M3, config A) |
| Recusa — F1 | 85,4% | **90,5%** | `tools_oraculo/README.md` (M2, pack 85) |
| Recusa — recusa-indevida (FP) | 6,7% | 3,3% | `tools_oraculo/README.md` (M2) |
| Decisão de chamar — F1 | \~94,9% (recall 100%) | — | `tools_v1/README.md` (Passo 4a) |
| Sintaxe da chamada válida | 99–100% | — | `tools_v1/README.md` (Passo 4b) |
| Reuso-correto / novo-tópico | 84,8% / 100% | — | `tools_v1/README.md` (Passo 4d) |

**Aparelho (S24+, Nemotron + 1.2B residentes)** **[MEDIÇÃO]** `android/README_TOOLS_HYBRID.md`
(Parte 2, `tool_e2e_results.json`, 5/5 casos corretos):
- **RAM PSS**: **3,86 GB** (pico com geração ativa); idle \~2,97 GB.
- **Latência**: reuso/saudação **\~3–7 s**; turnos **com busca \~10–12 s**. Busca RRF v4 on-device
  37–238 ms.

> Nota de comparabilidade: o "oráculo" entrega o chunk-ouro de bandeja (mede só a síntese); o
> "híbrido E2E" usa a busca real (v4, R@2 \~51%), por isso a aprovação cai de \~31% para \~19% — a
> diferença é **retrieval**, não o modelo (ver Decisão 2 e 4).

### A.3 — A 2ª opção por flag: o 2.6B tool (por que NÃO é o padrão)

`-Pcemig.toolModel=2.6b` seleciona o **LFM2.5-2.6B tool-trained r64 Q4** (+ thinking-OFF).
**[MEDIÇÃO]** `tools_2_6b/README.md`:

| | 1.2B tool (padrão) | 2.6B tool (flag) |
|---|--:|--:|
| Oráculo aprovação (Q4 / bf16) | 31,4 / 34,9% | **46,1 / 53,8%** |
| Recusa F1 (Q4 / bf16) | 85,4 / 90,5% | 91,6 / 95,3% |
| E2E S24+ | **\~6–8 s** (sem busca) / 10–12 s | **\~18–29 s** |
| RAM PSS S24+ | **3,86 GB** | **\~4,3–4,75 GB** |

**Por que o 1.2B é o padrão e o 2.6B não**: o 2.6B tem qualidade nitidamente maior (+15–19 p.p. em
oráculo), mas **não cabe no orçamento de voz de 10 s** (TTFT 11–17 s + decode) e a RAM inviabiliza o
S21 e aperta o S24+. Além disso o 2.6B tool teve **catastrophic forgetting** comportamental (−0,43),
que o SFT de síntese pura não tinha. **[MEDIÇÃO]** `tools_2_6b/README.md` (Passo 3 e 2g). O 2.6B é o
alvo de qualidade **quando latência/RAM destravarem** (aparelho ≥8 GB dedicado ou destilação
2.6B→1.2B).

---

## PARTE B — REGISTRO DE DECISÕES (o coração)

Cada linha responde: **o que se escolheu, contra o quê, com que evidência (número + fonte), por
quê, o que muda se você mexer, e quando revisitar.** Confira sempre na fonte antes de reverter.

---

### Decisão 1 — Número de chunks na janela: **2 (full)**

| | |
|---|---|
| **Escolha** | topK = **2**, trecho **full** (default do app inalterado). |
| **Alternativas medidas** | topK ∈ {2,3,4,5} × {full, trim} = 8 células, nas 151 reais, juiz 27B. |
| **Evidência** | A **visão** sobe (chunk-ouro visível **R@2 51,7% → R@5 65,6%**), mas o **gate/qualidade fica FLAT** (global \~1,8; gate-pass \~1–2% nas 8 células; Δgate máx **+0,7 p.p.**, dentro do ruído; `convert_ratio ≈ 0`). Custo no S24+: k5_full triplica o prompt (1419→3066 tok) e o **TTFT (7,9 s → 21,8 s)**. **TRIM** cabe na janela mas **piora o acerto de norma no aparelho: 13/20 vs 18/20 do full**. **[MEDIÇÃO]** `bench/ctx_topk/README.md` (tabela 8 células; PART 2). |
| **Por quê** | Enxergar mais trechos não vira resposta aprovada com o LM 1.2B: o gargalo é **citação/síntese** (o chunk-ouro está visível e o modelo ainda não cita o item), não a visão. Pagar +13,9 s de TTFT por ganho de visão que o modelo não converte é regressão pura de experiência. |
| **Impacto de mudar** | Subir p/ 5 trechos full: **latência \~3× (TTFT)** e n_ctx precisa ir a 4096 (+26 MiB KV), sem ganho de qualidade. TRIM neutraliza a latência mas **perde 5 acertos de norma em 20**. |
| **Quando revisitar** | Se o sintetizador mudar (2.6B v3 ou destilação que **cite o item**): aí a visão extra pode converter. Reabrir com `bench/ctx_topk` (regras: só embarcar se Δgate ≥ 3,0 p.p. **e** TTFT não regredir > 2,0 s). |

---

### Decisão 2 — Busca: **fala crua**, NÃO a `consulta` reescrita pelo tool-calling

| | |
|---|---|
| **Escolha** | A busca externa usa **SEMPRE a fala bruta** + classificador + RRF v4. A `consulta` que o modelo gera na tool-call é **ignorada**. |
| **Alternativas medidas** | (a) fala crua; (b) reescrita do 1.2B; (c) reescrita do 27B; medido no 1.2B e confirmado no 2.6B. |
| **Evidência** | **1.2B**: reescrita R@2 **22,3%** vs **fala crua 30,5%** (delta **−8,2 p.p.**); got_gold quando chama **19–25% (reescrita) vs \~51% (fala crua+v4)**. **2.6B**: reescrita R@2 **23,2%** vs crua 30,5% (delta **−7,3 p.p.**); nem com mais capacidade supera a fala crua (melhor caso empata em −1,4). O 27B rewrite **empata** com a crua em R@2 (−0,7) e só ganha em R@5 (+9,3). **[MEDIÇÃO]** `tools_v1/README.md` (Passo 4c/e), `tools_2_6b/README.md` (Passo 2c). |
| **Por quê** | O rewrite do modelo pequeno adiciona termos que dispersam o BM25. O poder de reescrita útil está só no 27B — e mesmo nele o ganho aparece só em R@5. Coerente com o achado antigo "reescrita PIORA — usar fala bruta" (`classifier/`, `finetune/README_M3.md` — removido na limpeza da camada 1+2; recuperável em `poc-completa-pre-limpeza:finetune/README_M3.md`). |
| **Impacto de mudar** | Passar a buscar com a consulta do modelo cai a aprovação E2E de **19,2% para 7–8%** (config B "tools puro") e sobe a alucinação (a maior parte da alucinação do "tools puro" — 70–80% — vem de **contexto errado**). **[MEDIÇÃO]** `tools_oraculo/README.md` (M3/M4). |
| **Quando revisitar** | Se surgir um rewriter (on-device ou 27B na nuvem, em modo conectado) que **supere** a fala crua em R@2 — hoje nenhum supera. |

---

### Decisão 3 — Classificador de NR sobre a **fala crua** (confiança-gated)

| | |
|---|---|
| **Escolha** | Estágio 1 leve (LogReg + TF-IDF char_wb 2-5 + word 1-2, Kotlin puro) sobre a **fala bruta**, com política **confiança-gated**: filtro-duro se prob(top-1) ≥ 0,5; senão **boost suave 5×** nas top-2; sem boost se top-1 = 'nenhuma'; **override** se a NR é citada explicitamente. |
| **Alternativas medidas** | Boost 2×/5×/10×; filtro-duro puro; **rewrite + boost** (pior); classificar via prompt no LFM2.5 (descartado); shootout de classificadores clássicos (logreg venceu por top-2 69% + probs calibradas). |
| **Evidência (ablação isolada, 151 reais)** | **Baseline fala bruta SEM boost R@2 13,9%** → **CONFIANÇA-GATED R@2 29,8% / R@5 41,1% / MRR 0,293** (**+15,9 p.p., 2,1×**). 5× é o joelho (10× não melhora). Rewrite+boost **piora** (22,5% < 28,5%). **[MEDIÇÃO]** `classifier/README.md` (FASE C, tabela `hybrid.py`). |
| **Por quê** | Saber a NR certa vale \~+30 p.p. de recall (escada: fala bruta 13,9% → norma-ouro 44,4%); o classificador recupera boa parte disso com custo trivial (<1 ms desktop / \~8 ms S24+). Prompt-classify no SLM: melhor formato (JSON) só 25,7% top-1 e custaria \~4,5 s TTFT/consulta. |
| **Impacto de mudar** | Remover o classificador reverte o R@2 de \~29,8% para 13,9% (baseline). Trocar o gate/boost mexe no equilíbrio filtro-duro×boost — o 5x-gated foi o melhor em R@2 **e** R@5. |
| **Nota de medição** | O brief suspeitava que **não existia** ablação isolada com/sem classificador na fala crua. **Ela existe** e está em `classifier/README.md` (baseline 13,9% vs gated 29,8%, mesmas 151, mesma fala bruta). Ver [divergências](#divergências-entre-o-brief-e-as-fontes). |
| **Quando revisitar** | O teto do desenho está claro: top-2 do classificador = 72% e BM25 intra-norma com fala bruta = 44,4%. Ganho além disso depende de **termos técnicos** (rewriter), não de classificação. |

---

### Decisão 4 — Arquitetura de ferramenta (híbrido) vs pipeline fixo

| | |
|---|---|
| **Escolha** | **Híbrido**: o modelo tool DECIDE quando buscar/reusar; a busca usa fala crua + classificador + RRF v4 (nunca a consulta do modelo). |
| **Alternativas medidas** | (A) híbrido; (B) tools puro (busca com a consulta do modelo); (C) pipeline fixo sem tool (busca v4 sempre). |
| **Evidência** | Aprovação E2E (151): **(A) híbrido 19,2% (Q4/bf16)** vs **(B) puro 7,1% (Q4) / 8,2% (bf16)** vs **(C) fixo 10,1% (Q4) / 16,2% (bf16)**. Decisão de chamar: **F1 95%, recall 100%**; reuso 84,8%; novo-tópico 100%. **[MEDIÇÃO]** `tools_oraculo/README.md` (M3), `tools_v1/README.md` (Passo 4). |
| **Por quê** | O híbrido bate o tools puro por \~11–12 p.p. e **não perde** para o pipeline fixo, e ainda entrega o **multiturno/decisão** que o pipeline fixo não tem (fim do reuso por Jaccard). |
| **Impacto de mudar** | Voltar ao fixo perde o multiturno (decisão de quando buscar/reusar). Ir para o "tools puro" (busca com a consulta) despenca a aprovação (ver Decisão 2). |
| **Quando revisitar** | O híbrido é a recomendação corrente; reabrir só se o produto multiturno mudar de requisito ou o rewriter superar a fala crua. |

---

### Decisão 5 — Modelo pequeno vs grande com tool: **1.2B (padrão)**, 2.6B por flag

| | |
|---|---|
| **Escolha** | **1.2B tool r128 Q4** embarcado; 2.6B tool r64 Q4 selecionável por build. |
| **Alternativas medidas** | 1.2B tool vs 2.6B tool (mesma bancada maçã-com-maçã, mesmos 4.236 diálogos). |
| **Evidência** | Oráculo Q4: **1.2B 31,4% vs 2.6B 46,1%** (bf16 34,9 vs 53,8). Custo aparelho: **1.2B \~6–8 s / 3,86 GB** vs **2.6B \~18–29 s / 4,3–4,75 GB**. **[MEDIÇÃO]** `tools_2_6b/README.md` (Passo 2a, 3). |
| **Por quê** | O 1.2B é o único que cabe no orçamento de voz (≤10 s) e coexiste com o ASR sob o LMK do S24+/S21. O salto de qualidade do 2.6B é real mas o custo de voz é proibitivo hoje. |
| **Impacto de mudar** | Ligar o flag 2.6B: +15–19 p.p. de aprovação em oráculo, mas latência \~3× e RAM que mata apps de fundo (inviável S21). |
| **Quando revisitar** | Quando latência/RAM destravarem (aparelho dedicado ≥8 GB, engine mais rápido) ou destilando 2.6B→1.2B. Preferir **bf16** se subir (Q4 custa \~8 p.p. no oráculo do 2.6B). |

---

### Decisão 6 — Composição do dado de treino (4 famílias, recusa **24,5%**)

| | |
|---|---|
| **Escolha** | 4 famílias: oráculo **75,5%** + recusa 9,8% + parcial 9,8% + distrator 4,9% (famílias de recusa = **24,5% < 25%**), 3065 pares régua-aprovados (dataset do `sft_v3`, reusado no tool-calling). |
| **Alternativas medidas** | Composição do **v2** (oráculo 48%, famílias de recusa **52%**: recusa 20,8 + parcial 20,8 + distrator 10,4) vs **v3** rebalanceado. |
| **Evidência** | v2→v3: **recusa-indevida (FP) caiu de 23–27% para 10–13%**; **F1 de recusa 90,7 → 95,7** (r64), recusa-correta ≥93% mantida. **Distrator caiu de 320 (v2) para 150 (v3) exemplos SEM perda** (o distrator "já bastava a 10% no v2"). **[MEDIÇÃO]** `sft_v2/README.md` (Parte 1, Achado 2), `sft_v3/README.md` (Parte 1, Parte 3). |
| **Por quê** | O v2 pagou over-refusal (recusava casos respondíveis) por ter 52% de recusa. Baixar para 24,5% e dobrar o oráculo mantém a recusa útil e derruba o dano colateral, mantendo o perfil cauteloso. |
| **Impacto de mudar** | Subir a fração de recusa reintroduz o over-refusal (recusa-indevida sobe); baixar demais reintroduz o chute confiante/alucinação (o problema do v1 só-oráculo). |
| **Quando revisitar** | Se o teto de "recusa-indevida" (\~3,3% no 27B) precisar ser fechado — hoje o gap restante é **capacidade de discriminar dá/não-dá**, não desenho de dados. |

---

### Decisão 7 — Rank do LoRA

| | |
|---|---|
| **Escolha** | **r128 Q4** no tool-calling do 1.2B (ranks quase indiferentes; r128 tem melhor reuso/argumento marginal). No SFT de síntese pura: **r64** (2.6B v3 e 1.2B). |
| **Alternativas medidas** | r16 / r32 / r64 / r128 (1 época cada). |
| **Evidência** | **eval_loss cai monotônica com o rank** em todos (não satura na loss). Mas a **régua satura**: no 2.6B v3 **r64 ≈ r128** (r64 tem melhor retrieval real e menor recusa-indevida); no 1.2B tool F1 94,9–95,4 e aprovação 22–24 (±3) — dentro do ruído entre ranks. **Exceção**: no `sft_v2` a varredura **moveu a agulha** (r16→r64 **+12,6 p.p.** de aprovação, −8,0 p.p. alucinação) porque o r16 do v1 estava sub-parametrizado (bug de targets). **[MEDIÇÃO]** `sft_v2/README.md` (Achado 1), `sft_v3/README.md` (Parte 4), `tools_v1/README.md` (Passo 4e). |
| **Por quê** | A saturação é **dependente do modelo/tarefa**: uma vez que MLP+atenção entram no LoRA, aumentar o rank só ajuda até o ponto em que a régua para de responder. |
| **Impacto de mudar** | Subir o rank aumenta os parâmetros treináveis (r128 = 5,64% vs r64 2,90%) sem ganho de régua confiável; no 1.2B tool a latência/RAM não mudam (mesma arch/quant). |
| **Quando revisitar** | Ao trocar o dataset ou a família de modelo — a saturação precisa ser re-medida (não assumir). |

---

### Decisão 8 — Quantização bf16 → Q4 (custo VARIA por modelo/métrica; calibração NÃO paga)

| | |
|---|---|
| **Escolha** | **Q4_0 SEM calibração (imatrix)** no embarque. |
| **Alternativas medidas** | bf16 (teto do treino) vs Q4_0 vs Q4_K_M; **com e sem** calibração imatrix; QAD de fábrica vs requant simples. |
| **Evidência (o custo é assimétrico)** | **`sft_v2`**: bf16→Q4 custa **−9,3 p.p.** de aprovação (o hedge é sensível ao 4-bit). **`sft_v3`**: bf16→Q4_0 **−0,6 p.p.** (negligível), mas a **alucinação no contexto recuperado sobe +6,7 p.p.** (retr aluc bf16 34,0 → Q4 40,7). **2.6B tool**: oráculo bf16 53,8 → Q4 **46,1** (\~−8 p.p.). **Calibração imatrix NÃO recuperou nada** — até piorou levemente o Q4_0 (−2,0) e subiu alucinação no Q4_K_M. O **QAD de fábrica** do 1.2B é **piso** no nosso domínio (2,7% oráculo, abaixo da requant simples 4,7%) → não há vantagem QAD a preservar. **[MEDIÇÃO]** `sft_v2/README.md` (Achado 3), `sft_v3/README.md` (Parte 4, `calibration_comparison.json`), `sft_1_2b/README.md` (Risco QAD), `tools_2_6b/README.md`. |
| **Por quê** | O 4-bit atinge mais os modelos que fazem hedge/recusa fina (v2); modelos com menos over-refusal (v3) quantizam quase de graça. A imatrix não paga na nossa distribuição. |
| **Impacto de mudar** | Embarcar bf16 (se houver folga de RAM/latência) recupera \~8 p.p. no 2.6B e reduz alucinação no v2; para o 1.2B tool o efeito é misto (r16 Q4 até sobe). |
| **Quando revisitar** | A cada novo checkpoint — o custo de quantização é **específico do modelo** e precisa ser medido bf16 vs Q4 lado a lado, não assumido. |

---

### Decisão 9 — ASR: **Nemotron 3.5 INT8** substituiu o Whisper Base

| | |
|---|---|
| **Escolha** | **Nemotron 3.5 Streaming 0.6B INT8** (sherpa-onnx) como ASR do vigente (`-Pcemig.asrEngine=nemotron`); fallback honesto para Whisper Base se os pesos faltarem. |
| **Alternativas medidas** | Whisper Tiny/Base/Small Q5_1; Nemotron INT8 (oficial e Ottema PT-BR) e GGUF (Q4_K_M–Q8_0); ASR nativo Android; ONNX INT4. |
| **Evidência** | **Termos técnicos: Nemotron 91,9% vs Whisper Base 85,5%**; latência Nemotron **2,03 s** (bench asr) / **\~1,2 s** no app vs Whisper Base 4,70 s; RAM **791,7 MB**. **Whisper Small descartado por medição**: **RTF 2,81 (clean) / 2,99 (ruído), \~15,8 s de latência** — inviável para push-to-talk. **[MEDIÇÃO]** `asr/README.md` (tabela §3), `sft_v3/README.md` (Parte 6). |
| **Por quê** | Nemotron acerta o jargão elétrico (desenergização, seccionadora, talabarte, LOTO) que o Whisper erra, e é mais rápido; o RAG depende do termo certo na transcrição. |
| **Impacto de mudar** | Voltar ao Whisper Base economiza RAM (\~197 MB vs \~792 MB) — importante no **S21 (6 GB)**, onde Nemotron + SLM aperta o LMK — mas perde termos técnicos. O `asr/README.md` recomenda **Whisper Base para o S21** e Nemotron para aparelhos de alta RAM; o vigente do app usa Nemotron por build no S24+. |
| **Quando revisitar** | Para o S21: Nemotron não desce de \~792 MB (nem quantizado GGUF, que sobe a RAM para 943 MB+). Um transdutor menor (Parakeet 110M) ou manter Whisper Base é o caminho de baixa RAM. |

---

### Decisão 10 — Recusa como dado sintético (a família que destravou o "não sei" honesto)

| | |
|---|---|
| **Escolha** | Fabricar dados **COM e SEM oráculo** (recusa/parcial/distrator), ensinando recusa honesta E útil (diz o que faltou), não a preguiçosa "diga só: Não sei". |
| **Alternativas medidas** | Treino só-oráculo (v1) vs treino com famílias de recusa (v2/v3). |
| **Evidência** | **recusa-correta 47% (v1 só-oráculo) → 100% (v2)**; **alucinação com contexto errado 51,8% → \~30%**. O v1 só-oráculo **PIOROU** a recusa vs o base (46,7% < 60%): ao ficar assertivo sem aprender a recusar, inventa. **[MEDIÇÃO]** `sft_v2/README.md` (Achado 2). |
| **Por quê** | Em segurança do trabalho, inventar com o contexto errado é inaceitável; a família de recusa ensina a delimitar. |
| **Impacto de mudar** | Remover as famílias de recusa reintroduz o chute confiante (alucinação de contexto errado sobe). |
| **Quando revisitar** | O custo (recusa-indevida) já foi atacado no v3 (Decisão 6). |

---

### Decisão 11 — Corpus: reparo **cirúrgico** (Anexo II NR-10), NÃO reprocessar tudo

| | |
|---|---|
| **Escolha** | Reparo cirúrgico só do Anexo II da NR-10 (via Document AI) + fix de tokenizer numérico + isolar tabelas genéricas fora do BM25. **NÃO substituir** `index_hf_36nr.db` pelo aditivo. |
| **Alternativas medidas** | Aditivar TODAS as 23 NRs tabulares reprocessadas ao BM25 vs reparo cirúrgico vs REPLACE colapsado. |
| **Evidência** | DocAI **eliminou os glifos PUA na origem** e reconstruiu o Anexo II **18/18 faixas** (verificado célula-a-célula, 43.803 números, 0,5% não-reparável). Mas **aditivar tudo POLUI o BM25: R@2 (151) 13,9 → 11,9 (−2,0 p.p.)** — mesmo efeito dos manuais. O caso 13,8 kV precisa de **2 alavancas independentes**: dado (DocAI) + tokenizer numérico (o app hoje descarta "13,8" e "kV"). **[MEDIÇÃO]** `docai/README.md` (recall 151, casos do capitão). |
| **Por quê** | Só a tabela **estruturada** (zona de risco) tem valor; despejos genéricos (portarias, CNAE) são ruído lexical que degrada o IDF da coleção. |
| **Impacto de mudar** | Substituir o índice pelo aditivo-23 derruba o recall geral (−2,0 R@2). O reparo cirúrgico só ajuda o caso 13,8 kV **se** o tokenizer numérico também for corrigido. |
| **Quando revisitar** | Achado honesto do `slm_oraculo` (removido na limpeza; ver `poc-completa-pre-limpeza:slm_oraculo/README.md` e o módulo agora em `corpus/corpus_fix.py`): **0/151 chunks-ouro do holdout tocam a tabela corrompida** (as perguntas de zona resolvem para seções definitórias 10.1/10.2/10.6), então o reparo quase não move o teto das 151. Revisitar se surgirem perguntas cujo chunk-ouro seja a tabela em si. |

---

### Decisão 12 — Prompt engineering: **rejeitado** (retorno \~zero, anti-evasão piora)

| | |
|---|---|
| **Escolha** | **NÃO alterar** o `SYNTHESIS_SYSTEM_PROMPT`. |
| **Alternativas medidas** | 6 variantes de prompt no 2.6B (baseline, `numeros`, `acao_primeiro`, `anti_evasao`, `combinado_4f/6f`), régua honesta, n=151. |
| **Evidência** | Aprovação varia 15,2–23,8%; a melhor (`numeros` 23,8%) **empata** com o baseline (23,2%) dentro do ruído. **Anti-evasão e ação-primeiro PIORAM** (forçam o modelo a inventar a ação → alucinação sobe a 55%). **[MEDIÇÃO]** `bench/prompt_teto/README.md` (Parte 1). |
| **Por quê** | A resposta evasiva do app não é (principalmente) culpa do prompt — é falta do trecho certo/legível. Mudar o prompt para "proibir evasão" no 1.2B trocaria "não disse nada" por "errado com confiança". |
| **Impacto de mudar** | Ligar anti-evasão sobe a alucinação (55%) sem subir a aprovação. |
| **Quando revisitar** | Prompt fica por último na fila de alavancas; só faz sentido depois de retrieval + capacidade do sintetizador. |

---

### Decisão 13 — Expansão do índice: coloquial (v3) + ASR-robusta (v4)

| | |
|---|---|
| **Escolha** | Índice FTS5 com **expansão de documento** (v3, campo `expansion`) + **expansão ASR-robusta** (v4, variantes de erro de transcrição no denso exp-only). Fusão RRF **k=10 pesos (BM25=2, texto=1, exp=2)**. |
| **Alternativas medidas** | Baseline BM25; +expansão v3; +denso EmbeddingGemma; fusão 3 sinais; expansão v4 no denso-texto (piorou) vs denso-exp (ajudou); reranker 27B (cloud, descartado por violar offline); pares-de-graça no classificador (Frente A do v4, descartada por mascarada). |
| **Evidência** | Expansão v3 é o maior ganho isolado: **R@2 29,8 → 37,7%, R@5 41,1 → 52,3%**. Fusão 3 sinais v3: **R@2 52,3% / R@5 63,6%**. v4 sobe pouco no texto limpo (**R@2 51,0 → 51,7%**) mas **muito sob ASR real: R@2 41,1 → 49,0% (+7,9 p.p.)** — o ganho que importa no campo. **Frente A (classificador com pares-de-graça) DESCARTADA**: sobe a acurácia do holdout (69→79%) mas **derruba o recall decisivo** (R@2 51→45–50%); mascarada +35 a +47 p.p. **[MEDIÇÃO]** `retrieval3/README.md`, `retrieval4/README.md`. |
| **Por quê** | Cada ponto de recall multiplica todo o pipeline; a expansão de documento generaliza (conhecimento de domínio), enquanto pares-de-graça do mesmo gerador são overfit que a régua decisiva expõe. |
| **Impacto de mudar** | Remover a expansão v3 volta o R@2 a \~29,8%. Adicionar a expansão v4 ao denso-texto (em vez do exp-only) **piora** (o ruído de ASR polui o vetor de passagem). |
| **Quando revisitar** | O R@2 real segue sendo o teto prático (\~51% no texto limpo, \~49% sob ASR). Novas fontes de recall (termos técnicos via rewriter que supere a fala crua) são a próxima alavanca. |

---

### Decisão 14 — Métrica: **régua honesta** (fatos + antitautologia, citação FORA do gate)

| | |
|---|---|
| **Escolha** | A régua de aprovação é `bench/regua`: **cobertura de fatos ≥ 0,5 E não-tautológico E sem alucinação**; a **citação é informativa, fora do gate**. |
| **Alternativas medidas** | Gate histórico (exigia `citacao_fonte ≥ 3`) vs régua honesta (só "a resposta contém a informação pedida?"). |
| **Evidência** | O gate antigo era **dominado por citação**: **32,3% das aprovações antigas REPROVAM na régua honesta, e 100% delas tinham citação** ("qualidade oca"). A **régua NÃO está severa**: o teto do 27B com chunk-ouro é **84,1%** (`prompt_teto`, prompt `numeros`) — bem acima de 60%, provando que a prova é passável por um modelo competente. **[MEDIÇÃO]** `bench/regua/README.md` (resposta a/c), `bench/prompt_teto/README.md` (Parte 2). |
| **Por quê** | "O EPI para altura é obrigatório" citando a NR passava no gate antigo sem responder nada. A régua honesta mede a informação, não o enfeite. |
| **Impacto de mudar** | Reintroduzir citação no gate infla a "qualidade" em \~32% sem conteúdo; baixar o limiar de cobertura aprova respostas incompletas. |
| **Quando revisitar** | O teto do 27B (84%) sobe/desce com o corpus (tabelas mal extraídas), não com a régua — a alavanca é dado/retrieval. Nota: o teto varia por fonte entre **82,1% e 84,3%** (ver [divergências](#divergências-entre-o-brief-e-as-fontes)). |

---

## PARTE C — LACUNAS E LIMITES (o que NÃO foi medido)

Honestidade obrigatória. Cada item marcado por status.

- **Barra de erro em n=151 (\~±3 p.p.)** **[MEDIÇÃO/LIMITE]**: várias comparações que embasam
  decisões estão **dentro do ruído** e isso está dito nas próprias linhas (ranks do tool-calling,
  `numeros` vs baseline no prompt, Δgate do topK). Não trate diferenças < 3 p.p. como reais sem
  3 runs.
- **ASR com voz humana real** **[NÃO MEDIDO]**: todo o benchmark de ASR usa **áudio sintético
  (edge-tts)** — prosódia estável, sem sotaque regional/hesitação/ofego de campo. A bancada de voz
  humana existe (`asr/data/real_voice/`, `bench_real_voice.py`) mas **os áudios ainda não foram
  gravados/medidos**. Fonte: `asr/README.md` (limitação metodológica), commit `c786426` ("WER de
  voz real fica por medir"). O "9/10 sob ASR" do retrieval4 usa TTS→Whisper, não voz humana.
- **Teste com usuário de campo real** **[NÃO MEDIDO]**: não há estudo com eletricista usando o app
  em condição operacional (ruído, luvas, pressa). Toda a "aceitação" é roteiro de 10 perguntas em
  modo avião no S24+.
- **S21 (6 GB) com o vigente** **[PROJEÇÃO]**: o vigente (Nemotron + 1.2B tool) mede **3,86 GB PSS
  no S24+**; no S21 isso **aperta o LMK** (projeção pela tabela de RAM). A coexistência Nemotron+SLM
  no S21 é declarada inviável no `asr/README.md` (recomenda Whisper Base para o S21). Não há E2E
  medido no S21.
- **Aprovação E2E honesta continua baixa (\~19%)** **[MEDIÇÃO/LIMITE]**: o gargalo é **duplo** —
  retrieval domina (aprovação 17,4% com chunk-ouro vs 4,2% sem, 4,1×) e, dado o chunk certo, o 1.2B
  converte \~10× menos que o 2.6B. Fonte: `bench/regua/README.md` (resposta d). Não é limite da
  métrica (o 27B faz 84%).
- **Leitura fina de tabela (caso 13,8 kV)** **[MEDIÇÃO/LIMITE]**: **nenhum SLM (nem o 2.6B, nem o
  27B com o índice atual) lê a faixa exata** do Anexo II — o 27B só acerta com a tabela reconstruída
  injetada no oráculo. Fontes: `poc-completa-pre-limpeza:slm_oraculo/` (removido na limpeza), `sft_v2/`, `prompt_teto/`.
- **Latência de GPU não vale para o aparelho** **[LIMITE METODOLÓGICO]**: toda geração de síntese na
  RTX 5070/Spark serve para **qualidade**, não latência; a latência de campo é projetada pela
  tabela do device-bench. Declarado em `poc-completa-pre-limpeza:slm_oraculo/` e `poc-completa-pre-limpeza:finetune2/` (ambos removidos na limpeza), `bench/prompt_teto/`.

---

## Divergências entre o brief e as fontes

Registradas por ordem do brief ("se divergir da fonte, vale a fonte"). Nenhuma altera a arquitetura
vigente; são imprecisões de memória do brief.

1. **Ablação isolada do classificador (Decisão 3) — NÃO é lacuna.** O brief pedia para confirmar se
   existe uma comparação limpa com/sem classificador sobre a fala crua e, se não, marcar LACUNA. **Ela
   existe**: `classifier/README.md` (FASE C) mede, nas mesmas 151 e com a mesma fala bruta, **baseline
   sem boost R@2 13,9%** vs **confiança-gated R@2 29,8%** (+15,9 p.p.). É exatamente a ablação isolada
   procurada. → **Marcado como MEDIÇÃO, não lacuna.**
2. **Teto do 27B (Decisão 14): o brief diz 84,3%.** As fontes divergem levemente: `bench/prompt_teto`
   = **84,1%** (oráculo, `numeros`); `tools_oraculo`/`tools_2_6b` citam **84,3%**; `slm_oraculo`
   (removido na limpeza; `poc-completa-pre-limpeza:slm_oraculo/README.md`, teto **refeito com corpus corrigido**) = **82,1%**. Todos > 60% (a conclusão "régua passável" se
   mantém). O valor mais recente/corrigido é **82,1%**.
3. **Whisper Small (Decisão 9): o brief diz "RTF 2,37, \~14s".** A fonte `asr/README.md` mede **RTF
   2,81 (clean) / 2,99 (ruído), latência \~15,8 s**. A conclusão (descartado por lento) é a mesma; os
   números do brief estão imprecisos.
4. **Alucinação da quantização v3 (Decisão 8): o brief diz "v3 −0,6pp mas +6,7pp de alucinação".** A
   fonte confirma: os −0,6 p.p. são de **aprovação em oráculo**; os **+6,7 p.p.** são de **alucinação
   no contexto RECUPERADO** (retr aluc bf16 34,0 → Q4 40,7 em `sft_v3`), não a alucinação em oráculo
   (que cai −0,7). Precisado aqui para evitar leitura errada.
5. **Latência do Nemotron (Decisão 9): o brief diz "1,2s".** É o valor medido **no app** (`sft_v3`,
   \~1,17 s); a bancada `asr/README.md` mede **2,03 s** (áudio de \~6 s). Ambos citados.
6. **Recusa F1 90,5 (Parte A): é o bf16.** O brief cita "recusa F1 90,5" para o vigente; a fonte dá
   **bf16 90,5% e Q4 (embarcado) 85,4%**. O embarcado é Q4 → o F1 do que roda no aparelho é **85,4%**.
7. **Alucinação do híbrido (Parte A): o brief diz "32–34%".** A fonte `tools_oraculo` (M3, config A)
   dá **Q4 34,2% / bf16 31,8%** — consistente; o embarcado (Q4) é **34,2%**.

---

*Procedência: task poc-as-is-decisions, branch `fm/poc-as-is-decisions`. Todos os números conferidos
nos READMEs de origem citados por decisão. Comentários em PT-BR; código em inglês.*
