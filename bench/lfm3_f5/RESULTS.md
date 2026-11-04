# RESULTS — F5: alavancas de retrieval no v4 + COMPRESSÃO de contexto

Pergunta do capitão: depois do smoke (NO-GO para trocar de retriever, GO para as alavancas
no v4), (1) medir as 3 alavancas do report e (2) **provar ou refutar** que comprimir o
contexto entregue ao LFM com um "modelinho de relevância" (estilo SCR do MobileRAG+)
mantém ou melhora a resposta com **menos tokens**.

Réguas (trava anti-mascaramento do retrieval4: só conta o que sobe numa régua sem
derrubar a outra; tolerância 0,5 p.p.):
- **POC 151** (`corpus/qa_pairs_v2.jsonl`, `check_hit` oficial): R@k / MRR.
- **benchmark_mobile 1536** (1136 com ouro em norma + 400 em manual): hit / evrec / nDCG /
  MRR por offsets de evidência (definição do mobile_eval, harness do smoke).
- **Fim-a-fim** (Frente 3): LFM3 **f4-r32-ep2** no RECORTE de 344 (normas 10/18/34/13/31/33),
  régua honesta (juiz 27B), formato do app (turno tool com o contexto da célula).

Tabela-mestra completa: `data/consolidation.json`. Controle de todas as Δ = **`v4app`**
(v4 + override de NR explícita + gate fiel ao app).

## 0. Paridade e correções do espelho

| | 151 R@2 | 151 R@5 | norma hit@5 | norma hit@2 |
|---|--:|--:|--:|--:|
| `v4_anchor` (espelho antigo = âncora F3/F4) | **51,7** | **65,6** | 0,871 | 0,746 |
| smoke (report) | 51,7 | 65,6 | 0,874 | 0,743 |
| `v4app` (override portado, gate fiel ao app) | 52,3 | 66,2 | 0,879 | 0,762 |

- 151 bate exato. No bench o realinhamento reimplementado difere do smoke em ≤0,3 p.p.
- **Override de NR explícita PORTADO** para `bench/ctx_topk/retrieval_v4.py`
  (`detect_explicit_nr`, `explicit_override=True` por padrão = app; `False` reproduz a âncora).
- **2ª divergência espelho×app achada**: quando o classificador diz `nenhuma` em top-1, o
  espelho (`retrieval3/bm25.py::HybridBm25.rank`) descarta `nenhuma` e aplica filtro/boost
  na top-2; o app (`HybridRetriever.searchV3`) busca SEM boost. Efeito pequeno (+0,2 p.p.
  hit@2 norma); `retriever_f5.py` implementa o gate fiel ao app (`gate="app"`).

## 1. Frente 1 — gate do classificador

| variante | 151 R@2 | 151 R@5 | 151 MRR | norma hit@1 | hit@2 | hit@5 | evrec@2 | lay hit@5 | trava |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| (d) **v4app** (controle) | 52,3 | 66,2 | 0,501 | 0,622 | 0,762 | 0,879 | 0,746 | 0,845 | — |
| (a) limiar 0,6 | 52,3 | 66,9 | 0,499 | 0,623 | 0,764 | 0,881 | 0,748 | 0,845 | ✓ |
| (a) limiar 0,7 | 52,3 | 66,9 | 0,495 | 0,623 | 0,764 | 0,883 | 0,748 | 0,845 | ✓ |
| (a) limiar 0,8 | 51,0 | 66,2 | 0,487 | 0,623 | 0,764 | 0,884 | 0,748 | 0,845 | ✗ |
| (b) boost 2× | 52,3 | 66,2 | 0,498 | 0,637 | 0,771 | 0,882 | 0,755 | 0,845 | ✓ |
| (b) boost 1,5× | 51,0 | 66,2 | 0,491 | 0,645 | 0,779 | 0,887 | 0,762 | 0,852 | ✗ |
| (b) boost 1× | 45,7 | 64,2 | 0,465 | 0,637 | 0,788 | 0,893 | 0,770 | 0,863 | ✗ |
| **(a+b) limiar 0,7 + boost 2×** | **52,3** | **66,9** | 0,491 | **0,640** | **0,775** | **0,886** | **0,759** | 0,849 | **✓** |
| (c) só override, sem classificador | 41,1 | 60,3 | 0,412 | 0,640 | 0,791 | 0,898 | 0,774 | 0,866 | ✗ |

Leitura:
- **(c) é a melhor do benchmark e a pior da POC**: −11,2 p.p. de R@2 nas 151. É exatamente
  a "mascarada" do retrieval4 → **reprovada pela trava**.
- O classificador ajuda a POC no **filtro duro** e atrapalha o benchmark no **boost
  suave** (o benchmark quase nunca atinge o filtro duro: 30/1136 hard, 616 soft, 171 none).
  Por isso as duas alavancas são separáveis: limiar 0,7 tira os filtros duros duvidosos
  (61→33 na POC) sem perda de R@2; boost 2× suaviza o empurrão errado no benchmark.
- **Vencedor: limiar 0,7 + boost 2×**: benchmark norma +1,3 hit@2 / +0,7 hit@5 /
  +1,8 hit@1 / +1,3 evrec@2; POC R@2 igual e R@5 +0,7. Custo: MRR das 151 −0,010 (R@1
  39,7→37,7%, 3 queries). Ganho modesto, mas é o único que passa nas duas réguas. Não
  recupera os ~3 p.p. do `noclf`: o resto do gap vem da distribuição do classificador
  (top-1 de 9,6% no bench) e pede retreino com dados fora da POC, não ajuste de gate.

## 2. Frente 2 — vizinhos ±1 sobre o topK=2

Conjunto = top-2 + vizinhos id±1 da mesma norma (até 6 chunks, ~3× o contexto bruto):

| | 151 hit (conjunto) | norma hit (conjunto) | norma evrec (conjunto) |
|---|--:|--:|--:|
| v4app top-2 | 52,3 | 0,762 | 0,746 |
| v4app top-2 + ±1 | **56,3** (+4,0) | **0,807** (+4,5) | **0,801** (+5,5) |
| limiar 0,7 + boost 2× top-2 + ±1 | 56,3 | 0,819 | 0,812 |

Os vizinhos sobem as duas réguas (o teto do smoke era 0,788 contra 0,743 do v4 âncora).
Como pede o brief, isso **não** vai cru ao LFM: entra comprimido de volta ao orçamento na
Frente 3 (célula c).

## 3. Frente 3 — COMPRESSÃO (hipótese do capitão)

### 3.1 O compressor (`compress.py`)
- **Modelinho de relevância** = o MESMO EmbeddingGemma-300M do app (zero modelo novo
  no aparelho). Cada chunk é segmentado em **unidades normativas** (item `10.2.8.1`,
  alínea `a)`, frase, linha de tabela); 26.738 unidades nos 2202 chunks.
- **Janela de 3 unidades** (como o SCR, que usa granite-107m com janela de 3 frases)
  embedada com o prompt de documento. As janelas não dependem da query: no app viram um
  índice pré-computado (~26,7k × 768; ~80 MB fp32 ou ~20 MB int8), e a compressão custa só
  produtos internos sobre o embedding de query que o app **já calcula**.
- Score = cos(query, janela) + 0,3·sobreposição lexical (radical-6, como o app) + prior de
  rank 0,05/(r+1). Seleção gulosa até o orçamento (tokenizer do LFM3), render na ordem
  do documento com `[…]` nas lacunas.
- **Lição do trim.py tratada**: toda unidade carrega o **código do item** a que pertence;
  quando a unidade selecionada começa no meio de um item, recebe o prefixo `(item X)`. O
  cabeçalho de fonte (`doc - seção - título`) é mantido por chunk.

### 3.2 Proxy sem LLM (quanto do gabarito sobrevive no contexto)
`data/compress/proxy_v4app_h03p05.json` (ver também as variantes de scorer):

| célula | 151 tok | 151 fatos no ctx | norma tok | norma fatos | norma evrec |
|---|--:|--:|--:|--:|--:|
| a topK=2 cru | 1000 | 0,699 | 997 | 0,889 | 0,746 |
| b topK=2 → 50% | 452 | 0,520 | 452 | 0,824 | 0,693 |
| c top2+±1 → 100% | 917 | 0,662 | 912 | 0,893 | 0,764 |
| d topK=5 → 50% | 459 | 0,491 | 457 | 0,869 | 0,765 |
| d topK=5 → 70% | 642 | 0,578 | 643 | 0,900 | 0,804 |

O proxy já mostrava a assimetria entre réguas: no benchmark (perguntas que citam o item)
comprimir top-5 a 50% **iguala** a evidência do top-2 cru com metade dos tokens; nas 151
coloquiais a compressão perde fatos (os chunks da POC são densos em itens e a fala não
casa as frases).

### 3.3 Fim-a-fim — LFM3 f4-r32-ep2, recorte 344, juiz 27B
`data/e2e/summary_f4-r32-ep2.json` (+ réplica `-rep2` das 3 células decisivas). TTFT
estimado = tokens de prompt ÷ 116 t/s (prefill do 1.2B Q4_0 no S24+ escalado pelos
parâmetros: é ESTIMATIVA, o LFM3 não foi medido no aparelho; a relação entre células não
depende da constante).

| célula | aprov. | cobertura | alucin. | recusa | cita NR | **cita item-ouro** | tok ctx | tok prompt | TTFT~ | ganhos/perdas vs a |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **a** topK=2 cru (app) | 55,5 / 55,8 | 0,622 | 14,0 | 24,7 | 99,4 | 65,2 | 983 | 1635 | 14,1 s | — |
| b topK=2 → 50% | 52,3 | 0,588 | 12,2 | 29,9 | 99,7 | 67,0 | 444 | 1096 | 9,5 s | 20/31 |
| b topK=2 → 70% | 53,2 | 0,594 | 12,2 | 27,6 | 100 | 66,5 | 616 | 1268 | 11,0 s | 20/28 |
| c top2+±1 → 70% | 53,8 | 0,620 | 15,1 | 24,4 | 100 | 70,1 | 627 | 1279 | 11,1 s | 26/32 |
| **c** top2+±1 → mesmo orçamento de a | 58,4 / 56,4 | 0,639 | 12,5 | 22,4 | 100 | **71,4** | 895 | 1547 | 13,4 s | 32/22 |
| **d** topK=5 → 50% | **57,6 / 57,3** | 0,632 | 12,8 | 25,9 | 99,4 | **73,2** | **448** | **1100** | **9,5 s** | 37/30 |
| d topK=5 → 70% | 55,8 | 0,624 | 12,2 | 25,9 | 99,7 | 70,1 | 630 | 1282 | 11,1 s | 38/37 |

(aprov. "x / y" = rodada 1 / réplica; média a 55,6 · c 57,4 · d50 57,5.)

### 3.4 Veredito da hipótese
- **Hipótese PROVADA na forma (d): o mesmo recall com metade do contexto.** Buscar mais
  (top-5) e comprimir a **50% dos tokens** do controle dá **aprovação ≥ controle** nas
  duas rodadas (+1,9 p.p. na média), **alucinação menor** (12,5 vs 13,8%), **citação do
  item certo MAIOR** (73,2 vs 65,2%, +8 p.p.; NR 99%+ em tudo) e **−535 tokens de prompt
  (−33%, TTFT estimado 14,1 → 9,5 s)**. O ganho de aprovação está dentro do ruído (teste
  do sinal pareado p≈0,46). O que se sustenta é: **não piora a resposta e corta 1/3 do
  prompt**.
- **(c) vizinhos comprimidos ao mesmo orçamento** também ≥ controle (média +1,8 p.p.) com
  a maior cobertura e citação de item 71%, mas economiza pouco (−88 tok). Serve como
  "mais recall, mesma janela".
- **Comprimir só o top-2 (b) PIORA**: −2,3 a −3,2 p.p. e recusa sobe +3 a +5 p.p. Cortar
  o que já é pouco remove contexto útil e o modelo passa a delimitar ("a norma não
  detalha"). A compressão compensa quando o orçamento liberado é **reinvestido em
  recall** (mais chunks), não quando só encolhe o prompt.
- **A lição do trim.py foi resolvida**: nenhuma célula comprimida piora a citação. Todas
  citam o item-ouro **mais** que o cru (66,5–73,2 vs 65,2%), porque o `(item X)` ancora o
  id junto da frase e a densidade aumenta.
- **Ressalva de régua**: o proxy da POC 151 (falas coloquiais) mostra perda de fatos no
  contexto comprimido (0,70 → 0,49 em d50). O fim-a-fim foi medido no recorte do
  benchmark, não nas 151. Antes de embarcar, rodar a célula d50 nas 151 (régua honesta,
  `bench/regua`) para fechar a trava dupla também na síntese.

## 4. Frente 4 — manuais como aba separada no RRF

9 manuais → `cache/manuals.db` (FTS5 próprio, 1092 chunks de ~1700 chars, offsets nativos)
+ denso próprio (EmbeddingGemma), 2 sinais extras no RRF com peso `w` (normas = 2/1/2).
**Nunca** no BM25 das normas.

| variante | 151 R@2 | 151 R@5 | norma hit@2 | norma hit@5 | **manual hit@5** (400) | manual content@5 | todas hit@5 |
|---|--:|--:|--:|--:|--:|--:|--:|
| sem manuais (v4app) | 52,3 | 66,2 | 0,762 | 0,879 | **0,000** | 0,380 | 0,650 |
| w=1 | 52,3 | 66,2 | 0,762 | 0,879 | 0,005 | 0,388 | 0,652 |
| **w=1,25** | 52,3 | 66,2 | 0,762 | 0,877 | **0,220** | 0,540 | 0,706 |
| w=1,5 | 52,3 | 66,2 | 0,761 | 0,868 | 0,472 | 0,708 | 0,765 |
| w=1,75 | 51,0 | 66,2 | 0,753 | 0,863 | 0,647 | 0,825 | 0,807 |
| w=2 | 50,3 | 62,9 | 0,730 | 0,847 | 0,782 | 0,912 | 0,830 |
| w=3 | 33,1 | 53,6 | 0,408 | 0,732 | 0,887 | 0,953 | 0,773 |
| w=2, gate por margem densa | 52,3 | 65,6 | 0,754 | 0,868 | 0,540 | 0,805 | 0,783 |
| w=2, gate por intenção (regex) | 52,3 | 66,2 | 0,762 | 0,879 | 0,033 | 0,412 | 0,659 |

- **O hit de manual sai de 0 por construção para 22–78%**, mas é um **trade-off
  monotônico**: cada ponto de manual custa norma. A 1,25 a diluição das normas é
  desprezível (−0,2 p.p. hit@5, 151 intacto) com +22 p.p. em manual (+5,6 p.p. nas 1536).
  A 1,5 custa −1,1 p.p. de norma por +47 p.p. de manual. A 2 a trava quebra (151 R@5 −3,3).
- **O gate por intenção via regex NÃO funciona**: as perguntas de manual soam como de
  norma (só 17/400 disparam "manual/guia/orientação/exemplo"). O gate por **margem densa**
  (manual mais similar que a norma) é o melhor compromisso a w=2: manual 54%, norma −1,1,
  151 R@2 intacto e R@5 −0,6, no limite da tolerância.
- **Decisão de produto**, não técnica: se o cliente valoriza manuais, `w=1,25` é "grátis"
  e `margem w=2` é o ponto de equilíbrio. Os 26% de queries de manual do benchmark são o
  argumento. Para manual alto sem custo de norma, o próximo passo é um classificador de
  intenção treinado (norma×manual), não regex.

## 5. Recomendação (o que embarca)

| # | alavanca | embarca? | ganho medido | custo |
|---|---|---|---|---|
| 1 | **Gate: limiar 0,7 + boost 2×** (`HybridRetriever` confThreshold 0.5→0.7, boostFactor 5→2) | **SIM** | bench norma +1,3 hit@2 / +0,7 hit@5, 151 R@5 +0,7, R@2 = | 2 constantes; MRR 151 −0,01 |
| 2 | Alinhar o espelho Python ao app (override + `nenhuma` sem boost) | **SIM (feito no espelho)** | âncora de treino = app | dados F3/F4 futuros mudam pouco |
| 3 | **Compressão d50: top-5 + compressão SCR-Gemma a 50%** | **SIM, depois de validar nas 151** | aprov ≥ controle (+1,9), citação de item +8 p.p., alucinação −1,3 | índice de janelas (~20 MB int8) + 5 chunks na busca; **−33% do prompt (≈ −4,6 s de TTFT estimado)** |
| 4 | Vizinhos ±1 | só **junto** de compressão (célula c) | conjunto +4,5 p.p. de hit | sem compressão triplica o contexto |
| 5 | Manuais no RRF | **decisão de produto** (w=1,25 grátis; margem w=2 equilíbrio) | manual 0 → 22–54% | −0,2 a −1,1 p.p. de norma |
| ✗ | Sem classificador (só override) | **NÃO** | melhor no bench | −11 p.p. R@2 nas 151 (mascarada) |
| ✗ | Comprimir só o top-2 | **NÃO** | — | −2 a −3 p.p. de aprovação, recusa +3 a +5 |

## Reprodução

```bash
make -C bench/lfm3_f5 align manuals units retrieval proxy   # local (classifier/.venv + :8399)
make -C bench/lfm3_f5 e2e-prompts                            # data/e2e/prompts.jsonl
# Spark (setsid nohup; BATCH 8 — batch 48 estourou a memória unificada da GB10):
#   python gen_e2e.py --model <merged data2x-ep4> --adapter <f4 checkpoint-284> --tag f4-r32-ep2 --batch 8 --max-new 768
make -C bench/lfm3_f5 score TAG=f4-r32-ep2 consolidate
```

Artefatos: `data/retrieval/*.json` (agregados por variante), `data/compress/proxy_*.json`,
`data/e2e/{prompts,gen,scored,summary}_*`, `data/consolidation.json`. Caches reprodutíveis
(gitignored): `cache/{align.json,units.json,units_win3.npz,manuals.db,manuals_dense.npz,qemb.npz,rank_*.jsonl}`.
Artefatos externos (não versionados, md5 iguais ao smoke): `classifier/models/classic_winner.pkl`
`e921aee3…`, `bench/ctx_topk/data/dense_text.bin` `0e6cadfe…`, `dense_exponly.bin` `782c1130…`;
merged data2x-ep4 `model.safetensors` `2b2aa2fa…`; adapter f4-r32-ep2 (checkpoint-284)
`adapter_model.safetensors` `18a40d6c…`.
