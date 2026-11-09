# Smoke de recuperação: retriever v4 da POC nas 1536 queries de NR do benchmark_mobile

**Task:** `lfm3-retrieval-smoke` (scout, sem branch/PR) · **Data:** 2026-11-03 · **Custo real:** ~2 h de agente; cada rodada leva ~1 min (CPU + GPU local para o encoder de 300M).
**Pergunta do capitão:** vale abrir uma fase dedicada "MobileRAG+ vs v4" para comparar ou trocar o retriever?

## TL;DR (veredito)

**NO-GO para uma fase dedicada de comparação/troca de retriever. GO para 3 correções baratas e localizadas no v4** (ver §7).

1. **O harness está validado contra os números publicados.** Rodando o mesmo código de métrica sobre os chunks deles (`corpus.sqlite` do `mobilerag-plus-assets@5ca7668`), reproduzo **exatamente** o BM25 publicado (hit 0,705 / nDCG@5 0,591 / MRR@5 0,563 nas 2508 queries) e o Naive-RAG exato do Round 2 (0,733 / 0,608 / 0,579). Logo as métricas abaixo seguem a mesma definição das deles.
2. **Nas 1136 queries cujo gold é texto de NORMA** (a parte que o nosso índice cobre), **o v4 como está no app** tem **hit@5 0,874 · hit@2 0,743 · hit@1 0,597 · evidence_recall@5 0,859 · nDCG@5 0,740 · MRR@5 0,706**. No corpus equalizado (só os 36 textos de norma, chunks deles), o melhor primeiro estágio do MobileRAG+, que é o **Qwen3-Embedding-0.6B fine-tuned com busca exata**, faz **0,804 / 0,690 / 0,549 / 0,795 / 0,681 / 0,651**. **O v4 está à frente em todas as métricas e em todos os vocabulários**, inclusive no `lay` (0,845 vs 0,768), onde BM25 puro despenca (0,447).
3. **Nas 1536 completas o v4 cai para hit@5 0,646.** Isso vem do escopo, não da qualidade: **400 queries (26%) têm gold em MANUAL** e o índice do app não indexa manuais (decisão anterior da POC: manuais poluem o BM25). Nessas o `source_id` nunca bate, então hit = 0 por construção. Pelo conteúdo (o trecho recuperado contém ≥70% das palavras da evidência), o v4 acerta 31% delas, contra um teto de 38% para o que existe no nosso índice.
4. **O maior defeito do v4 que o benchmark revelou é o classificador-gate, não a busca.** Nas 1536 o classificador NR acerta **top-1 em só 9,6%** (no holdout de 151 da POC acerta 51%). Como dá boost na NR errada, ele **tira 2,9 p.p. de hit@5 e 4,0 p.p. de hit@2** do v4 nas queries de norma. Ele também **ignora "Conforme a NR-33"** escrito na pergunta. O app tem um override de NR explícita (`HybridRetriever.kt:99-104`) que o espelho Python usado nos treinos F3/F4 não reproduz.
5. **O ganho real do MobileRAG+ está no pós-recuperação (vizinhos ±2 + SCR), não no primeiro estágio.** Isso é complementar ao v4 e barato de testar em cima dele. Por exemplo: em 51 das 292 falhas de top-2 do v4, o chunk-ouro é **vizinho imediato (±1)** de um chunk recuperado.

## 1. O que foi feito

| Etapa | Detalhe |
|---|---|
| Dados | `UFGCEMIGONA/benchmark_mobile`, snapshot `5a5aa32ae4087481be828e071aff169827ec2993`, `normas_regulamentadoras/train-00000-of-00001.parquet`: 1536 linhas = 384 respostas × 4 vocabulários (clean/technical/operational/lay). 45 `source_file`, sendo **1136 queries com gold em norma (36 arquivos) e 400 em manual (9 arquivos)**. Os offsets de evidência batem **2016/2016** com `nrs.parquet` (`conteudo`/`manual`), o mesmo texto-fonte do nosso índice. |
| Retriever | **v4 exatamente como no repo/app**: `bench/ctx_topk/retrieval_v4.py::RetrieverV4`, o mesmo módulo que `bench/lfm3_f3/retrieve_f3.py` e `bench/lfm3_f4/retriever_lazy.py` usam como âncora dos treinos F3/F4. Composição: BM25 gated-expandido (índice `android/app/src/main/assets/index.db`, md5 `e90e94bd…`, 2202 chunks, 36 NRs, pesos 1.5/3/2/1/1, classificador `classic_winner.pkl` md5 `e921aee3…`) + denso EmbeddingGemma-300M-qat-Q4_0 texto e exp-only (bins DVEC1 `dense_text.bin` md5 `0e6cadfe…`, `dense_exponly.bin` md5 `782c1130…`), fundidos por RRF k=10 w=(2,1,2). Top-5 com a fala crua. |
| Paridade | Mesmo setup nas 151 da POC: **R@2 51,7% / R@5 65,6%**, exatamente a régua publicada do v4 (`retrieval4/README.md`). |
| Ablações (mesmo índice) | `v4app` = v4 + override de NR explícita como no `HybridRetriever.kt` · `v4noclf` = v4 sem classificador (BM25 só com `app_fts_query`) · `bm25g` = só o sinal BM25-gated · `bm25raw` = FTS5 puro do índice v4 · `dense_text` / `dense_exp` = cada sinal denso sozinho. |
| Validação do harness | Baixei o `corpus.sqlite` deles (3764 chunks, vetores Qwen armazenados) e o `qwen3-embedding-0.6b-v2-q8_0.gguf` (llama-server, pooling last, prompt `Instruct: …\nQuery:` do `mobilerag-plus.json`). Reimplementei o `score_row`/`overlap`/`ndcg` deles (`mobile_rag/tests/user/benchmark_mobile/benchmark_mobile_test.py@4c4d59c`, linhas 174-214 e 332-384) e o `mobile_rag/lexical/bm25.py`. Resultado na tabela §3.1. |

### 1.1 Critério de relevância (registrado)

Uso **o mesmo critério do mobile_eval**, adaptado aos nossos chunks:

- **Os chunks do nosso índice não têm offsets.** O chunker junta linhas, colapsa espaços, tira anotações de portaria e põe o prefixo `NR-XX · seção título — `. Por isso cada chunk foi **realinhado ao texto-fonte**: tokens `\w+` sem acento e minúsculos, âncora de 6-grama, depois `difflib.SequenceMatcher` numa janela. O chunk vira a **união dos blocos casados** (≥3 tokens, lacunas ≤8 tokens fundidas). Resultado: 2202/2202 chunks alinhados, cobertura de tokens p5 = 1,00, só 9 chunks abaixo de 0,9 (tabelas da NR-15/NR-33).
- **chunk relevante** ⇔ `chunk.source_file == row.source_file` **E** o span do chunk sobrepõe ≥1 caractere da evidência.
- **gain(chunk)** = caracteres de evidência cobertos ÷ caracteres de evidência (união de todas as citações da linha).
- **hit@k** = algum dos k primeiros é relevante. É o `hit` deles com k=5, que equivale ao "source_id correto presente + sobreposição"; só bater o arquivo aparece separado como `hit_src`.
- **evidence_recall@k** = caracteres de evidência cobertos pela união dos spans dos k primeiros ÷ total. O nosso é pré-orçamento (sem SCR e sem corte por tokens). O deles é medido **depois** do SCR e do corte para 1664 tokens. Ver caveats.
- **nDCG@5** = DCG(gains) / DCG(5 melhores gains de todos os chunks do arquivo-ouro). É a definição deles.
- **MRR@5** = 1/rank do primeiro relevante.
- **Critério secundário `content_hit`** (não oficial): o trecho recuperado contém ≥70% dos tokens de alguma evidência. Uso só para as queries de manual e para checar o alinhamento.

**Artefato conhecido do alinhamento:** 32 das 1136 queries de norma não têm nenhum chunk nosso que sobreponha a evidência por offset. Em todas as 32, porém, existe um chunk com ≥70% do conteúdo, e o v4 recupera 30 delas. São falhas de alinhamento em tabelas, como a NR-15 Anexo 6 de mergulho. Por isso **o hit@5 de norma do v4 está subestimado em até ~2,6 p.p.** (0,874 por offset, ~0,90 corrigido, 0,937 pelo critério de conteúdo).

## 2. Resultados do v4 (nosso índice, fala crua, top-5)

### 2.1 v4 como está (âncora F3/F4), por vocabulário

**Subconjunto norma (1136 queries cujo gold está no que indexamos):**

| vocab | n | hit@1 | hit@2 | hit@5 | evidence_recall@5 | ev. cobre 100% @5 | nDCG@5 | MRR@5 | NR certa @5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 284 | 0,634 | 0,785 | 0,901 | 0,887 | 0,694 | 0,775 | 0,742 | 0,972 |
| technical | 284 | 0,651 | 0,792 | 0,891 | 0,877 | 0,683 | 0,778 | 0,749 | 0,986 |
| operational | 284 | 0,546 | 0,690 | 0,859 | 0,841 | 0,665 | 0,703 | 0,663 | 0,951 |
| lay | 284 | 0,556 | 0,704 | 0,845 | 0,830 | 0,651 | 0,705 | 0,670 | 0,958 |
| **agregado** | **1136** | **0,597** | **0,743** | **0,874** | **0,859** | 0,673 | **0,740** | **0,706** | 0,967 |

**Todas as 1536 (inclui as 400 de manual, que têm hit = 0 por construção):**

| vocab | n | hit@1 | hit@2 | hit@5 | evidence_recall@5 | nDCG@5 | MRR@5 | NR certa @5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 384 | 0,469 | 0,581 | 0,667 | 0,656 | 0,573 | 0,549 | 0,935 |
| technical | 384 | 0,482 | 0,586 | 0,659 | 0,649 | 0,575 | 0,554 | 0,951 |
| operational | 384 | 0,404 | 0,510 | 0,635 | 0,622 | 0,520 | 0,491 | 0,901 |
| lay | 384 | 0,411 | 0,521 | 0,625 | 0,614 | 0,522 | 0,496 | 0,909 |
| **agregado** | **1536** | **0,441** | **0,549** | **0,646** | **0,635** | **0,548** | **0,522** | 0,924 |

evidence_recall@2 (o que o app manda com topK=2): **norma 0,725** (clean 0,764 · technical 0,769 · operational 0,677 · lay 0,690); todas as 1536: 0,536.
Por tipo de pergunta (norma, hit@5 / evrec@5): factual 0,860/0,847 · procedural 0,812/0,804 · conditional 0,906/0,894 · multievidence 0,840/0,802.
**Manual (400):** hit por `source_id` = 0 em todas as variantes. Pela NR (arquivo de norma da mesma NR no top-5): 0,802. Pelo conteúdo (`content_hit`): @2 0,273, @5 0,310, com teto de 0,380 (só 38% dessas evidências existem literalmente em algum chunk de norma).

### 2.2 Ablações no mesmo índice (agregados)

| variante | subset | hit@1 | hit@2 | hit@5 | evrec@2 | evrec@5 | nDCG@5 | MRR@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **v4 (como está)** | norma 1136 | 0,597 | 0,743 | 0,874 | 0,725 | 0,859 | 0,740 | 0,706 |
| v4app (+override NR explícita, = app) | norma | 0,621 | 0,755 | 0,883 | 0,738 | 0,868 | 0,756 | 0,724 |
| **v4noclf (sem classificador)** | norma | **0,626** | **0,783** | **0,903** | **0,765** | **0,889** | **0,772** | **0,739** |
| bm25g (só sinal BM25-gated) | norma | 0,306 | 0,446 | 0,674 | 0,437 | 0,666 | 0,495 | 0,437 |
| bm25raw (FTS5 puro do índice v4) | norma | 0,638 | 0,773 | 0,882 | 0,756 | 0,870 | 0,768 | 0,736 |
| dense_text (Gemma texto+exp) | norma | 0,594 | 0,727 | 0,836 | 0,711 | 0,821 | 0,720 | 0,691 |
| dense_exp (Gemma exp-only) | norma | 0,447 | 0,589 | 0,724 | 0,573 | 0,707 | 0,589 | 0,556 |
| v4 | todas 1536 | 0,441 | 0,549 | 0,646 | 0,536 | 0,635 | 0,548 | 0,522 |
| v4app | todas | 0,459 | 0,559 | 0,653 | 0,546 | 0,642 | 0,559 | 0,536 |
| v4noclf | todas | 0,463 | 0,579 | 0,668 | 0,565 | 0,657 | 0,571 | 0,547 |

Por vocabulário (norma, hit@5):

| variante | clean | technical | operational | lay |
|---|---:|---:|---:|---:|
| v4 | 0,901 | 0,891 | 0,859 | 0,845 |
| v4app | 0,923 | 0,905 | 0,859 | 0,845 |
| v4noclf | 0,937 | 0,908 | 0,894 | 0,873 |
| bm25raw | 0,937 | 0,915 | 0,852 | 0,824 |
| dense_text | 0,884 | 0,877 | 0,803 | 0,782 |
| dense_exp | 0,718 | 0,658 | 0,750 | 0,768 |

Leitura:

- **A expansão (v3/v4) cumpre o papel.** Com expansão, o BM25 puro do índice v4 já perde só 11 p.p. de clean para lay (0,937 → 0,824). O BM25 deles, sem expansão, perde 49 p.p. no corpus equalizado (0,933 → 0,447). O denso exp-only é o único sinal que **melhora** de clean para lay (0,718 → 0,768), exatamente como desenhado.
- **O classificador atrapalha neste benchmark.** Top-1 9,6% e top-2 14,7% nas 1536. Distribuição do top-1: `nenhuma` 323, nr-01 243, nr-16 228, nr-26 209. Ele colapsa em poucas classes porque o vocabulário das perguntas sintéticas do benchmark (com "Conforme a NR-xx", cenários longos e hesitações) está fora da distribuição de treino. Mesmo "Conforme a NR-10, qual a distância" vira `nenhuma`.
- **Mas no holdout da POC o classificador ajuda muito.** Sem ele, as 151 caem de R@2 51,7 → 40,4 e de R@5 65,6 → 59,6. O override de NR explícita é neutro/positivo nos dois benchmarks (151: 52,3/66,2; benchmark: +0,9 p.p. de hit@5 em norma). **Conclusão: o gate está sobreajustado à distribuição das 151.** É o mesmo padrão de "mascarada" que a POC já documentou na Frente A do `retrieval4`.

## 3. Comparação com o mobile_eval publicado

Fonte: `github.com/vinyteles/cemig-mobile`, branch `mobile_eval` @ `4c4d59c8adf09bfc17e92ca699d7e38d594f4be3`, arquivo `evaluation/BENCHMARK_MOBILE_RESULTS.md` (o repo é privado, acessado via `gh api`). O `main` só tem `evaluation/MOBILE_DEVICE_RESULTS.md` (latência no S25 FE, sem métricas de recuperação novas).

### 3.1 Tabela deles (copiada), 2508 queries = 1536 NR + 972 documentos internos, corpus de 99 textos em 3764 chunks de ≤2000 caracteres

**Round 1 (DGX Spark):**

| Pipeline | Judge score | Correct | Partially correct | Incorrect | hit | nDCG@5 | MRR@5 | Evidence recall | Prompt tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive-RAG | 0.677 | 0.624 | 0.108 | 0.269 | 0.697 | 0.579 | 0.553 | 0.585 | 1,363 |
| BM25 | 0.635 | 0.598 | 0.073 | 0.328 | 0.705 | 0.591 | 0.563 | 0.595 | 1,364 |
| MobileRAG | 0.669 | 0.596 | 0.146 | 0.258 | 0.697 | 0.603 | 0.585 | 0.582 | 1,142 |
| **MobileRAG+** | **0.713** | **0.641** | 0.144 | **0.215** | **0.777** | **0.635** | **0.608** | **0.647** | 1,556 |

**Round 2 (h100-eliomar, Naive-RAG com busca exata):**

| Pipeline | Judge score | Correct | hit | nDCG@5 | MRR@5 | Evidence recall | Context tokens | Retrieval | First token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive-RAG (exact) | 0.710 | 0.660 | 0.733 | 0.608 | 0.579 | 0.613 | 1,205 | 0.22 s | 3.30 s |
| BM25 | 0.638 | 0.603 | 0.705 | 0.591 | 0.563 | 0.595 | 1,209 | 0.23 s | 3.27 s |
| MobileRAG | 0.668 | 0.600 | 0.697 | 0.603 | 0.585 | 0.582 | 987 | 0.63 s | 3.22 s |
| MobileRAG+ | 0.710 | 0.641 | 0.777 | 0.635 | 0.608 | 0.647 | 1,401 | 1.84 s | 5.28 s |

Por configuração, NR: evidence recall Naive 0.582 · BM25 0.598 · MobileRAG 0.578 · **MobileRAG+ 0.640** (Round 1).
Hit por vocabulário (as 2508, Round 1):

| Pipeline | clean | technical | operational | lay |
|---|---:|---:|---:|---:|
| Naive-RAG | 0.73 | 0.70 | 0.71 | 0.64 |
| BM25 | 0.88 | 0.85 | 0.70 | 0.40 |
| MobileRAG | 0.73 | 0.70 | 0.71 | 0.64 |
| MobileRAG+ | 0.81 | 0.78 | 0.79 | 0.72 |

Componentes: primeiro estágio = EcoVector (Qwen3-Embedding-0.6B-v2 fine-tuned, Q8_0 de 639 MB, 1024-d, k-means + HNSW, 6 de 19 clusters); MobileRAG+ = **vizinhos ±2 de cada hit (~20 chunks)** + SCR com granite-107m (janela de 3 frases); leitor Energy-Mobile int8 com contexto de 2048 tokens.

### 3.2 Reprodução da tabela deles com o meu harness (prova de que as métricas coincidem)

| (2508 queries, corpus deles) | hit@5 | nDCG@5 | MRR@5 |
|---|---:|---:|---:|
| BM25 publicado | 0.705 | 0.591 | 0.563 |
| **BM25 reproduzido** (cópia do `bm25.py` + `score_row`) | **0.705** | **0.591** | **0.563** |
| Naive-RAG exato publicado (Round 2) | 0.733 | 0.608 | 0.579 |
| **Naive exato reproduzido** (vetores deles + Qwen Q8_0 GGUF na query) | **0.733** | **0.608** | **0.579** |

O evidence_recall deles é pós-SCR/orçamento, então não reproduzi o número (o meu "top-5 inteiro" dá 0,694 no BM25 e 0,719 no Naive, acima dos 0,595/0,613 deles porque não corta pelo orçamento).

### 3.3 A comparação mais próxima de maçã-com-maçã possível sem fase dedicada

Subconjunto **1136 queries com gold em norma**. Todas as linhas usam o mesmo código de métrica e o mesmo texto-fonte (`nrs.parquet`). Muda o chunking (o nosso é estrutural por item de NR, ~1500 caracteres; o deles corta a cada ≤2000 caracteres) e o corpus de distratores.

| Retriever | Corpus de busca | hit@1 | hit@2 | hit@5 | evrec@5 | nDCG@5 | MRR@5 | lay hit@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **v4 (como está, âncora F3/F4)** | nosso índice, 36 normas, 2202 chunks | 0,597 | 0,743 | 0,874 | 0,859 | 0,740 | 0,706 | 0,845 |
| v4app (app real, override NR) | idem | 0,621 | 0,755 | 0,883 | 0,868 | 0,756 | 0,724 | 0,845 |
| v4noclf | idem | 0,626 | 0,783 | 0,903 | 0,889 | 0,772 | 0,739 | 0,873 |
| Qwen-FT exato (= Naive R2) | deles completo (3764: normas + manuais + internos) | 0,480 | 0,628 | 0,754 | 0,739 | 0,620 | 0,589 | 0,711 |
| BM25 deles | deles completo | 0,463 | 0,599 | 0,703 | 0,691 | 0,587 | 0,561 | 0,380 |
| **Qwen-FT exato** | **deles, só os 36 textos de norma (1765 chunks)** | 0,549 | 0,690 | 0,804 | 0,795 | 0,681 | 0,651 | 0,768 |
| BM25 deles | deles, só normas | 0,550 | 0,680 | 0,762 | 0,752 | 0,661 | 0,639 | 0,447 |
| EmbeddingGemma (nosso encoder) exato | deles, só normas | 0,525 | 0,656 | 0,784 | 0,772 | 0,656 | 0,626 | 0,736 |
| RRF(Qwen-FT, BM25 c/ query do app) k=10 w=1,2 | deles, só normas | 0,605 | 0,755 | 0,861 | 0,850 | 0,740 | 0,710 | 0,694 |
| RRF(Qwen-FT, Gemma, BM25-appq) w=1,1,2 | deles, só normas | 0,619 | 0,776 | 0,901 | 0,891 | 0,767 | 0,733 | 0,792 |

O que dá para dizer com honestidade:

- **No primeiro estágio, o v4 não perde para o retriever do MobileRAG+.** Contra o Qwen-FT exato no corpus equalizado de normas, o v4 está **+7,0 p.p. em hit@5, +5,3 em hit@2, +6,4 em evrec@5 e +5,9 em nDCG@5**, e ganha nos quatro vocabulários. Contra o Qwen-FT no corpus completo deles a vantagem aparente (+12 p.p.) é inflada pelos distratores de manual e de documentos internos que o corpus deles tem e o nosso não. **Não concluo equivalência nem superioridade estrita**, porque os chunks e os distratores diferem. O sinal, porém, é consistente em todas as métricas.
- **Um híbrido "à la v4" construído sobre os chunks deles** (RRF de 2 densos + BM25 com o `app_fts_query`, sem expansão e sem classificador) chega a hit@5 0,901, praticamente o mesmo que o v4noclf no nosso índice (0,903). Duas leituras: (a) o ganho vem da **fusão híbrida**, que é exatamente o "next step 1" do próprio relatório deles; (b) o Qwen-FT soma pouco a um híbrido que já tem Gemma (RRF Gemma+BM25 sozinho dá 0,862, e somar o Qwen leva a 0,901, +3,9 p.p.), a um custo de **+639 MB e ~556 ms por query no S25 FE**.
- **O MobileRAG+ compete em outro eixo**, o pós-recuperação: vizinhos e SCR. O "hit 0.777" deles olha ~20 chunks (os próprios autores dizem que não é comparável) e o evidence_recall é pós-orçamento. Nenhuma das duas coisas foi medida aqui do nosso lado.

### 3.4 Caveats (o que impede declarar equivalência)

1. **Corpus distinto.** Nosso índice = 36 normas (2202 chunks, sem manuais, com campo de expansão gerado pelo 27B). Corpus deles = 99 textos (normas + 9 manuais + documentos internos; 3764 chunks). Nas 1536, 26% das queries têm gold em manual que nunca recuperaríamos pelo `source_id`.
2. **Chunking próprio.** O nosso é estrutural (item/seção da NR, texto normalizado, prefixo de cabeçalho), sem offsets: foi preciso realinhar, com um erro estimado de ≤2,6 p.p. de hit em norma (§1.1). O deles corta a ≤2000 caracteres com offsets exatos. Chunks maiores sobem o hit por construção; os nossos têm média de 1488 e mediana de 1615 caracteres, contra mediana de ~1800 deles, então o tamanho é próximo mas não igual.
3. **Métricas com o mesmo nome e definição diferente.** `hit` do MobileRAG+ olha ~20 chunks. `evidence_recall` deles é pós-SCR + orçamento de 1664 tokens com o tokenizer do Energy-Mobile; o nosso é top-k inteiro, pré-orçamento. `nDCG`/`MRR`/`hit` (exceto MobileRAG+) têm a mesma definição, validada em §3.2.
4. **Queries.** Os números publicados somam NR + documentos internos (2508); os meus são só NR (1536/1136). O relatório deles só quebra judge score e evidence recall por configuração.
5. **O classificador do v4** foi treinado em falas sintéticas da POC e não viu este benchmark. A expansão do índice é gerada por chunk, sem ver queries; não há contaminação conhecida, mas também não houve calibração para este benchmark.
6. **Encoder de query.** O deles usa o Q8_0 GGUF no host; o meu reproduz isso de forma idêntica (os números de §3.2 batem). No aparelho a paridade deles é 21/30 no top-5 (`MOBILE_DEVICE_RESULTS.md`).

### 3.5 O que precisaria ser igualado numa fase dedicada (se um dia for aberta)

- **Mesmo corpus de busca**: decidir se entram manuais e documentos internos; indexar os dois lados sobre o mesmo conjunto de textos.
- **Mesmas fronteiras de chunk**, ou ao menos chunks com offsets nativos no nosso pipeline (eliminar o realinhamento).
- **Mesmo orçamento e mesmo leitor** para o evidence_recall pós-orçamento: 2048 tokens, tokenizer do LFM vs Energy-Mobile, topK=2 do app vs top-5 deles.
- **Mesma janela de candidatos** (top-5 vs ~20 com vizinhos) e SCR ligado/desligado nos dois lados.
- **Mesmo aparelho** para latência e RAM (S24+ vs S25 FE, ambos Exynos 2400) e o custo do embedder (Gemma 265 MB vs Qwen 639 MB).
- **Juiz**: o deles é o gpt-oss-120b; a régua honesta da POC usa o 27B.

## 4. Cinco falhas típicas do v4

IDs = `id` do benchmark. "Veio" = top-3 do v4 (doc · seção · título no nosso índice).

1. **Classificador anula a NR citada na própria pergunta** (`e1ae09b572716c0128641d00`, clean/factual)
   - Q: *"Quais são as medidas de proteção obrigatórias para processos ou equipamentos que possam gerar ou acumular eletricidade estática, conforme a NR-10?"*
   - Classificador: `nr-01`/`nr-18` (soft boost). **Veio:** nr-20·20.12 Capacitação · nr-10·10.1 Objetivo · nr-18·18.6.5 Condutores. **Devia:** nr-10·10.8 (chunk que contém *"10.9.3 Os processos ou equipamentos susceptíveis de gerar ou acumular eletricidade estática devem dispor de proteção específica…"*).
   - O v4app (override de NR explícita, que o app tem e o espelho Python não) e o v4noclf acertam.
2. **Boost na NR errada por sobreposição semântica entre normas** (`7d4a80ddf3506ded8fa05373`, lay/conditional)
   - Q: *"Quando é obrigatório colocar piso que não escorrega, tipo nas escadas e rampas? Quer dizer, quando tem risco de alguém cair, né?"*
   - Classificador `nr-35`/`nr-18`. **Veio:** nr-18·18.8 Escadas, Rampas e Passarelas · nr-18·18.8.6 Escadas · nr-18·18.8.6. **Devia:** nr-08·8.1 (*"Nos pisos, escadas fixas, rampas… onde houver risco de escorregamento, devem ser empregados materiais ou sistemas antiderrapantes."*).
   - bm25raw, dense_text e v4noclf acertam; o boost 5× na NR-18 empurra o chunk certo para fora.
3. **Pergunta sem referência, genérica, NR errada** (`48e91f0b2243480e31296a77`, lay/conditional)
   - Q: *"Se acontecer uma emergência do tipo que a norma diz pra gente se preocupar, o que tem que fazer com a área e com as pessoas que não estão ajudando no socorro?"*
   - **Veio:** nr-01·1.5.6 Preparação e Resposta a Emergências · nr-10·10.11 · nr-30·30.20. **Devia:** nr-22·22.30 Plano de Atendimento a Emergências (*"…toda a área de risco deve ser interditada e as pessoas não diretamente envolvidas no atendimento…"*).
   - É ambiguidade genuína: a query omite que se trata de mineração. Só o denso-texto sozinho acerta. Nenhum retriever lexical resolve.
4. **NR e anexo certos, chunk vizinho** (`3aa12da34de17975946afb94`, operational/factual)
   - Q: *"…preciso definir quem precisa estar habilitado antes de começar o trabalho. Qual é a exigência de capacitação para esses ocupantes [do cesto]?"*
   - **Veio:** chunks 522, 520 e 515 da NR-12 Anexo XII (itens 7.x, 4.28…). **Devia:** chunk **521**, NR-12 Anexo XII, item 4.40 (*"Dentre os ocupantes do cesto, pelo menos um trabalhador deve ser capacitado em código de sinalização…"*).
   - É o caso que o "vizinho ±1" do MobileRAG+ resolve. Isso acontece em **36 das 143 falhas de top-5** (vizinho a ±2) e em **51 das 292 falhas de top-2** (vizinho a ±1).
5. **Pergunta vaga, a evidência é a seção de objetivo** (`b740e58f0872f804486c35cc`, lay/factual)
   - Q: *"…o que é que a norma exige pra gente poder usar esses equipamentos? Tipo, em quais quatro momentos principais ela dá as regras…"*
   - **Veio:** nr-12·12.1.4 · nr-12·12.1 · nr-31·31.12 (máquinas). **Devia:** nr-06·6.1 Objetivo (*"…requisitos para aprovação, comercialização, fornecimento e utilização de Equipamentos de Proteção Individual – EPI"*).
   - "Equipamentos" sem "proteção individual" puxa máquinas. Nenhuma variante acerta.

Bônus, para mostrar o artefato do escopo: `ad09baee832974bb9635bbb9` (clean, gold = **manual** da NR-22). O v4 traz em 1º o chunk nr-22·22.30, que contém literalmente *"22.31.1.2 O disposto no item 22.31.1 desta NR não se aplica durante a fase de abertura da mina"*, que é a evidência. Mas como o `source_file` é o manual, a métrica oficial conta hit = 0.

**Anatomia das 143 falhas de top-5 do v4 em norma:** NR certa com o chunk longe 38 · vizinho ±2 do ouro 36 · NR errada 37 · artefato de alinhamento 32 (o v4 recupera 30 dessas pelo conteúdo).

## 5. Custo / reprodução

- **Máquina:** RTX 5070 local. O encoder Gemma roda em `llama-server` (`~/llama.cpp/build-cuda`, `--embedding --pooling mean`, :8399); o Qwen3-Emb Q8_0 em :8398 com `--pooling last -np 1`. Com batch de 32 e 4 slots, o servidor caiu 1 vez; com 1 query por request rodou estável.
- **Tempos:** v4 + 6 ablações nas 1536 = 69 s · validação no corpus deles (2508 queries, 5 variantes) = 85 s · BM25 não usa GPU.
- **Interpretador:** `classifier/.venv` (sklearn 1.9.1, por causa do pickle). Do lado deles, um venv descartável com `sqlite-vec`.
- **Artefatos não versionados** (gitignored no repo, copiados de outro worktree com md5 conferido): `bench/ctx_topk/data/dense_{text,exponly}.bin`, `classifier/models/classic_winner.pkl`. Assets deles vieram de `huggingface.co/vinyteles/mobilerag-plus-assets@5ca76688…` (`benchmark_mobile/corpus.sqlite` sha256 `0b571032…`, `embedders/qwen3-embedding-0.6b-v2-q8_0.gguf`).
- **Scripts do worktree** (descartados no teardown; o essencial está no Apêndice): `smoke/align.py`, `smoke/eval_smoke.py`, `smoke/metrics.py`, `smoke/ceiling.py`, `smoke/parity.py`, `smoke/diag_gate.py`, `smoke/validate_theirs.py`, `smoke/validate_naive.py`, `smoke/norma_only_qwen.py`, `smoke/their_corpus.py`.

## 6. Limites deste smoke

- Só recuperação: nenhum LLM nem juiz. Não comparo judge score.
- A métrica "evidence_recall" não é a deles (pré vs pós-orçamento).
- O espelho Python do v4 (`retrieval_v4.py`) **não implementa o override de NR explícita** do `HybridRetriever.kt`, por isso reporto `v4` (âncora F3/F4) e `v4app` (o app). A diferença é de +0,9 p.p. de hit@5 em norma. **Os dados de treino da F3/F4 usaram o v4 sem override**: diferença pequena, mas existe.
- Alinhamento de chunks heurístico, com erro limitado (§1.1).

## 7. Recomendação

**NO-GO para a fase "MobileRAG+ vs v4" como comparação/troca de retriever.** Com base equalizada até onde foi possível, o primeiro estágio do v4 (fala crua, top-5) supera o primeiro estágio do MobileRAG+ (Qwen-FT, busca exata, que é o teto do EcoVector) em todas as métricas e vocabulários, com um embedder menor (265 MB vs 639 MB). Trocar de retriever não tem upside medido que pague uma fase dedicada. O ganho publicado do MobileRAG+ vem de vizinhos + SCR, que não dependem da troca.

**GO, em escopo pequeno (horas, não uma fase)**, para 3 alavancas que o benchmark revelou:

1. **Recalibrar o gate do classificador.** No benchmark ele tira 2,9 p.p. de hit@5 e 4,0 de hit@2; no holdout de 151 sem ele cai 11 p.p. de R@2. Opções a medir **nos dois** benchmarks: subir o limiar do filtro duro, reduzir o boost de 5×, ou manter só o override de NR explícita. Portar também o override para o `retrieval_v4.py`, para alinhar âncora de treino e app. A régua dupla (151 + benchmark_mobile) é exatamente a trava anti-mascaramento do `retrieval4`.
2. **Testar vizinhos ±1 em cima do v4 com topK=2.** Teto medido: hit@2 de norma 0,743 → **0,788** se o vizinho entrasse. Só que topK=2 + vizinhos triplica o contexto, e o `bench/ctx_topk` já mostrou que n_ctx 2048 não comporta isso. O teste precisa vir junto com recorte do tipo SCR (o `trim.py` determinístico piorou a citação; o SCR por similaridade de frase deles é a hipótese nova).
3. **Decidir o escopo de manuais.** É decisão de produto: 26% das queries de NR do benchmark (e 9 dos 45 arquivos) estão em manuais. A POC tirou os manuais do BM25 porque poluem o recall nas 151. Se o benchmark do cliente valoriza manuais, a saída é tabela/aba separada (a recomendação antiga) com um sinal próprio no RRF, e não misturar no BM25.

**Opcional e de baixo valor esperado:** o Qwen-FT como 4º sinal no RRF deu +3,9 p.p. de hit@5 no corpus deles sobre Gemma+BM25. Custa +639 MB de RAM e ~0,5 s por query no aparelho, então só vale se as alavancas 1 e 2 se esgotarem.

**Trabalho a enviar:** nenhum bug de produto reproduzido. O achado acionável é o desalinhamento entre o espelho Python e o app (override de NR explícita ausente no `bench/ctx_topk/retrieval_v4.py`). Se firstmate promover, a correção é ~15 linhas e afeta a geração de dados de F3/F4 futuros.

## Apêndice: núcleo do código (para reprodução)

```python
# Relevância e métricas (espelha mobile_eval benchmark_mobile_test.py:174-214, 332-384)
def covered(spans, ev):                      # chars de evidência cobertos pela união dos spans
    cov = set()
    for es, ee in ev:
        for s, e in spans: cov.update(range(max(s, es), min(e, ee)))
    return len(cov)
ev_chars = covered([(0, max(e for _, e in ev))], ev)
gain = lambda chunk: covered(chunk.spans, ev) / ev_chars if chunk.source_file == row.source_file else 0.0
ndcg5 = DCG(gains[:5]) / DCG(sorted(gain(c) for c in chunks_do_arquivo_ouro)[::-1][:5])   # DCG = Σ g/log2(r+1)
mrr5  = 1 / primeiro_rank_com_gain>0
hit@k = any(gain>0 em top-k);  evidence_recall@k = covered(união spans top-k do arquivo-ouro, ev) / ev_chars

# Alinhamento chunk→fonte: tokens \w+ sem acento; âncora 6-grama; difflib.SequenceMatcher(chunk_tokens,
# janela_fonte[c0-30 : c0+2.5*len+60]); spans = blocos casados (size>=3), lacunas<=8 tokens fundidas;
# corpo = texto após " — " (remove o prefixo "NR-XX · seção título").

# v4 (âncora F3/F4): R4.RetrieverV4(qs).search(id, question, top_k=5)  [bench/ctx_topk/retrieval_v4.py]
# v4app: s1 = filtro duro na NR explícita (regex \bnr[\s._-]*(\d{1,2})\b) senão hy.rank; RRF k=10 w=(2,1,2)
# v4noclf: s1 = hy.bm.search(app_fts_query(q), 60); mesmos 2 densos; mesmo RRF
```

-- revisado segundo as diretrizes da fábrica
