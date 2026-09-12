# Reprocessamento do corpus com Document AI Form Parser (`docai/`)

Reprocessamento das NRs tabulares com o **Google Document AI Form Parser** (entende
estrutura de tabela e faz OCR interno) para reparar as tabelas destruídas na extração do
`nrs.parquet` — em especial a **tabela do Anexo II da NR-10** (raios de zona de risco/
controlada por faixa de tensão), que tornava **impossível** responder ao caso do capitão
"qual a distância segura para 13,8 kV?".

Ordem do capitão: reprocessar em vez de remendar com regras caseiras. Execução pesada na
**DGX Spark** (`walcyrios@spark-b431`), pasta única `~/cemig-poc/`. Só scripts e
resultados voltam ao repo.

## TL;DR — veredito

| Achado | Evidência |
|---|---|
| **Glifos de fonte privada ELIMINADOS na origem** | DocAI produz `≥`/`<` Unicode reais; `glyphs={}` nas 23 NRs. O parquet tinha **47 glifos PUA** (`\uf03c`=<, `\uf0b3`=≥, `\uf0a3`=≤, `\uf0b7`=•) em 4 normas. |
| **Anexo II da NR-10 reconstruído 100%** | As 18 faixas de tensão × (Rr, Rc) casam célula-a-célula com o parquet; ex.: `≥10 e 15 → 0,38 / 1,38`. |
| **Caso do capitão 13,8 kV: de IMPOSSÍVEL a recuperável** | O índice atual tem **ZERO chunks com "13,8"**. Após DocAI + expansão de faixa, existe **1 chunk com "13,8" E "0,38"**, e ele fica **top-1** quando o token "13,8" é preservado na busca. |
| **Aditivo de TODAS as tabelas POLUI o BM25** | Índice aditivo-23 derruba **R@2 151 de 13,9 → 11,9 (-2,0)** — mesmo efeito documentado dos manuais. Só a tabela **estruturada** (zona-de-risco) tem valor; despejos genéricos (portarias, CNAE) são ruído. |
| **Ganho é destravado por 2 alavancas independentes** | (1) **dado** (DocAI reconstrói o Anexo II); (2) **tokenizer** (o app descarta "13,8" e "kV"). Sem a (2), o app não chega ao chunit certo mesmo com o dado corrigido. |
| **Recomendação** | **NÃO substituir o índice de produção com o aditivo.** Aplicar **reparo cirúrgico** só do Anexo II da NR-10 (`replace` de 3 chunks corrompidos) **acompanhado** do fix de tokenizer numérico. Isolar tabelas genéricas fora do BM25 (aba/tabela separada), como já se fez com os manuais. |

## Pipeline (scripts)

Todos em `docai/`, comentários em pt-BR, código em inglês, não-interativos, checkpoint incremental.

1. **`01_batch_process.py`** — upload dos PDFs → `batch_process_documents` (LRO async) →
   download dos shards JSON. Processor Form Parser **`83cc594eb5e2ea45`** (descoberto via
   API, projeto `sebrae-arquitetura-ia-teste`, location `us`), bucket
   `gs://ceiaidp-raw-docs-dev/temp-eletrico-documentai/`. Auth por ADC
   (`authorized_user` com `refresh_token`, copiado p/ a Spark).
   - **Limite de 100 páginas**: NR-12/15/28 excedem; fatiadas em blocos de 90 páginas
     (`pypdf`) antes do batch.
2. **`02_parse_tables.py`** — parse dos shards + **costura entre páginas** (cabeçalho
   repetido / continuidade de nº de colunas) + **normalização de glifos** (mapa
   determinístico PUA→ASCII) + **achatamento em frases pesquisáveis** com expansão de faixa
   de tensão.
3. **`03_verify_cells.py`** — **verificação célula-a-célula** contra o parquet: todo número
   das frases tem de existir na fonte (após normalizar glifos). O que não fecha vai para a
   lista "não reparável automaticamente" **com contagem**.
4. **`04_reindex_compare.py`** — índice **aditivo** (chunks atuais + frases DocAI) e
   comparação nas **mesmas 151** pelo caminho do app (`app_fts_query`, pesos 1.5/3/2/1,
   casamento por conteúdo `check_hit`). Roda os 2 casos do capitão.
5. **`05_e2e_approval.py`** — **aprovação honesta E2E** (bench/regua `ruler.py` +
   `gabarito_151`) no subconjunto NR-10 do holdout (única NR-piloto nas 151), síntese com o
   1.2B QAD embarcado na RTX 5070.
6. **`06_replace_ceiling.py`** — config **REPLACE** (troca cirúrgica dos 3 chunks
   corrompidos do Anexo II, sem despejo genérico) + **teto do tokenizer** (app-tok vs
   numeric-preserving) no caso do capitão.

Reproduzir: subir os scripts p/ `~/cemig-poc/scripts/` na Spark (etapas 1-2); etapas 3-6
rodam localmente (`classifier/.venv`, tem `requests`; FTS5 disponível). Resultados em
`docai/results/*.json`.

## Diagnóstico antes/depois

### Glifos privados (PUA)
`data/diagnostic_before.json` — **47** glifos no parquet: `\uf0b3`(≥) 20×, `\uf03c`(<) 19×,
`\uf0a3`(≤) 6×, `\uf0b7`(•) 2× (37 na NR-10). **Depois (DocAI): 0** — o Form Parser lê a
fonte corretamente e emite Unicode padrão. Nenhum mapa de glifos foi necessário sobre a
saída do DocAI (o mapa em `02` é defensivo).

### Tabelas costuradas / reparadas / não-reparáveis (23 NRs)
- **696 tabelas cruas → 562 costuradas** (121 costuras marcadas **ambíguas** — colunas
  batem mas cabeçalho difere; costuradas e registradas p/ auditoria). **9.290 frases**
  pesquisáveis geradas.
- **Verificação célula-a-célula**: **43.803 números verificados** contra o parquet;
  **2** rótulos de tensão gerados (expansão de faixa, procedência declarada); **230
  não-reparáveis automaticamente** (**0,5%**) — todos artefatos de OCR (células mescladas
  tipo `52205`, colunas de código CNAE repetidas `3333333`), **nenhuma corrupção de valor
  regulatório**. Preferimos relatar 230 casos para olho humano a fingir cobertura total.
- **Anexo II da NR-10**: **18/18 linhas** reconstruídas e verificadas 1:1 com o parquet.

## Recall nas 151 (caminho do app, casamento por conteúdo)

| Config | R@2 (151) | R@5 (151) | R@2 (NR-10, n=45) |
|---|---|---|---|
| Atual (`index_hf_36nr.db`) | 13,9 | 20,5 | 15,6 |
| Aditivo **5 NRs-piloto** | 15,2 (+1,3) | 21,2 (+0,7) | 20,0 (+4,4) |
| Aditivo **23 NRs** | **11,9 (-2,0)** | 19,9 (-0,6) | 15,6 (+0,0) |
| Aditivo **só NR-10** | 13,9 (+0,0) | 20,5 (+0,0) | 15,6 (+0,0) |
| REPLACE (Anexo II NR-10) | 10,6 (-3,3) | 17,2 (-3,3) | — |

**Leitura honesta**: o +4,4 do 5-piloto em NR-10 é **majoritariamente ruído de IDF**
(perturbação das estatísticas de coleção por chunks de outras NRs; as perguntas que
fliparam — qa-040, qa-044 — **não** são da tabela de zona). O aditivo-23 **polui** o BM25
(portarias/CNAE viram ruído lexical). O REPLACE cai porque colapsar 3 chunks do Anexo II em
1 reduz casamentos de seção. **Nenhuma config aditiva/replace paga a substituição do índice
de produção sob o retrieval BM25 atual.**

## Casos do capitão

### Caso 1 — "qual a distância segura para média tensão de 13,8 kV?" (`06_replace_ceiling.py`)

| Índice × tokenizer | top-1 | Contém 13,8 + 0,38? |
|---|---|---|
| Atual, tok do app | nr-10/10.6 | ❌ |
| Atual, tok **numérico** | nr-10/Anexo Ii (**chunk corrompido**) | ❌ (sem "13,8", com glifos) |
| **DocAI (replace), tok do app** | nr-10/10.6 (Anexo II em rank-3) | ✅ no rank-3 |
| **DocAI (replace), tok numérico** | **nr-10/Anexo Ii** | ✅ **top-1** |

**Conclusão**: hoje a resposta é **impossível** (o índice atual não tem "13,8" em lugar
nenhum; o chunk do Anexo II tem glifos e não mapeia 13,8→0,38). Com o DocAI, o chunk
correto passa a **existir e ser recuperável**; com o fix de tokenizer numérico ele vira
**top-1**. As duas alavancas são necessárias e independentes.

### Caso 2 — "o poste que preciso subir não me parece firme"
Cai em NR-22/NR-30 (norma errada) **antes e depois** — a reextração de tabela **não ajuda**
neste caso (é problema de **classificação de norma**: era NR-35/NR-01, não NR-07/NR-10, e
não depende de tabela). Sem ganho honesto a reportar aqui; a alavanca é o classificador/
rewriter, não o Form Parser.

## Aprovação honesta E2E (bench/regua, NR-10, `05_e2e_approval.py`)
Cobertura de fatos **subiu** (0,116 → 0,131) mas aprovação ficou **flat** (3→2/45, 1 flip =
ruído no piso). **Confirma o gargalo DUPLO já documentado**: com o chunk certo dado, o 1.2B
embarcado converte pouco em resposta aprovada (gargalo LM). O reparo de dado é **condição
necessária, não suficiente** — destrava o retrieval, mas a síntese segue sendo o teto no
aparelho. (Latência de GPU não vale p/ aparelho — `engine:cuda`.)

## Recomendação final
1. **NÃO substituir** `index_hf_36nr.db` pelo aditivo (poluição -2,0 R@2) nem pelo REPLACE
   colapsado (-3,3 R@2) **como estão**.
2. **Reparo cirúrgico do Anexo II da NR-10**: manter os **3 chunks por página** (preservar
   casamento de seção) mas com o **texto DocAI limpo + frases de faixa expandidas** (13,8
   etc.), em vez de colapsar em 1. É o único ponto com dado estruturado de alto valor no
   holdout.
3. **Destravar o tokenizer numérico** no `Fts5Retriever` (preservar `13,8`/`0,38`/`kV`) —
   sem isso, nenhum reparo de dado chega ao chunk certo. Maior alavanca isolada p/ o caso
   do capitão.
4. **Isolar tabelas genéricas** (portarias, CNAE, tempos de descompressão) **fora do BM25**
   (tabela/aba separada), como já feito com os manuais — evita a poluição.
5. As demais 18 NRs tabulares foram reprocessadas e verificadas (artefatos reprodutíveis),
   mas **não** entram no índice de produção sem ganho medido.

## Procedência / dados
- Scripts e `results/*.json` versionados. Bancos `.db`, shards e dumps volumosos ficam em
  `~/cemig-poc/` na Spark e em `docai/data*/` (gitignored, reprodutíveis).
- `results/reconstructed_nr10_anexo_ii.json` — o artefato-chave (18 frases verificadas).
- Juiz/síntese: 1.2B QAD embarcado na RTX 5070; régua honesta `bench/regua/ruler.py` +
  `gabarito_151`.
