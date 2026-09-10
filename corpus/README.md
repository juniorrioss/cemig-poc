# Corpus Pipeline — CEMIG POC Assistente por Voz 100% Offline

Módulo de processamento de texto, chunking semântico estruturado, indexação em SQLite FTS5 (BM25) e avaliação de recuperação para o assistente por voz móvel (Samsung Galaxy S21/S24+).

O objetivo deste pipeline é converter os documentos regulamentares das Normas Regulamentadoras (PDFs oficiais) em um banco de dados leve (`index.db` ≤ 50 MB), embarcável diretamente no app Android, viabilizando citação de fonte obrigatória (norma + item/seção) e recuperação rápida acionada por tool-calling do SLM.

---

## 1. Justificativa das 5 NRs Selecionadas

Para a prova de conceito focada em operários de campo do setor elétrico (manutenção de redes de distribuição, linhas de transmissão, subestações e canteiros de obras), foram selecionadas as seguintes 5 normas:

1. **NR-10 — Segurança em Instalações e Serviços em Eletricidade (Mandatória)**:
   - Norma central e indispensável. Regulamenta todas as fases de geração, transmissão, distribuição e consumo de energia elétrica.
   - Define a regra de ouro das **6 etapas obrigatórias de desenergização** (item 10.5.1), trabalhos em alta tensão (AT) e SEP (em dupla obrigatória), delimitação de zonas de risco e controlada, vestimentas anti-arco elétrico, prontuário (PIE), habilitação/autorização e direito de recusa.
2. **NR-06 — Equipamentos de Proteção Individual (EPI)**:
   - Vital para o eletricista: critérios de seleção, guarda, conservação e responsabilidades do trabalhador e empregador.
   - Especifica os EPIs dielétricos e de proteção contra arco elétrico (Anexo I): luvas de borracha isolante, sobreluvas de couro, calçados sem biqueira metálica condutora, capacetes classe B e óculos com proteção UV.
3. **NR-35 — Trabalho em Altura**:
   - Rotina diária de eletricistas de rede aérea (trabalho acima de 2,00 m em postes de concreto/madeira, torres de transmissão e cestos aéreos).
   - Regulamenta Sistemas de Proteção Contra Quedas (SPQ), talabarte duplo em Y com absorvedor de energia, cinto tipo paraquedista, pontos de ancoragem e condições meteorológicas impeditivas (chuva, ventos fortes e descargas atmosféricas).
4. **NR-12 — Segurança no Trabalho em Máquinas e Equipamentos**:
   - Fundamental para serviços elétricos em painéis de força/comando, transformadores e motores industriais.
   - Estabelece procedimentos de manutenção com bloqueio de energias perigosas (**LOTO — Lockout & Tagout**, item 12.11), aterramento obrigatório de carcaças metálicas e circuitos de parada de emergência.
5. **NR-18 — Segurança e Saúde no Trabalho na Indústria da Construção**:
   - Abrange instalações elétricas temporárias em canteiros de obra (item 18.6), proibição de cabos elétricos no solo obstruindo passagem, aterramento de estruturas metálicas, proteção obrigatória por dispositivo DR e distâncias mínimas de segurança em relação a linhas elétricas aéreas vizinhas.

---

## 2. Estrutura dos Arquivos e Entregáveis

```
corpus/
├── extract.py         # [1] Leitura de PDFs e estruturação hierárquica por seção/item
├── chunk.py           # [2] Chunks de ~200-400 tokens com cabeçalho contextual
├── build_index.py     # [3] Geração do banco SQLite FTS5 (index.db) com BM25
├── synth_qa.py        # [4] Geração de dataset sintético P&R de operários (JSONL) via LLM CLI
├── eval_retrieval.py  # [5] Avaliação de Recall@1/3/5 e MRR do BM25
├── test_pipeline.py   # Testes unitários do pipeline (stdlib unittest)
├── qa_pairs.jsonl     # Dataset com 101 pares pergunta-ouro/resposta/chunk-fonte
├── index.db           # Banco SQLite FTS5 gerado (~1.54 MB)
├── Makefile           # Automação do pipeline ponta a ponta
└── README.md          # Documentação técnica e resultados
```

---

## 3. Como Rodar o Pipeline Ponta a Ponta

### Requisitos
- Python 3.10+ (utiliza apenas a biblioteca padrão `sqlite3`, `json`, `re` etc. + dependência mínima `pypdf` para leitura dos PDFs).
- Os PDFs fonte devem estar em `/home/rios/projetos/cemig-mobile-llm/firstmate/data/NRs/` (ou caminho configurável via `--pdf-dir`).

### Execução em Comando Único

Na raiz do repositório ou dentro de `corpus/`:
```bash
make all
```

Isso executará automaticamente:
1. `make extract` — extrai o texto estruturado dos PDFs das 5 NRs para `corpus/data/extracted/`.
2. `make chunk` — cria os chunks de ~200–400 tokens com cabeçalhos contextuais em `corpus/data/chunks.json`.
3. `make index` — compila o banco `corpus/index.db` com tabela virtual FTS5 e BM25.
4. `make eval` — executa o benchmark de recuperação contra os 101 pares de teste em `corpus/qa_pairs.jsonl`.

### Execução de Testes Unitários
```bash
make test
```

### Execução Passo a Passo (Manual)

1. **Extração de texto**:
   ```bash
   python -m corpus.extract --pdf-dir /home/rios/projetos/cemig-mobile-llm/firstmate/data/NRs --nrs nr-10,nr-06,nr-35,nr-12,nr-18
   ```

2. **Geração de chunks**:
   ```bash
   python -m corpus.chunk --extracted-dir corpus/data/extracted --output-file corpus/data/chunks.json
   ```

3. **Criação do índice SQLite FTS5**:
   ```bash
   python -m corpus.build_index --chunks-file corpus/data/chunks.json --db corpus/index.db --rebuild
   ```

4. **Consulta interativa de teste via BM25**:
   ```bash
   python -m corpus.build_index --db corpus/index.db --query "desenergizacao ordem sequencia seccionamento"
   ```

5. **Avaliação de Retrieval**:
   ```bash
   python -m corpus.eval_retrieval --db corpus/index.db --qa-file corpus/qa_pairs.jsonl
   ```

6. **(Opcional) Regeneração dos pares sintéticos**:
   ```bash
   python -m corpus.synth_qa --db corpus/index.db --output corpus/qa_pairs.jsonl --force
   ```

---

## 4. Resultados de Recuperação Obtidos (BM25)

A avaliação foi realizada com **101 pares de Perguntas & Respostas Ouro** distribuídos nas 5 normas regulamentadoras. Foram comparadas quatro abordagens de recuperação:

| Estratégia de Retrieval | Recall@1 | Recall@3 | Recall@5 | MRR |
|:------------------------|:--------:|:--------:|:--------:|:---:|
| **1. Pergunta Bruta de Voz (Stopwords + OR)** | 10.9% | 21.8% | 22.8% | 0.1559 |
| **2. Tool-Calling SLM (Termos Soltos OR)** | 44.6% | 65.3% | 71.3% | 0.5530 |
| **3. Tool-Calling SLM (Termos Soltos + Boost Doc)** | 47.5% | 72.3% | 78.2% | 0.5959 |
| **4. Tool-Calling SLM (Filtrado por Norma `doc=nr-XX`)** | **51.5%** | **77.2%** | **85.1%** | **0.6541** |

### Detalhamento por Norma Regulamentadora (Estratégia 4 — Recomendada)

| Norma | Descrição | Qtd Pares | Recall@1 | Recall@3 | Recall@5 | MRR |
|:-----:|:----------|:---------:|:--------:|:--------:|:--------:|:---:|
| **NR-10** | Eletricidade (Mandatória) | 45 | **64.4%** | **97.8%** | **97.8%** | **0.7926** |
| **NR-06** | EPIs Dielétricos e Gerais | 8 | **50.0%** | **75.0%** | **87.5%** | **0.6562** |
| **NR-35** | Trabalho em Altura | 18 | **44.4%** | **77.8%** | **88.9%** | **0.6333** |
| **NR-18** | Canteiro e Proximidade de Redes | 17 | 41.2% | 47.1% | 64.7% | 0.4853 |
| **NR-12** | Máquinas e Bloqueio LOTO | 13 | 30.8% | 46.2% | 61.5% | 0.4231 |
| **Total** | **Todas as 5 NRs** | **101** | **51.5%** | **77.2%** | **85.1%** | **0.6541** |

> **Meta atingida:** O Recall@5 global atingiu **85.1%** (superando com folga o critério de **≥ 80%** estabelecido pelo projeto), e na norma crítica principal (**NR-10**) alcançou **97.8%**!

---

## 5. Análise dos Resultados: Por que o Tool-Calling é Crucial

Os resultados experimentais confirmam taxativamente a hipótese de arquitetura definida para o assistente CEMIG:

1. **A pergunta bruta de voz falha no BM25 lexico (Recall@5 de apenas 22.8%)**:
   - Um operário na rede elétrica fala pelo rádio ou microfone de modo coloquial e situacional:
     > *"Ô, tô aqui na subestação e o encarregado falou que não dá pra desligar o circuito hoje. Posso trabalhar só com a luva de borracha mesmo ou tem outro jeito antes de partir pro EPI?"*
   - Essa fala contém palavras como *"hoje"*, *"jeito"*, *"partir"*, *"encarregado"*, que não constam na norma técnica. O BM25 semântico se dilui e ranqueia chunks irrelevantes.
2. **O SLM com Tool-Calling extrai os termos essenciais da norma (Recall@5 salta para 71.3% a 85.1%)**:
   - O modelo de linguagem local na CPU/NPU do smartphone (S21/S24+), antes de responder, reformula a consulta em palavras-chave técnicas:
     > `tool_call: search_regulations(norma="nr-10", query="protecao coletiva desenergizacao tensao seguranca prioridade")`
   - O índice SQLite FTS5 processa a consulta OR com tokenizer `unicode61 remove_diacritics 2`, ignorando acentuação e casando perfeitamente com os cabeçalhos e termos normativos.
3. **Escopo por Norma (`doc = 'nr-XX'`)**:
   - Quando o SLM detecta o contexto (ex: choque/rede elétrica → NR-10; altura/poste → NR-35; vestimenta/luva → NR-06), ele filtra ou pondera a busca, elevando o Recall@5 para **85.1%** global e **97.8% na NR-10**.

---

## 6. Características Técnicas do Banco `index.db`

- **Tamanho final do banco**: **1.54 MB** (muito abaixo do limite de 50 MB, viabilizando inclusão direta nos assets do APK Android).
- **Esquema relacional**:
  ```sql
  CREATE TABLE chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      doc TEXT NOT NULL,
      section TEXT NOT NULL,
      title TEXT NOT NULL,
      page INTEGER NOT NULL,
      text TEXT NOT NULL
  );
  ```
- **Tabela virtual FTS5**:
  ```sql
  CREATE VIRTUAL TABLE chunks_fts USING fts5(
      doc,
      section,
      title,
      text,
      content='chunks',
      content_rowid='id',
      tokenize='unicode61 remove_diacritics 2'
  );
  ```
- **Formato dos Chunks**:
  - Cada chunk possui entre **200 e 400 tokens** (média: 346 tokens; mediana: 364 tokens).
  - Cada chunk inicia obrigatoriamente com o cabeçalho contextual:
    `NR-10 · 10.2.8 Medidas de Proteção Coletiva — 10.2.8.1 Em todos os serviços executados...`
  - Metadados anexados permitem citação precisa de fonte: documento (`nr-10`), seção (`10.2.8`), título formatado e página de referência.

---

## 7. Geração do Dataset Sintético P&R (`synth_qa.py`)

- **LLM CLI Utilizado**: Anthropic Claude Code CLI (`claude -p`), versão 2.1.241.
- **Metodologia**:
  - Foram mapeados 72 tópicos operacionais críticos das 5 NRs abordando situações reais da rotina de operários elétricos (chuva com raio, trabalho em poste, disjuntor travado, uso de aliança, teste de ausência de tensão, resgate em altura, bloqueio com cadeado LOTO).
  - O prompt forçou estritamente que a pergunta soasse com fala natural brasileira de campo ("Tô no poste...", "O encarregado falou que...", "Posso subir sozinho?").
  - A resposta-ouro (`golden_answer`) inclui a resposta técnica completa e a citação formal obrigatória da norma e do item aplicável.
  - O dataset resultante está versionado em `corpus/qa_pairs.jsonl` com 101 registros prontos para execução imediata offline.
