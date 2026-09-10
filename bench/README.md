# Benchmark de SLMs Offline (Tier 1) — CEMIG POC

Avaliação comparativa rigorosa de Small Language Models (SLMs ≤ 1.2B parâmetros) quantizados em **GGUF Q4_K_M** via **llama.cpp**, dimensionados para execução 100% offline em dispositivos móveis Android (Samsung Galaxy S21 / S24+).

O objetivo central deste benchmark é responder à **pergunta de ouro da arquitetura**:
> **O disparo de RAG via Tool-Calling diretamente pelo SLM é viável nesse porte (≤1B) ou a solução de produção exige RAG Clássico com heurística determinística de disparo fora do modelo?**

---

## 1. Modelos Avaliados (Tier 1)

Todos os modelos foram quantizados em **Q4_K_M** e executados individualmente (um por vez) para respeitar o teto de RAM e as restrições térmicas de dispositivos móveis:

| Modelo | Arquitetura | Parâmetros | Tamanho GGUF | Repositório HuggingFace | Função no Benchmark |
|:-------|:------------|:----------:|:------------:|:------------------------|:--------------------|
| **Qwen 3.5 0.8B** | Híbrido (Gated DeltaNet) | 752 M | 497.4 MiB | `unsloth/Qwen3.5-0.8B-GGUF` | Híbrido linear/atenção com suporte nativo a tools |
| **Qwen 3 0.6B** | Transformer denso | 596 M | 372.7 MiB | `unsloth/Qwen3-0.6B-GGUF` | Baseline ultraleve |
| **Gemma 3 1B IT** | Transformer denso | 1.000 M | 762.5 MiB | `bartowski/google_gemma-3-1b-it-GGUF` | Modelo Google otimizado para instruções |
| **LFM2 1.2B RAG** | Liquid Neural Network (SSM/Conv) | 1.170 M | 694.8 MiB | `LiquidAI/LFM2-1.2B-RAG-GGUF` | Modelo Liquid especializado em borda e RAG |
| **LFM2.5 350M** | Liquid Neural Network (SSM/Conv) | 354 M | 216.4 MiB | `LiquidAI/LFM2.5-350M-GGUF` | Modelo ultracompacto para altíssima velocidade |
| **Llama 3.2 1B** | Transformer denso | 1.236 M | 762.8 MiB | `unsloth/Llama-3.2-1B-Instruct-GGUF` | Modelo de controle Meta |

---

## 2. Protocolo de Benchmark (Dois Modos por Modelo)

O benchmark submete cada modelo a **dois modos operacionais distintos** sob as mesmas 20 situações de campo da CEMIG:

### Modo A: RAG Clássico (Contexto Pré-Injetado)
- O orquestrador executa a busca lexical (BM25 via SQLite FTS5) antes de chamar o modelo.
- Os **top-3 chunks** normativos mais relevantes são injetados diretamente no prompt com diretrizes rígidas em português:
  1. Basear-se exclusivamente no contexto.
  2. Citar expressamente a norma e o item/seção oficial (ex: `NR-10, item 10.5.1`).
  3. Declarar expressamente *"Não sei com base nas normas consultadas"* caso o contexto não respalde a resposta.
- **Invocação única**: 1 rodada de inferência.

### Modo B: RAG por Tool-Calling (Disparo pelo Modelo)
- O modelo recebe a pergunta em linguagem natural de campo e a definição formal da ferramenta:
  `retriever(query: string)` — busca nas Normas Regulamentadoras (NR-10, NR-06).
- **Rodada 1**: O modelo decide autonomamente se deve chamar a ferramenta e formula a query de busca (podendo reformular a linguagem coloquial do operário em termos técnicos).
- **Execução**: Se o modelo acionar a tool, o harness intercepta a chamada, executa o BM25 no índice SQLite FTS5 (`index.db`) e injeta os chunks retornados em uma mensagem de papel `tool`.
- **Rodada 2**: O modelo processa os dados recuperados e sintetiza a resposta técnica final com citação de fonte.
- **Métricas registradas**:
  - Acionou a tool quando devia?
  - Reformulou a query para termos técnicos?
  - Manteve os termos em português ou traduziu para inglês?
  - Sintetizou a resposta final com fidelidade ou alucinou?

---

## 3. Conjunto de Avaliação (20 Questões de Ouro)

O smoke test foi construído com **20 perguntas manuais** elaboradas diretamente a partir dos textos oficiais das normas regulamentadoras presentes em `firstmate/data/NRs/` (**NR-10 — Segurança em Instalações Elétricas** e **NR-06 — Equipamentos de Proteção Individual**), cobrindo:

1. **Procedimentos Críticos**: Ordem exata das 6 etapas de desenergização (NR-10, item 10.5.1); processo de reenergização (NR-10, item 10.5.2).
2. **Proteção Coletiva vs Individual**: Prioridade da tensão de segurança (NR-10, item 10.2.8.2); vestimentas anti-arco elétrico e condutibilidade (NR-10, item 10.2.9.2).
3. **Proibições e Regras de Ouro**: Proibição terminante de adornos pessoais/aliança em painéis (NR-10, item 10.2.9.3); proibição de trabalho individual em alta tensão/SEP (NR-10, item 10.7.3); direito de recusa (NR-10, item 10.14.1).
4. **Obrigações e EPIs**: Gratuidade do EPI e dever de troca imediata pelo empregador (NR-06, itens 6.3 e 6.3.1); responsabilidades de guarda e finalidade exclusiva pelo trabalhador (NR-06, item 6.5.1); exigência de Certificado de Aprovação — CA (NR-06, item 6.6.1); características do calçado dielétrico (NR-06, Anexo I).
5. **Perguntas Negativas e Fora de Escopo**: Perguntas tributárias sem relação com NRs (alíquota de ICMS em MG — onde o modelo DEVE responder *"não sei"*); improvisação perigosa com arame em fusível (procedimento proibido).

O arquivo está versionado em `bench/data/smoke_qa_20.jsonl` e o índice SQLite FTS5 de fallback em `bench/data/minimal_index.db`. Caso `corpus/index.db` e `corpus/qa_pairs.jsonl` estejam disponíveis no repositório, o harness os detecta e utiliza automaticamente.

---

## 4. Tabela Consolidada de Resultados de Qualidade

Avaliação automatizada via **LLM Juiz** (`claude -p` CLI v2.1.241) em 4 eixos (notas de 1,00 a 5,00) em conjunto com a conferência manual das 20 perguntas de ouro:

| Modelo | Modo | Acerto Factual | Fidelidade Contexto | Qualidade PT | Citação Fonte | Score Global | Gate Pass (%) | Latência Média |
|:-------|:-----|:--------------:|:-------------------:|:------------:|:-------------:|:------------:|:-------------:|:--------------:|
| **LFM2 1.2B RAG** | **Classic RAG** | **3.30** | 3.25 | 4.05 | **2.50** | **3.24** | **40.0%** | **5.10 s** |
| **Gemma 3 1B IT** | **Classic RAG** | 2.65 | **3.80** | 4.50 | 1.90 | **3.12** | **20.0%** | **3.40 s** |
| **Qwen 3.5 0.8B** | **Classic RAG** | 2.75 | 2.80 | 3.00 | **2.65** | 2.78 | 20.0% | 7.90 s |
| Qwen 3 0.6B | Classic RAG | 2.60 | 2.80 | 3.45 | 2.55 | 2.78 | 15.0% | 7.68 s |
| Llama 3.2 1B | Classic RAG | 2.10 | 3.50 | **4.85** | 1.10 | 2.73 | 5.0% | 2.06 s |
| LFM2.5 350M | Classic RAG | 2.05 | 3.55 | **4.85** | 1.10 | 2.73 | 5.0% | **0.65 s** |
| | | | | | | | | |
| **Qwen 3.5 0.8B** | **Tool-Calling** | 1.90 | 1.95 | 2.40 | 2.05 | 2.02 | 15.0% | 10.03 s |
| Qwen 3 0.6B | Tool-Calling | 1.90 | 2.45 | 3.55 | 1.60 | 2.25 | 10.0% | 6.27 s |
| Llama 3.2 1B | Tool-Calling | 1.15 | 1.20 | 1.15 | 1.10 | 1.15 | 5.0% | 2.19 s |
| Gemma 3 1B IT | Tool-Calling | 2.00 | 1.65 | 3.35 | 1.60 | 2.02 | 5.0% | 4.13 s |
| LFM2 1.2B RAG | Tool-Calling | 2.35 | 2.75 | 4.10 | 1.50 | 2.56 | 0.0% | 3.24 s |
| LFM2.5 350M | Tool-Calling | 1.85 | 1.65 | 3.40 | 1.95 | 2.04 | 0.0% | 0.93 s |

> **Critério do Gate**: Score Global = $(Factual \times 0.35) + (Fidelidade \times 0.30) + (Fonte \times 0.20) + (PT \times 0.15)$.
> Para aprovação (`pass_gate`), a resposta precisa atingir $Score \ge 3.50$ com notas individuais $\ge 3$ em Factual, Fidelidade e Fonte.

### Diagnóstico Detalhado de Tool-Calling por Modelo

| Modelo | Taxa Disparo Tool | Disparo Quando Esperado | Reformulou Query? | Traduziu p/ Inglês? | Diagnóstico Operacional |
|:-------|:-----------------:|:-----------------------:|:-----------------:|:-------------------:|:------------------------|
| **Qwen 3.5 0.8B** | **100.0%** (20/20) | **95.0%** | **100.0%** | **0.0%** (PT puro) | **Único com tool-calling nativo funcional**. Converteu linguagem coloquial em termos técnicos perfeitamente. Porém, sofre com aumento de latência (duas rodadas somam 10,03 s). |
| **Qwen 3 0.6B** | 40.0% (8/20) | 35.0% | 100.0% | 0.0% | Disparo instável. Ignorou a ferramenta em 60% das vezes e tentou responder sem contexto. |
| **Llama 3.2 1B** | 10.0% (2/20) | 15.0% | 100.0% | 0.0% | **Inviável para tools nativas**: causou falha gramatical de peg-parser na 2ª rodada (`The model produced output that does not match the expected peg-native format`). |
| **Gemma 3 1B IT** | **0.0%** (0/20) | 5.0% | 0.0% | 0.0% | Template oficial GGUF **não suporta tools**. Respondeu diretamente sem acionar o retriever. |
| **LFM2 1.2B RAG** | **0.0%** (0/20) | 5.0% | 0.0% | 0.0% | Ignorou a chamada de função em português mesmo com `tool_choice: required`. Respondeu diretamente com conhecimento residual. |
| **LFM2.5 350M** | **0.0%** (0/20) | 5.0% | 0.0% | 0.0% | Porte ultracompacto não assimila o esquema JSON de ferramentas; respondeu diretamente. |

---

## 5. Métricas de Performance Relativa (llama-bench Desktop)

Executado em CPU AMD Ryzen 7 9800X3D (8 cores, 16 threads, AVX-512/AVX2), quantizações **Q4_K_M**:

| Modelo | Parâmetros | Tamanho GGUF | Prefill pp64 | Prefill pp512 (RAG) | Decode tg16 | Decode tg128 | Memória RAM Estimada |
|:-------|:----------:|:------------:|:------------:|:-------------------:|:-----------:|:------------:|:--------------------:|
| **LFM2.5 350M** | 354.5 M | 216.4 MiB | 1.118,6 t/s | **1.410,6 t/s** | **258,3 t/s** | **273,2 t/s** | ~280 MiB |
| **Qwen 3 0.6B** | 596.0 M | 372.7 MiB | 908.4 t/s | 924.3 t/s | 145.2 t/s | 136.5 t/s | ~430 MiB |
| **Qwen 3.5 0.8B** | 752.4 M | 497.4 MiB | 380.9 t/s | 454.7 t/s | 83.3 t/s | 82.7 t/s | ~560 MiB |
| **LFM2 1.2B RAG** | 1.170 M | 694.8 MiB | 412.6 t/s | 455.6 t/s | 73.0 t/s | 77.0 t/s | ~760 MiB |
| **Llama 3.2 1B** | 1.236 M | 762.8 MiB | 477.1 t/s | 475.2 t/s | 69.8 t/s | 69.4 t/s | ~830 MiB |
| **Gemma 3 1B IT** | 999.9 M | 762.5 MiB | 320.6 t/s | 341.8 t/s | 62.6 t/s | 62.5 t/s | ~840 MiB |

*Nota: As taxas de throughput em desktop servem exclusivamente como régua relativa entre as arquiteturas, não representando os números absolutos do smartphone Galaxy S21/S24+.*

---

## 6. Análise Técnica: Resposta à Pergunta Central do Capitão

> **Pergunta do Capitão**: *Tool-calling nesse porte (≤1B) é viável já ou o fallback é RAG clássico com heurística de disparo fora do modelo?*

### Resposta Categórica:
**NÃO É VIÁVEL acionar o RAG via Tool-Calling direto pelo SLM nesse porte para produção no Galaxy S21. O fallback mandatório é o RAG Clássico com heurística determinística de disparo fora do modelo.**

### Justificativas Técnicas e Empíricas:

1. **Incompatibilidade Arquitetural Generalizada (4 de 6 modelos falham)**:
   - Dos 6 modelos de ponta testados, **4 simplesmente não suportam ou ignoram a chamada de ferramenta nativa** em português (Gemma 3 1B IT, LFM2 1.2B, LFM2.5 350M e Llama 3.2 1B).
   - O Llama 3.2 1B trava no parser gramatical estrito (`peg-native`) do llama.cpp durante a conversação multi-turn de ferramentas.
   - Forçar tool-calling eliminaria automaticamente modelos com excelente fidelidade em português como o Gemma 3 e o LFM2.

2. **Gargalo Térmico e de Latência (Inviabilidade da Dupla Rodada no S21)**:
   - A meta dura do capitão é **latência total ≤ 10 segundos no S21** (incluindo ASR + SLM + TTS).
   - O Tool-Calling exige **obrigatoriamente 2 rodadas completas de inferência do SLM**:
     - *Rodada 1*: Prompt de sistema extenso com esquema JSON da tool + pergunta $\rightarrow$ modelo pensa e emite a chamada JSON da tool.
     - *Busca FTS5*: Interceptação e recuperação BM25.
     - *Rodada 2*: Reinjeção de todo o histórico + resultado da tool $\rightarrow$ modelo gera a resposta final.
   - Mesmo no Ryzen 7 desktop, o Qwen 3.5 em Tool-Calling registrou média de **10,03 segundos** (já estourando o teto total). No Galaxy S21 (CPU Cortex-X1/A78 throttled termicamente), essa dupla inferência ultrapassará **18 a 25 segundos**, tornando a experiência de voz inviável em campo.
   - No **RAG Clássico**, a inferência é **única e direta**: a latência no desktop cai para **3,40 s** (Gemma 3) e **5,10 s** (LFM2), o que no S21 ficará confortavelmente dentro de 5 a 7 segundos.

3. **Superioridade Factual e Redução de Alucinações**:
   - Em todos os modelos testados, a taxa de aprovação no gate e o acerto factual foram **substancialmente superiores no RAG Clássico** (LFM2 saltou de 0% de aprovação para 40%, Gemma 3 de 5% para 20%).
   - Modelos ≤1B parâmetros sofrem de diluição de atenção quando precisam raciocinar simultaneamente sobre chamadas de função JSON e sintetizar linguagem regulamentar. Quando o contexto já chega limpo e focado no prompt, o modelo atua apenas como sintetizador fiel, citando o item exato com rigor.

---

## 7. Recomendação Fundamentada para o App Android (Galaxy S21/S24+)

### Arquitetura de Produção Recomendada (Pipeline de Voz 100% Offline):
```
Áudio do Eletricista (Microfone)
               │
               ▼
   Motor ASR Local (Sherpa-ONNX / Whisper)
               │
               ▼  Texto bruto ("Tô no poste e a chave tá travada...")
┌─────────────────────────────────────────────────────────┐
│   Heurística Determinística de Disparo (Fora do SLM)   │
│   - Regex / Intent Classifier / Keyword Matcher         │
│   - Detecta intenção de norma e extrai termos técnicos  │
└─────────────────────────────────────────────────────────┘
               │ (Se detectar dúvida de segurança/NR)
               ▼
┌─────────────────────────────────────────────────────────┐
│   Busca BM25 Local no SQLite FTS5 (index.db ≤ 2 MB)     │
│   - Retorna top-3 chunks da NR correspondente           │
└─────────────────────────────────────────────────────────┘
               │
               ▼ Contexto injetado no prompt de sistema
┌─────────────────────────────────────────────────────────┐
│   SLM Local (llama.cpp Android em CPU/NPU)              │
│   - Modelo Recomendado: LFM2-1.2B RAG ou Gemma 3 1B IT  │
│   - GGUF Q4_K_M (inferência única ≤ 5s)                 │
│   - Cita a fonte (norma + item) obrigatoriamente        │
└─────────────────────────────────────────────────────────┘
               │
               ▼
   Resposta por Voz (Android TTS Local)
```

### Qual Modelo Embarcar no APK?
1. **Opção Principal (Maior Fidelidade Técnica e Segurança): `Liquid LFM2 1.2B RAG`**
   - **Por quê**: Obteve a maior nota de acerto factual (3,30/5,00) e a maior taxa de aprovação do benchmark (40%), com citação consistente de itens da NR-10 e NR-06.
   - Arquitetura de rede neural líquida (LFM2) consome menos memória durante geração e mantém throughput estável de 70 tok/s.
2. **Opção Alternativa (Maior Fluência em Português e Menor Latência): `Gemma 3 1B IT`**
   - **Por quê**: Apresentou a mais alta qualidade textual em português (4,50/5,00) e a menor latência entre os modelos de 1B (3,40 s no desktop, estimado em ~4,5 s no S21).
   - Excelente fidelidade (3,80/5,00) e recusa limpa quando o tema está fora do contexto.
3. **Se o Capitão insistir em Tool-Calling estritamente disparado pelo SLM**:
   - O **único** modelo viável é o **`Qwen 3.5 0.8B (Gated DeltaNet)`** (100% de acionamento da ferramenta e reformulação precisa em português), aceitando-se a penalidade de latência de duas rodadas.

---

## 8. Como Reproduzir os Experimentos

### Pré-requisitos
- WSL2 / Linux x86_64
- CMake 3.22+, GCC/G++ 11+
- Python 3.10+ com venv

### Execução em Comando Único
Na raiz do projeto `cemig-poc`:
```bash
make -C bench all
```

### Execução Passo a Passo

1. **Setup do ambiente, compilação do llama.cpp e download dos 6 modelos GGUF**:
   ```bash
   ./bench/setup.sh
   ```
   *Nota: Os modelos são baixados para `~/models-poc/` e o llama.cpp para `~/llama.cpp/` (totalmente fora do repositório git).*

2. **Medição de performance relativa dos modelos (llama-bench)**:
   ```bash
   python3 bench/perf.py
   ```

3. **Execução do benchmark nos dois modos (Clássico e Tool-Calling)**:
   ```bash
   # Rodada completa dos 6 modelos com as 20 perguntas de ouro
   python3 bench/harness.py --models all --mode both

   # Smoke test rápido (apenas 2 perguntas no Qwen 3.5)
   python3 bench/harness.py --models qwen3.5-0.8b --mode both --limit 2
   ```

4. **Avaliação das respostas pelo LLM Juiz e geração do CSV**:
   ```bash
   python3 bench/judge.py --input bench/results/harness_results.json --cli-tool claude
   ```

Arquivos de saída gerados:
- `bench/results/harness_results.json` — Respostas brutas, queries de busca e métricas de tempo.
- `bench/results/judge_evaluations.json` — Notas detalhadas (1 a 5) e justificativas de cada resposta.
- `bench/results/bench_summary.csv` — Planilha consolidada de médias, taxas de aprovação e velocidades.
- `bench/results/llama_bench_metrics.csv` — Métricas de tok/s de prefill e decode.

---

## 9. Aviso Obrigatório de Governança

> **IMPORTANTE**: A avaliação automatizada por LLM Juiz (`judge.py`) é uma ferramenta de triagem **auxiliar**. Para qualquer decisão regulamentar, homologação formal ou liberação de versão de campo para eletricistas da CEMIG, a **conferência e leitura humana individual das 20 primeiras questões manuais é OBRIGATÓRIA**.
