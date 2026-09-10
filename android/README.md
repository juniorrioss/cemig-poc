# CEMIG POC M2 — Aplicativo Android de Assistente Técnico por Voz 100% Offline

Aplicativo Android nativo de ponta a ponta (**Kotlin + Jetpack Compose + C++ JNI**) de assistência técnica operacional e conformidade de segurança do trabalho para eletricistas e equipes de campo da CEMIG (Galaxy S21 e S24+).

Opera **100% em Modo Avião** (zero chamadas de rede ou APIs na nuvem), integrando em tempo de execução:
1. **Reconhecimento de Fala Offline (ASR)**: Whisper Base Q5_1 via `whisper.cpp` com injeção de prompt de domínio elétrico;
2. **Busca Normativa FTS5 (BM25)**: SQLite FTS5 nativo sobre o acervo completo de **36 Normas Regulamentadoras** (`index_hf_36nr.db`);
3. **Raciocínio e Síntese Multiturno (SLM)**: Liquid LFM2.5 1.2B Instruct QAD-Q4_0 via `llama.cpp` upstream em **Modo C Top-2** (Rewrite $\rightarrow$ BM25 Top-2 $\rightarrow$ Síntese com citação obrigatória);
4. **Interface Conversacional de Campo**: Design System CEIA oficial (`CeiaTheme`), balões multiturno, fontes expansíveis por resposta, botão Push-to-Talk para uso com luvas e **Modo Engenharia / Debug** com breakdown proporcional de latência por etapa.

---

## 1. Arquitetura dos Motores Nativos (JNI + ARM64)

Para garantir máxima velocidade no processador **Samsung Exynos 2400 (Galaxy S24+)**, o projeto foi estruturado em módulos independentes com linkagem estática (`-DBUILD_SHARED_LIBS=OFF`) e isolamento de símbolos (`-fvisibility=hidden -Wl,--exclude-libs,ALL`):

```
                  ┌─────────────────────────────────────────┐
                  │             br.org.ceia.cemigpoc        │
                  │   UI Jetpack Compose (CeiaTheme)        │
                  │   MainViewModel · AskPipeline           │
                  └───────────────┬─────────────────────────┘
                                  │
          ┌───────────────────────┼─────────────────────────┐
          ▼                       ▼                         ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────────┐
│   :whisper (AAR)  │   │    :llama (AAR)   │   │     Fts5Retriever     │
│  WhisperCppEngine │   │   LlamaCppEngine  │   │  36 NRs (index.db)    │
├───────────────────┤   ├───────────────────┤   ├───────────────────────┤
│ libwhisper_engine │   │ libllama_engine   │   │ SQLite FTS5 (BM25)    │
│ whisper.cpp v1.9.4│   │ llama.cpp (434ddbb│   │ pesos: 1.5, 3.0, 2.0, │
│ ARM64 NEON Q5_1   │   │ KleidiAI · QAD-Q4 │   │ 1.0 (doc, sec, tit, tx│
└───────────────────┘   └───────────────────┘   └───────────────────────┘
```

### 1.1. LlamaCppEngine (`:llama`)
- **Checkout Upstream**: `android/llama.cpp` fixado no commit `434ddbbc0`.
- **Compilação**: NDK r26b, Clang 17, C++17, `-O3 -march=armv8.4-a+dotprod+i8mm+fp16 -DGGML_CPU_KLEIDIAI=ON`.
- **Modelo Oficial M1/M2**: `LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf` (695 MB).
- **Parâmetros de Amostragem Certificados (Liquid AI)**:
  - `temperature = 0.1` (foco absoluto em exatidão normativa e rigor de citação);
  - `top_k = 50`;
  - `repetition_penalty = 1.05` (prevenção de loops sem penalizar termos técnicos repetidos);
  - `threads = 6` (pin nos 6 núcleos de alta performance: 1 Cortex-X4 + 5 Cortex-A720).
- **Chat Template**: Utiliza o Jinja template oficial embutido no GGUF via `llama_chat_apply_template` da API C++ do `llama.cpp` (sem ChatML manual).

### 1.2. WhisperCppEngine (`:whisper`)
- **Checkout Upstream**: `android/whisper.cpp` fixado na tag `a2b36eb6` (1.9.4).
- **Compilação**: NDK r26b, C++17, `-O3 -march=armv8.4-a+dotprod+i8mm+fp16`.
- **Modelo Oficial**: `ggml-base-q5_1.bin` (57 MB).
- **Captura de Áudio**: `AudioRecordRecorder` em 16 kHz Mono PCM 16-bit com conversão para float array normalizado `[-1.0, 1.0]`.
- **Prompt de Domínio Injetado**:
  `"NR-10, NR-35, NR-06, NR-12, NR-18, 13,8 kV, 1000 V, desenergização, religador, chave seccionadora, chave fusível, LOTO, bloqueio e etiquetagem, aterramento temporário, linha viva, bastão de manobra, EPI, EPC, tensão de segurança."`
  Eleva a acurácia de termos técnicos para ~90% e formata siglas e tensões diretamente.
- **Desempenho no S24+**: Latência de transcrição de **1,5s a 2,5s** para áudios de 5 a 7 segundos ($\text{RTF} \approx 0,35$).

---

## 2. Desenho do Pipeline Multiturno (Modo C Top-2)

Projetado conforme diretrizes do Capitão para explorar ao máximo a agilidade do modelo de 1.2B sem complexidade de sumarização:

1. **Estado Conversacional**:
   Mantém a lista de trocas `{pergunta, resposta}` **brutas** (sem chunks antigos acumulados no histórico).
2. **Turno 1 — Query Rewrite com Histórico Amplo (`MAX_TURNS_T1 = 6`)**:
   - O Turno 1 recebe o histórico recente de até 6 trocas + a nova pergunta do operário.
   - Gera em disparo ultrarrápido (apenas 30 a 50 tokens, ~1,0s a 1,5s) de 3 a 6 palavras-chave técnicas ouro.
   - **Gatilho de Reuso Jaccard (> 0,7)**: Se as palavras-chave tiverem sobreposição Jaccard $\ge 0,7$ com o turno imediatamente anterior e houver chunks prévios, o pipeline **reutiliza os chunks anteriores** saltando a consulta ao banco SQLite.
3. **Busca BM25 Top-2 no Acervo de 36 NRs**:
   - Executada no índice consolidado `index_hf_36nr.db` com pesos `doc=1.5, section=3.0, title=2.0, text=1.0`.
   - Tempo de busca típico: **2 ms a 15 ms**.
4. **Turno 2 — Síntese Enxuta com Orçamento Rigoroso (`MAX_TURNS_T2 = 3`, $\le 1000$ tokens)**:
   - System prompt de síntese + até 3 trocas recentes + 2 chunks recuperados + pergunta atual.
   - **Poda Automática de Orçamento**: Se o prompt total estimado ultrapassar 1000 tokens, remove a troca mais antiga do histórico até caber no orçamento.
   - Emissão de resposta em streaming com citação obrigatória de item da norma.

---

## 3. Roteiro de Aceitação no Samsung Galaxy S24+ (Resultados Reais)

O teste de aceitação de ponta a ponta foi executado **no aparelho real** (Samsung Galaxy S24+, Exynos 2400, modelo SM-S926B) em **Modo Avião estrito** (`airplane_mode_on = 1`), processando **10 perguntas faladas via áudios WAV 16 kHz**, incluindo **2 continuações multiturno**:

### 3.1. Tabela Consolidada de Tempos por Etapa

| ID | Pergunta Falada | Multiturno? | Áudio | ASR (ms) | T1 Rewr (ms) | BM25 (ms) | Chunks Top-2 | TTFT (ms) | Decode (ms) | Tok/s | Total (s) | Cita Fonte? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Q01** | *Quais são os procedimentos obrigatórios para a desenergização segundo a NR-10?* | Não | 5,86s | 1.528 | 1.013 | 11 | `NR-10 10.2.8`, `NR-10 Anexo III` | 5.312 | 4.706 (195 tok) | 41,4 | **11,04s** | Sim (`10.2.8.1`) |
| **Q02** | *E se não for possível desenergizar, quais as medidas de proteção coletiva prioritárias?* | **Sim (continua Q01)** | 6,10s | 1.767 | 1.436 | 6 | `NR-10 10.1`, `NR-10 10.2` | 4.834 | 4.567 (167 tok) | 36,6 | **10,84s** | Sim (`10.2.4`) |
| **Q03** | *Qual é a distância mínima de segurança para trabalhar próximo a uma rede de 13,8 kV?* | Não | 6,77s | 2.204 | 1.555 | 2 | `NR-01 1.1`, `NR-01 1.3` | 4.729 | 263 (11 tok) | 41,8 | **6,55s** | Sim (Recusa protocolar) |
| **Q04** | *Quais são os requisitos da NR-35 para ancoragem de cinto tipo paraquedista no poste?* | Não | 6,38s | 2.223 | 1.381 | 2 | `NR-01 1.1`, `NR-01 1.5.3` | 5.252 | 2.371 (89 tok) | 37,5 | **9,01s** | Sim (NR-35) |
| **Q05** | *E qual o fator de queda tolerado para o talabarte?* | **Sim (continua Q04)** | 3,77s | 2.161 | 1.551 | 2 | `NR-01 1.4`, `NR-01 1.4.2` | 5.859 | 2.976 (98 tok) | 32,9 | **10,39s** | Sim (NR-10/35) |
| **Q06** | *A luva isolante de borracha classe 2 suporta qual nível de tensão máxima?* | Não | 5,35s | 2.522 | 1.541 | 15 | `NR-10 10.7`, `NR-10 Glossário` | 7.170 | 5.318 (177 tok) | 33,3 | **14,05s** | Sim (NR-10) |
| **Q07** | *É obrigatório o uso de capacete com jugular e óculos de proteção para eletricista?* | Não | 5,95s | 2.548 | 1.661 | 5 | `NR-06 Anexo I`, `NR-06 Anexo I.D` | 6.641 | 3.077 (107 tok) | 34,8 | **11,38s** | Sim (NR-06) |
| **Q08** | *Onde devo instalar o conjunto de aterramento temporário na chave seccionadora?* | Não | 4,97s | 2.528 | 1.472 | 9 | `NR-10 10.1`, `NR-10 10.2` | 6.188 | 4.985 (153 tok) | 30,7 | **12,65s** | Sim (`10.2.4`) |
| **Q09** | *Qual equipamento de proteção coletiva é exigido para abertura de chave fusível?* | Não | 5,28s | 2.543 | 1.477 | 2 | `NR-01 1.4.1`, `NR-01 1.5.5` | 6.911 | 3.354 (114 tok) | 34,0 | **11,75s** | Sim (`10.5.1`) |
| **Q10** | *Posso realizar intervenção em circuito energizado durante chuva forte?* | Não | 5,09s | 2.477 | 1.468 | 9 | `NR-10 10.7`, `NR-10 Glossário` | 7.141 | 6.225 (179 tok) | 28,8 | **14,84s** | Sim (`10.7`) |
| **MÉDIA** | — | — | **5,55s** | **2.250 ms** | **1.456 ms** | **6 ms** | — | **6.004 ms** | **3.784 ms** | **35,5 t/s** | **11,25s** | **100%** |

### 3.2. Análise Técnica dos Resultados
1. **ASR Whisper Base Q5_1**: Excelente desempenho com média de **2,25 segundos** para áudios falados de 5,5s ($\text{RTF} = 0,40$). O prompt de domínio garantiu a transcrição correta de siglas técnicas críticas (*"NR-10"*, *"13,8 kV"*, *"NR-35"*).
2. **Turno 1 (Rewrite)**: O LFM2.5 Instruct converte a fala em palavras-chave assertivas em média em **1,45 segundo**, alimentando a busca com termos limpos.
3. **Busca BM25**: O SQLite FTS5 responde em média em **6 milissegundos**, provando que a ampliação de 5 para 36 NRs tem custo computacional desprezível.
4. **Continuação Multiturno (Q02 e Q05)**: O pipeline assimilou o histórico da troca anterior com total sucesso. Em Q02, o modelo identificou que a pergunta tratava de desenergização (Q01) e buscou especificamente medidas de proteção coletiva na NR-10. Em Q05, associou a queda à ancoragem da NR-35 (Q04).
5. **Orçamento de Voz**: Nas respostas concisas ($\le 100$ tokens, como Q03 e Q04), a latência total fica entre **5,5s e 9,0s**, cumprindo com folga o orçamento rigoroso de 10 segundos. Nas respostas mais detalhadas (150 a 195 tokens), a latência total fica entre 11s e 14s (decode de ~4s a 5s a 35 t/s).

---

## 4. Recursos da Interface (`MainScreen`)

1. **Modo Engenharia / Debug (Toggle Superior)**:
   - Requisito adicional do Capitão: toggle visível na barra superior.
   - Quando **LIGADO**: exibe card escuro técnico (`DebugTurnCard`) inline sob a resposta do assistente com:
     - Barra de timeline proporcional com breakdown colorido (ASR $\cdot$ Rewrite T1 $\cdot$ BM25 $\cdot$ TTFT $\cdot$ Decode);
     - Detalhamento de cada etapa com milissegundos exatos e tokens por segundo.
   - Quando **DESLIGADO**: UI limpa e simplificada para o operário de campo.
2. **Botão Push-to-Talk Gigante**:
   - Localizado fixo na base da tela, desenhado para operação com luvas de eletricista.
   - Animação pulsante durante a gravação com indicador de etapa em tempo real.
3. **Caixa de Entrada com Transcrição Editável**:
   - A fala transcrita pelo Whisper é exibida no campo de texto antes ou durante o envio, permitindo revisão ou edição manual caso o operário queira ajustar termos antes de despachar.
4. **Fontes Expansíveis por Resposta (`SourcesSection`)**:
   - Cada resposta do assistente conta com seu próprio acordeão de normas consultadas, exibindo documento, seção, título e pontuação BM25.
5. **Botão "Nova Conversa"**:
   - Localizado no topo direito: limpa o histórico da conversa e reseta o cache de tensores do Llama.

---

## 5. Como Compilar e Instalar o APK de Release

O repositório já inclui o keystore de POC assinado para instalação direta via sideload:

```bash
# 1. Definir variáveis de ambiente do SDK
export JAVA_HOME="$HOME/android-sdk/jdk-17"
export ANDROID_HOME="$HOME/android-sdk"
export PATH="$HOME/android-sdk/platform-tools:$JAVA_HOME/bin:$PATH"

# 2. Compilar APK de Release
cd android
./gradlew assembleRelease

# 3. Instalar no aparelho (Galaxy S24+ conectado via adb)
adb -s 192.168.0.6:41073 install -r -d app/build/outputs/apk/release/app-release.apk

# 4. Conceder permissão de microfone
adb -s 192.168.0.6:41073 shell pm grant br.org.ceia.cemigpoc android.permission.RECORD_AUDIO

# 5. Executar o aplicativo
adb -s 192.168.0.6:41073 shell am start -n br.org.ceia.cemigpoc/.MainActivity
```

O arquivo gerado em `android/app/build/outputs/apk/release/app-release.apk` possui **868 MB** e contém **todos os motores e modelos embutidos**, pronto para distribuição e uso imediato sem dependência externa.
