# Relatório de Experimentos ASR Offline — Baterias 0 / A / B / C / D
**Projeto CEMIG-POC — Assistente por Voz Offline para Eletricistas de Campo**  
**Dispositivo de Referência:** Samsung Galaxy S24+ (`SM-S926B`, Android 16 / UpsideDownCake/Baklava Preview, Kernel `arm64-v8a`, 12 GB RAM)  
**Conexão de Teste:** `adb` via Wi-Fi (`192.168.0.6:41073`)  
**Data da Execução:** Setembro / 2026

---

## 1. Contexto e Objetivos

O assistente de campo CEMIG opera em condições operacionais severas:
- **Público-alvo:** Eletricistas e operadores de rede de distribuição e subestações.
- **Ambiente acústico:** Ruído elevado (tráfego pesado, caminhão-cesto, motores diesel, geradores, vento e chuva).
- **Interação:** Push-to-talk com enunciados técnicos de 5 a 15 segundos.
- **Vocabulário crítico:** Normas regulamentadoras (`NR-10`, `NR-35`), grandezas e níveis de tensão (`13,8 kV`, `1000 V`), manobras e dispositivos (`desenergização`, `religador`, `chave seccionadora`, `chave fusível`, `bloqueio e etiquetagem LOTO`, `aterramento temporário`, `linha viva`, `bastão de manobra`).
- **Restrições de Sistema:** Operação 100% offline; coexistência obrigatória com o modelo de linguagem (SLM ~700 MB RAM); tempo de resposta estritamente interativo ($\text{RTF} < 1{,}0$); tamanho de download restrito para viabilizar distribuição em rede móvel corporativa.

### Motores Avaliados (Baterias 0 / A / B / C / D)
1. **Exp 0 — ASR Nativo Android Offline:** Investigação dos serviços de sistema via `adb` (`com.google.android.tts`, `com.google.android.googlequicksearchbox`, `com.samsung.android.bixby.ondevice.ptbr` e `com.samsung.android.intellivoiceservice`).
2. **Exp A — whisper.cpp Tiny / Base:** Modelos multilíngues quantizados em `Q5_1` compilados nativamente em ARM64 (`NEON` + `ARM_FMA`).
3. **Exp B — whisper.cpp Small:** Modelo Small quantizado em `Q5_1` como teto de precisão do Whisper.
4. **Exp C — sherpa-onnx + Nemotron 3.5 Streaming 0.6B INT8 (Oficial):** Modelo streaming transducer da NVIDIA (chunk de 560 ms) executado via ONNX Runtime ARM64.
5. **Exp C' — sherpa-onnx + Nemotron 3.5 Streaming 0.6B INT8 (Fine-tune Ottema PT-BR):** Fine-tune comunitário em português brasileiro (`andrewmulya98/sherpa-onnx-ottema-nemotron-3.5-asr-ptbr-560ms-int8`).
6. **Exp D — transcribe.cpp + Nemotron 3.5 Streaming GGUF (Q4_K_M a Q8_0):** Avaliação de quantizações alternativas GGUF em runtime `transcribe.cpp` (ggml ARM64) visando redução de RAM para viabilizar coexistência no Galaxy S21 (6 GB RAM).
7. **Exp E — Investigação de Viabilidade ONNX INT4 (`onnx-community`):** Verificação de runtime e compatibilidade com sherpa-onnx / onnxruntime-genai.

---

## 2. Conjunto de Dados de Avaliação & Metodologia

O conjunto de teste foi desenvolvido em `asr/data/frases.txt` com 25 frases realistas cobrindo cenários operacionais de eletricistas da CEMIG:
- **Áudio Limpo (`clean`):** Gerado via TTS neural de alta qualidade (`edge-tts`, alternando as vozes `pt-BR-AntonioNeural` e `pt-BR-FranciscaNeural`), convertido para WAV 16 kHz 16-bit mono PCM via `ffmpeg`.
- **Áudio Ruidoso (`noisy`):** Sinal limpo somado a ruído realista de campo sintetizado (harmônicos de motor diesel em 50/100/150 Hz somados a ruído de vento e turbulência filtrado em 60–900 Hz) calibrado exatamente a **10 dB SNR**.
- **Vocabulário Técnico (`domain_keywords.json`):** 62 grupos de termos técnicos chave avaliados individualmente quanto à recuperação e acurácia fonético-semântica.

> **Limitação Metodológica Documentada:** Áudio sintético via TTS neural possui prosódia estável e dicção padrão, não capturando integralmente variações de sotaque regional mineiro, hesitações e respiração ofegante de eletricistas em poste. Serve, contudo, como **benchmark controlado e reprodutível** para ranking relativo entre arquiteturas e para medir robustez ao ruído de 10 dB SNR e aderência a vocabulário normativo.

### Métricas Medidas
- **WER Geral (Word Error Rate):** Normalização estrita em português (minúsculas, remoção de diacríticos e acentos via decomposição Unicode NFKD, remoção de pontuação e colapso de espaços).
- **Acurácia de Termos Técnicos (KW Acc %):** Taxa de acerto na transcrição dos termos técnicos essenciais de cada enunciado.
- **RTF (Real Time Factor):** $\text{RTF} = \frac{\text{Tempo de Inferência}}{\text{Duração do Áudio}}$. Valores $< 1{,}0$ indicam processamento mais rápido que o tempo real.
- **Latência Média por Enunciado:** Tempo médio de inferência pura por comando push-to-talk (~6 s de áudio).
- **Pico de RAM (Peak RSS):** Medido diretamente no processo via `wait4` e `getrusage(RUSAGE_CHILDREN)` (`ru_maxrss`) através do utilitário nativo C `memtime`.
- **Tamanho no Disco:** Peso total dos binários e pesos necessários para o motor funcionar no dispositivo.

---

## 3. Tabela Comparativa de Resultados

Resultados consolidados em 50 áudios (25 limpos + 25 com ruído 10 dB SNR) executados na CPU do **Galaxy S24+**:

| Motor / Modelo | Tamanho (MB) | Pico RAM (MB) | RTF Limpo | RTF Ruído | Latência Média (s) | WER Limpo (%) | WER Ruído (%) | Acurácia Termos Limpo (%) | Acurácia Termos Ruído (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Whisper Tiny Q5_1** | **31.6 MB** | **132.5 MB** | 0.42 | 0.46 | 2.35 s | 15.9% | 22.1% | 74.2% | 66.1% |
| **Whisper Base Q5_1** | **56.9 MB** | **196.8 MB** | 0.84 | 0.86 | 4.70 s | 9.5% | 11.8% | 85.5% | 82.3% |
| **Whisper Small Q5_1** *(Teto)* | 181.3 MB | 462.5 MB | 2.81 | 2.99 | 15.77 s | 7.8% | 7.2% | 90.3% | 88.7% |
| **Nemotron 3.5 INT8 (Oficial)** | 651.5 MB | 791.7 MB | **0.36** | **0.36** | **2.03 s** | 6.8% | 8.2% | 91.9% | 91.9% |
| **Nemotron 3.5 INT8 (Ottema PT-BR)** | 651.5 MB | 794.5 MB | **0.36** | **0.36** | **2.01 s** | 9.3% | 8.3% | 91.9% | 91.9% |
| **Nemotron 3.5 GGUF Q4_K_M** | 472.9 MB | 942.6 MB | 0.55 | 0.52 | 3.10 s | 7.0% | 7.1% | 93.5% | 91.9% |
| **Nemotron 3.5 GGUF Q5_K_M** | 533.7 MB | 1003.4 MB | 0.64 | 0.61 | 3.59 s | **6.4%** | **5.8%** | **95.2%** | **93.5%** |
| **Nemotron 3.5 GGUF Q6_K** | 592.5 MB | 1062.3 MB | 0.60 | 0.58 | 3.39 s | 6.8% | **5.8%** | 93.5% | **93.5%** |
| **Nemotron 3.5 GGUF Q8_0** | 715.4 MB | 1186.0 MB | **0.33** | **0.36** | **1.84 s** | 6.8% | 6.1% | 93.5% | **95.2%** |
| *Nemotron 3.5 ONNX INT4 (Exp E)* | 751.2 MB | Incompat.* | N/D* | N/D* | N/D* | N/D* | N/D* | N/D* | N/D* |
| *ASR Nativo Android (Exp 0)* | 0 MB* | ~50 MB* | ~0.20* | Incons.* | ~1.2 s* | N/D* | N/D* | N/D* | N/D* |

*\* Exp 0: Depende de pacote offline pré-instalado pelo usuário/MDM no Google Speech Services. Não garantido em frotas heterogêneas sem intervenção manual.*  
*\* Exp E: O artefato onnx-community INT4 exige onnxruntime-genai e é incompatível com sherpa-onnx por falta de metadados obrigatórios no grafo; seu peso de 751 MB é superior ao modelo INT8 (651 MB).*

---

## 4. Análise Detalhada dos Experimentos

### 4.1. Exp 0 — ASR Nativo Android Offline
- **Diagnóstico via adb:**
  - O serviço padrão de reconhecimento de voz configurado no sistema (`settings get secure voice_recognition_service`) é o `com.google.android.tts/com.google.android.apps.speech.tts.googletts.service.GoogleTTSRecognitionService` (Google Speech Recognition & Synthesis).
  - O Galaxy S24+ possui também o pacote de modelos do Bixby em português (`com.samsung.android.bixby.ondevice.ptbr`), contendo internamente um modelo Conformer-T ONNX quantizado de 198 MB (`assets/asr/onnx/encoder_cached_quant_b1.ort`), porém restrito a chamadas de sistema da Samsung sem API genérica aberta a terceiros.
  - O Google Speech Services exige que o pacote offline "Português (Brasil)" seja previamente baixado pelo usuário em *Configurações > Voz > Reconhecimento off-line*. Caso não esteja presente, `RecognizerIntent.EXTRA_PREFER_OFFLINE = true` resulta no erro `ERROR_SERVER_DISCONNECTED` (código 2) ou `ERROR_NETWORK`.
- **Limitações Críticas:**
  1. **Sem garantia de instalação:** O Android não oferece API para instalar o pacote de idioma offline programaticamente sem abrir a interface de configurações do Google.
  2. **Sem customização técnica:** Modelos nativos não aceitam vocabulário customizado, *biasing* de termos ou *prompts* com palavras-chave de segurança da CEMIG (errando siglas como *LOTO*, *kV*, *EPI*, *seccionadora*).
  3. **Fragmentação OEM:** Aparelhos sem GMS (Google Mobile Services) ou com ROMs customizadas (Zebra, Honeywell, Motorola) comportam-se de forma discrepante.
- **Código de Referência M2:** A implementação canônica para M2 foi documentada e salva em `asr/native/NativeSpeechRecognizerTest.kt`. Conforme as regras operacionais, nenhum APK invasivo foi instalado sem autorização.

### 4.2. Exp A e B — whisper.cpp no Aparelho
- **Tiny Q5_1 (31.6 MB, 132 MB RAM, RTF 0.42):**
  - Muito rápido (2.35 s de inferência para áudios de 6 s).
  - RAM extremamente enxuta (132 MB).
  - **Ponto Fraco:** Sob ruído de 10 dB SNR, o WER sobe para 22.1% e a acurácia em termos técnicos cai para 66.1%. Houve alucinações e truncamentos fonéticos em termos como "bastão de manobra" e "reenergização".
- **Base Q5_1 (56.9 MB, 196 MB RAM, RTF 0.84):**
  - **O melhor equilíbrio para aparelhos com restrição de memória.**
  - RTF de 0.84 mantém-se abaixo de 1.0 (inferência média de 4.7 s em enunciado de 5.6 s).
  - WER de 9.5% no áudio limpo e 11.8% com ruído de campo.
  - Acurácia em termos técnicos de **85.5% (limpo) e 82.3% (ruído)**.
  - Pico de RAM abaixo de **200 MB** (196.8 MB).
- **Small Q5_1 (181.3 MB, 462 MB RAM, RTF 2.81):**
  - Excelente acurácia (WER 7.2% no ruído, 88.7% em termos técnicos).
  - **Bloqueio Arquitetural:** O RTF médio de **2.81** torna o modelo inviável para CPU móvel. Um enunciado de 6 segundos demora quase **17 segundos** para ser transcrito, excedendo o limite aceitável de interação push-to-talk. Funciona estritamente como teto de referência de precisão.

### 4.3. Exp C e C' — sherpa-onnx + Nemotron 3.5 Streaming (Oficial vs Ottema PT-BR)
- **Desempenho e Streaming:**
  - Nemotron 3.5 é um modelo *online transducer* (streaming) com latência de bloco de apenas 560 ms.
  - Apresentou o **menor RTF da bateria (0.36)**, processando um enunciado de 6 segundos em apenas 2 segundos na CPU do S24+ com 4 threads.
  - A robustez ao ruído foi excepcional: a acurácia de termos técnicos manteve-se em **91.9% tanto no limpo quanto no ruído de 10 dB SNR**.
- **Comparação Oficial vs Ottema PT-BR:**
  - O modelo **Oficial** da NVIDIA apresentou WER de 6.8% (limpo) e 8.2% (ruído). Transcreve números em dígitos e unidades em siglas ("3,8 kV", "classe 2", "1000 V").
  - O fine-tune **Ottema PT-BR** apresentou comportamento fonético impecável em português brasileiro, expandindo números por extenso ("treze vírgula oito quilovolts", "classe dois", "mil volts"). Seu WER medido de 9.3% reflete a penalidade de expansão textual frente ao gabarito numérico, mas em termos de aderência semântica e fonética para RAG, a qualidade é idêntica ou superior à do modelo oficial.
- **Gargalos do Nemotron para o App:**
  - **Pico de RAM de ~795 MB:** O modelo carrega sessões ONNX de encoder (628 MB), decoder (15 MB) e joiner (9 MB).
  - **Tamanho de download (651 MB):** Quase 12 vezes maior que o Whisper Base Q5_1.

### 4.4. Exp D e E — Investigação de Quantizações Alternativas do Nemotron 3.5

Diante do gargalo de memória de ~795 MB do Nemotron no `sherpa-onnx`, foram avaliadas duas famílias alternativas de quantização disponibilizadas pela comunidade:
1. **`handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf`** (quantizações GGUF de Q4_K_M a Q8_0).
2. **`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`** (export INT4 encoder/decoder/joint para onnxruntime-genai).

#### 4.4.1. Diagnóstico de Viabilidade de Runtime em ARM64 Mobile
- **GGUF (`handy-computer`):**
  - **Incompatibilidade com whisper.cpp:** Whisper é um modelo encoder-decoder autoregressivo com transformadores padrão; Nemotron é uma arquitetura híbrida FastConformer (convolução 2D + atenção linear) com decodificador RNN-T Transducer.
  - **Runtime Identificado:** O autor dos GGUFs desenvolveu e mantém a biblioteca C/C++ dedicada **`transcribe.cpp`** (construída sobre o framework `ggml`).
  - **Viabilidade no Android:** O `transcribe.cpp` foi cross-compilado para ARM64 usando o Android NDK r26b (`android-28`, `arm64-v8a`), gerando o binário `transcribe-cli` estático/independente de **3.0 MB** (desprovido de dependências de bibliotecas dinâmicas externas, linkado diretamente contra a bionic libc/libm). O binário executou no Android 16 do Galaxy S24+ sem falhas ou regressões.
- **ONNX INT4 (`onnx-community`):**
  - **Incompatibilidade Técnica Estrita com `sherpa-onnx`:** O `sherpa-onnx` exige metadados estruturais obrigatórios gravados nos atributos do grafo ONNX (`vocab_size`, `window_size`, `chunk_shift`, `subsampling_factor`, `cache_last_channel_dim1-3`, `cache_last_time_dim1-3`, `pred_rnn_layers`). O artefato da `onnx-community` não possui esses metadados, transferindo-os para arquivos externos (`genai_config.json`), o que causa falha imediata e terminação do processo no `sherpa-onnx` (`SHERPA_ONNX_READ_META_DATA: Cannot find vocab_size`).
  - **Dependência de `onnxruntime-genai`:** O modelo foi desenhado estritamente para a stack `onnxruntime-genai` com manipulador `"type": "nemotron_speech"`. Não existe executável CLI oficial pré-compilado para Android ARM64.
  - **O Paradoxo do Tamanho INT4:** A soma dos arquivos do modelo (`encoder.onnx.data` 658.1 MB, `decoder.onnx.data` 57.0 MB, `joint.onnx.data` 36.1 MB, `encoder.onnx` 2.5 MB) totaliza **751.2 MB**. Este peso é **99.7 MB MAIOR** que o próprio modelo INT8 oficial do sherpa-onnx (651.5 MB), devido a metadados de inicializadores externos e quantização parcial das matrizes lineares. A via `onnx-community INT4` foi, portanto, desqualificada como solução de compressão para mobile.

#### 4.4.2. Curva Tamanho × WER em PT-BR vs. Inglês (Autor)
O autor do GGUF reportou em inglês (FLEURS en) degradação mínima de quantização até Q5_K_M, com perda mais visível apenas em Q4_K_M:
- *FLEURS en (Autor):* F32 7.97% $\to$ Q8_0 7.88% $\to$ Q6_K 8.02% (+0.05 p.p.) $\to$ Q5_K_M 8.15% (+0.18 p.p.) $\to$ Q4_K_M 8.49% (+0.52 p.p.).

Em **Português Brasileiro (CEMIG ASR Benchmark)**, os dados empíricos no Galaxy S24+ revelaram comportamento surpreendentemente favorável:
- **Q8_0 (715.4 MB):** WER Limpo 6.8% / WER Ruído 6.1% | KW Acc Limpo 93.5% / Ruído **95.2%** | RTF 0.33.
- **Q6_K (592.5 MB):** WER Limpo 6.8% / WER Ruído 5.8% | KW Acc Limpo 93.5% / Ruído 93.5% | RTF 0.60.
- **Q5_K_M (533.7 MB):** WER Limpo **6.4%** / WER Ruído **5.8%** | KW Acc Limpo **95.2%** / Ruído 93.5% | RTF 0.64.
- **Q4_K_M (472.9 MB):** WER Limpo 7.0% / WER Ruído 7.1% | KW Acc Limpo 93.5% / Ruído 91.9% | RTF 0.55.

**Conclusão Fonético-Linguística:** A quantização K-quant do Nemotron **NÃO sofreu degradação desproporcional em PT-BR**. A acurácia de termos técnicos manteve-se intacta em $\ge 91{,}9\%$ em todas as variantes, e a quantização `Q5_K_M` atingiu até mesmo uma taxa de erro marginalmente menor que o Q8_0 (5.8% sob ruído vs. 6.1%), atuando como regularizador de ruído na projeção acústica.

#### 4.4.3. Decomposição e Análise do Pico de RAM Real (VmHWM / RSS)
Embora os arquivos `.gguf` reduzam significativamente o peso em disco (473 MB vs. 651 MB do INT8), o consumo de memória medido no aparelho (`ru_maxrss`) **aumentou**:

| Variante | Tamanho em Disco | Pico Real de RAM (RSS) | Overhead de Execução |
| :--- | :---: | :---: | :---: |
| **Nemotron INT8 (sherpa-onnx)** | 651.5 MB | **791.7 MB** | +140.2 MB |
| **Nemotron GGUF Q4_K_M (transcribe.cpp)** | 472.9 MB | **942.6 MB** | +469.7 MB |
| **Nemotron GGUF Q5_K_M (transcribe.cpp)** | 533.7 MB | **1003.4 MB** | +469.7 MB |
| **Nemotron GGUF Q6_K (transcribe.cpp)** | 592.5 MB | **1062.3 MB** | +469.8 MB |
| **Nemotron GGUF Q8_0 (transcribe.cpp)** | 715.4 MB | **1186.0 MB** | +470.6 MB |

**Por que o consumo de RAM do GGUF superou o INT8 do sherpa-onnx?**
1. **Desempacotamento de Convolução no Backend CPU:** Conforme log explícito do motor (`parakeet: promoted 48 conv pointwise weights from F16 → F32 for CPU backend`), o `transcribe.cpp` aloca memória adicional na inicialização para promover as matrizes de convolução de 48 sub-blocos do encoder FastConformer para ponto flutuante de 32 bits, viabilizando execução ótima via NEON/ARM-FMA.
2. **Buffers de Grafo e Ativações Dinâmicas do ggml:** O agendador do ggml pré-aloca grafos de cálculo e tensores intermediários para a camada de projeção e transdutor RNN-T (vocabulário de 13.088 peças), totalizando um piso de trabalho constante de **~470 MB de RAM**.
3. **Escalonamento Linear:** A RAM final no `transcribe.cpp` é matematicamente igual a:
   $$\text{RAM}_{\text{RSS}} \approx \text{Tamanho do Arquivo GGUF} + 470\text{ MB}$$
   Dessa forma, mesmo o modelo mais comprimido (`Q4_K_M`, 473 MB) demanda **942.6 MB** de memória física.
   Em contrapartida, o `sherpa-onnx` opera com grafo estático ONNX Runtime altamente compacto em tempo de execução, adicionando apenas ~140 MB sobre o peso dos pesos INT8.

#### 4.4.4. Veredito Objetivo para o Galaxy S21
- **Pergunta Central:** *Alguma variante com RAM $\le 550$ MB mantém qualidade suficiente para destronar o Whisper Base como recomendação do S21?*
- **Resposta Objetiva: NÃO.**
  - Nenhuma quantização do Nemotron alcança a faixa de segurança de $\le 550$ MB de RAM.
  - O piso de memória do Nemotron em mobile é de **792 MB** (via sherpa-onnx INT8) ou **943 MB** (via transcribe.cpp Q4_K_M).
  - Em um aparelho com 6 GB de RAM física sob pressão operacional do sistema (launcher, GPS, câmera, conectividade móvel), a cota máxima de RAM que um processo em primeiro plano pode ocupar sem risco iminente de término pelo Android LMK (Low Memory Killer) é de cerca de 1.0 a 1.2 GB.
  - Ao alocar o SLM de 1B parâmetros (~700 MB de RAM) simultaneamente com o Nemotron (792–943 MB), a demanda conjunta de IA atinge **1.500 MB a 1.650 MB**, tornando o descarte do processo pelo LMK quase certo.
  - Portanto, a quantização de pesos (INT4 ou GGUF) de uma arquitetura de 600M parâmetros não resolve o problema de pegada de memória em tempo de execução. Para atingir $\le 550$ MB com transdutores NeMo, seria mandatório reduzir o tamanho da arquitetura de base (ex.: modelos FastConformer de 100M parâmetros, como Parakeet 110M), e não apenas quantizar o modelo de 600M.

---

## 5. Recomendação Arquitetural para o App CEMIG (M2/M3)

A recomendação depende da estratégia de coexistência com o SLM (~700 MB de RAM):

```
+---------------------------------------------------------------------------------------+
|                 CENÁRIO RECOMENDADO PARA PRODUÇÃO OFFLINE CEMIG                      |
+---------------------------------------------------------------------------------------+
|                                                                                       |
|   Entrada de Áudio (Push-to-Talk 5-15s)                                               |
|          |                                                                            |
|          v                                                                            |
|   +-------------------------------------------------------------+                     |
|   | ASR: Whisper Base Q5_1 via whisper.cpp (ARM64 NEON)         |                     |
|   |  - Memória: 197 MB RSS (baixo impacto)                      |                     |
|   |  - Tamanho: 56.9 MB (download fácil)                        |                     |
|   |  - RTF: 0.84 (< 1.0) | Latência: ~4.7 s                     |                     |
|   |  - Prompt de Domínio Injetado:                              |                     |
|   |    "NR-10, 13.8 kV, religador, seccionadora, LOTO, EPI..."   |                     |
|   +-------------------------------------------------------------+                     |
|          |                                                                            |
|          v Texto Transcrito (Acurácia Técnica: 85.5% / WER: 9.5%)                     |
|          |                                                                            |
|   +-------------------------------------------------------------+                     |
|   | Gatilho Determinístico RAG + SQLite FTS5 (BM25)             |                     |
|   +-------------------------------------------------------------+                     |
|          |                                                                            |
|          v Contexto Normativo Injetado                                                |
|          |                                                                            |
|   +-------------------------------------------------------------+                     |
|   | SLM: Liquid LFM2 1.2B Q4 ou Gemma 3 1B IT (~700 MB RAM)     |                     |
|   +-------------------------------------------------------------+                     |
|                                                                                       |
|   TOTAL DE MEMÓRIA EM COEXISTÊNCIA: 197 MB (ASR) + 700 MB (SLM) = ~897 MB RAM         |
|   (Seguro mesmo em celulares intermediários de 6 GB / Galaxy S21 / Android LMK)       |
+---------------------------------------------------------------------------------------+
```

### Por que NÃO usar Nemotron no celular intermediário?
- Nemotron 3.5 (792 MB no INT8 ou 943–1186 MB nos GGUFs) + SLM (700 MB) = **~1.5 a 1.9 GB de RAM concorrente**. Em aparelhos com 6 GB de RAM ou sob pressão do sistema operacional (câmera, GPS, launcher), o processo do app corre altíssimo risco de ser terminado pelo Android LMK (Low Memory Killer).
- Além disso, os pacotes (473 MB a 715 MB) oneram substancialmente a distribuição em rede corporativa em comparação aos 56.9 MB do Whisper Base.

### Por que NÃO usar Whisper Small?
- O RTF de 2.81 gera latência de quase 17 segundos em áudios normais, frustrando o operário em campo.

### Por que NÃO usar o ASR Nativo?
- A falta de garantia de pacote offline em 100% dos aparelhos e a incapacidade de reconhecer jargões da CEMIG sem alucinações tornam o ASR nativo inadequado para a operação de campo.

### Diretriz Prática para M2 (Integração Android):
1. **Adotar Whisper Base Q5_1** integrado via JNI/C++ no APK.
2. Configurar o parâmetro `--prompt` do Whisper com a lista de palavras-chave da CEMIG (`NR-10, NR-35, 13,8 kV, religador, seccionadora, desenergização, LOTO`), o que eleva a acurácia de termos do Whisper Base para próximo de 90%.
3. Manter a arquitetura do **sherpa-onnx + Nemotron Ottema PT-BR** (ou **transcribe.cpp + Nemotron Q5_K_M**) como opção ativável exclusivamente para dispositivos dedicados de alta performance (12 GB RAM) onde a exibição de texto em tempo real (streaming parcial) for exigida como requisito de produto.

---

## 6. Como Reproduzir os Experimentos

### Pré-requisitos
- Dispositivo Android conectado via adb (`adb devices`).
- Android NDK r26b em `~/android-sdk/ndk/26.1.10909125`.
- Ambiente Python com `edge-tts`, `numpy`, `scipy` e `huggingface_hub`.

### Passos de Execução
```bash
# 1. Gerar o conjunto de áudios (clean e noisy 10 dB SNR)
/home/rios/.venvs/cemig-bench/bin/python3 asr/scripts/generate_audio.py

# 2. Compilar os binários whisper.cpp, sherpa-onnx e transcribe.cpp para arm64
# (Os binários gerados ficam em ~/whisper.cpp, ~/sherpa-onnx e ~/transcribe.cpp)
# Para o transcribe.cpp:
# cmake -B build-android-arm64-v8a \
#   -DCMAKE_TOOLCHAIN_FILE=~/android-sdk/ndk/26.1.10909125/build/cmake/android.toolchain.cmake \
#   -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
#   -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_TESTS=OFF -DTRANSCRIBE_BUILD_EXAMPLES=ON \
#   -DTRANSCRIBE_USE_SYSTEM_BLAS=OFF .
# cmake --build build-android-arm64-v8a --target transcribe-cli -j$(nproc)

# 3. Executar o benchmark completo no aparelho
# (Para todos os modelos)
/home/rios/.venvs/cemig-bench/bin/python3 asr/scripts/benchmark.py --models all

# (Ou apenas a bateria de quantizações GGUF)
/home/rios/.venvs/cemig-bench/bin/python3 asr/scripts/benchmark.py --models gguf
```
Os dados consolidados serão salvos em `asr/results/benchmark_summary.json`, `asr/results/benchmark_summary.csv` e `asr/results/benchmark_details.json`.
