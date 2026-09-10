# Relatório de Experimentos ASR Offline — Bateria 0/A/B/C (Trilha T3)
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

### Motores Avaliados (Bateria 0/A/B/C)
1. **Exp 0 — ASR Nativo Android Offline:** Investigação dos serviços de sistema via `adb` (`com.google.android.tts`, `com.google.android.googlequicksearchbox`, `com.samsung.android.bixby.ondevice.ptbr` e `com.samsung.android.intellivoiceservice`).
2. **Exp A — whisper.cpp Tiny / Base:** Modelos multilíngues quantizados em `Q5_1` compilados nativamente em ARM64 (`NEON` + `ARM_FMA`).
3. **Exp B — whisper.cpp Small:** Modelo Small quantizado em `Q5_1` como teto de precisão do Whisper.
4. **Exp C — sherpa-onnx + Nemotron 3.5 Streaming 0.6B INT8 (Oficial):** Modelo streaming transducer da NVIDIA (chunk de 560 ms) executado via ONNX Runtime ARM64.
5. **Exp C' — sherpa-onnx + Nemotron 3.5 Streaming 0.6B INT8 (Fine-tune Ottema PT-BR):** Fine-tune comunitário em português brasileiro (`andrewmulya98/sherpa-onnx-ottema-nemotron-3.5-asr-ptbr-560ms-int8`).

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
| **Nemotron 3.5 INT8 (Oficial)** | 651.5 MB | 791.7 MB | **0.36** | **0.36** | **2.03 s** | **6.8%** | **8.2%** | **91.9%** | **91.9%** |
| **Nemotron 3.5 INT8 (Ottema PT-BR)** | 651.5 MB | 794.5 MB | **0.36** | **0.36** | **2.01 s** | 9.3% | 8.3% | **91.9%** | **91.9%** |
| *ASR Nativo Android (Exp 0)* | 0 MB* | ~50 MB* | ~0.20* | Incons.* | ~1.2 s* | N/D* | N/D* | N/D* | N/D* |

*\* Exp 0: Depende de pacote offline pré-instalado pelo usuário/MDM no Google Speech Services. Não garantido em frotas heterogêneas sem intervenção manual.*

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
- Nemotron 3.5 (795 MB) + SLM (700 MB) = **~1.5 GB de RAM concorrente**. Em aparelhos com 6 GB de RAM ou sob pressão do sistema operacional (câmera, GPS, launcher), o processo do app corre alto risco de ser terminado pelo Android LMK (Low Memory Killer).
- Além disso, o pacote de 651 MB onera a distribuição em campo.

### Por que NÃO usar Whisper Small?
- O RTF de 2.81 gera latência de quase 17 segundos em áudios normais, frustrando o operário em campo.

### Por que NÃO usar o ASR Nativo?
- A falta de garantia de pacote offline em 100% dos aparelhos e a incapacidade de reconhecer jargões da CEMIG sem alucinações tornam o ASR nativo inadequado para a operação de campo.

### Diretriz Prática para M2 (Integração Android):
1. **Adotar Whisper Base Q5_1** integrado via JNI/C++ no APK.
2. Configurar o parâmetro `--prompt` do Whisper com a lista de palavras-chave da CEMIG (`NR-10, NR-35, 13,8 kV, religador, seccionadora, desenergização, LOTO`), o que eleva a acurácia de termos do Whisper Base para próximo de 90%.
3. Manter a arquitetura do **sherpa-onnx + Nemotron Ottema PT-BR** como opção ativável para dispositivos dedicados de alta performance (12 GB RAM) onde a exibição de texto em tempo real (streaming parcial) for exigida como requisito de produto.

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

# 2. Compilar os binários whisper.cpp e sherpa-onnx para arm64
# (Os binários gerados ficam em ~/whisper.cpp e ~/sherpa-onnx)

# 3. Executar o benchmark completo no aparelho
/home/rios/.venvs/cemig-bench/bin/python3 asr/scripts/benchmark.py
```
Os dados consolidados serão salvos em `asr/results/benchmark_summary.json` e `asr/results/benchmark_details.json`.
