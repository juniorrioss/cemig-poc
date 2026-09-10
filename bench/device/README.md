# Benchmark de SLMs no Dispositivo Físico — Samsung Galaxy S24+

Este diretório contém o pipeline de compilação cruzada, automação via ADB e os resultados empíricos reais de inferência para os 6 modelos SLM Tier 1 (≤1.2B) quantizados em **GGUF Q4_K_M** diretamente no hardware-alvo.

---

## 1. Hardware e Ambiente de Teste

- **Dispositivo**: Samsung Galaxy S24+ (SM-S926B)
- **SoC**: Samsung Exynos 2400 (`erd9945`)
- **CPU**: 10 núcleos arm64 (1× Cortex-X4 @ 3.21 GHz + 2× Cortex-A720 @ 2.90 GHz + 3× Cortex-A720 @ 2.60 GHz + 4× Cortex-A520 @ 1.95 GHz)
- **Memória RAM**: 12 GB LPDDR5X (11,4 GiB visíveis ao SO)
- **Sistema Operacional**: Android 16 (plataforma arm64-v8a)
- **Modo de Conexão**: ADB over Wi-Fi (`192.168.0.6:41073`)
- **Alocação no Aparelho**: Somente `/data/local/tmp/poc/` (espaço temporário, isolado de dados de usuário, limpo ao término de cada rodada).

---

## 2. Toolchain e Compilação Cruzada

Os binários nativos foram compilados via CMake e Android NDK no host Linux:

- **NDK**: Android NDK r26b (`26.1.10909125`)
- **Compilador**: Clang 17.0.2 (`aarch64-linux-android31`)
- **Alvo**: Android API 31, ABI `arm64-v8a`
- **Flags ARM validadas**:
  ```bash
  -march=armv8.4-a+dotprod+i8mm+fp16
  ```
  Ativa suporte a Dot Product (`HAVE_DOTPROD`), Matriz Int8 (`HAVE_MATMUL_INT8`) e Aritmética Vetorial FP16 (`HAVE_FP16_VECTOR_ARITHMETIC`).
- **Otimizações do llama.cpp**:
  - `GGML_CPU_KLEIDIAI=ON`: Micro-kernels KleidiAI da ARM habilitados para aceleração de tensores em núcleos Cortex.
  - `BUILD_SHARED_LIBS=OFF`: Binários estaticamente linkados (`llama-bench` e `llama-cli`), sem dependências externas de `.so` além de `libc`, `libm` e `libdl`.
  - Binários finais reduzidos com `llvm-strip` (5,1 MB para `llama-bench`, 14 MB para `llama-cli`).

### Alocação de Threads no Exynos 2400
Nos testes empíricos de sintonia no Exynos 2400:
- Utilizar **6 threads** (`-t 6`) concentra a execução exatamente nos **6 núcleos de alta performance** (1× Cortex-X4 + 5× Cortex-A720).
- Utilizar 8 ou 10 threads força a concorrência nos 4 núcleos pequenos Cortex-A520 (eficiência), causando descompasso na barreira de sincronização OpenMP/pthreads e **degradando o prefill em mais de 30%**.

---

## 3. Tabela Consolidada de Resultados

Metodologia de teste:
- **Prompt**: 1500 tokens (`pp1500`) — simula injeção de contexto normativo RAG (trechos da NR).
- **Geração**: 128 tokens (`tg128`) — simula resposta sintetizada em voz.
- **Repetições**: 3 rodadas por modelo no `llama-bench`.
- **Pico de RAM**: Monitoramento de `VmHWM` via `/proc/<pid>/status` a cada 50ms.
- **Carga Fria**: Tempo de `load time` da primeira inicialização do contexto a partir do armazenamento flash.
- **Térmica**: Leitura de sensores do HAL (`dumpsys thermalservice`) antes e após a bateria.

| Modelo | Parâmetros | GGUF Q4_K_M | Prefill tok/s (pp1500) | Decode tok/s (tg128) | Pico RAM (MiB) | Carga Fria (s) | Temp AP Final (Δ) | Latência S24+ (1500+100) | Orçamento ≤10s (S24+) | Latência Proj. S21 (÷2.25×) | Orçamento ≤10s (S21) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Liquid LFM2.5 350M** | 0.35B | 218.7 MiB | **400.55** (±3.5) | **75.24** (±18.5) | **512.2 MiB** | **6.82s** | 60.0°C (+6.4°C) | **5.07s** | **SIM** | **11.42s** | NÃO (Quase) |
| **Qwen 3 0.6B Instruct** | 0.6B | 378.3 MiB | **143.65** (±1.4) | **38.24** (±3.5) | **989.2 MiB** | **14.61s** | 61.3°C (+12.1°C) | **13.06s** | NÃO (+3.1s) | **29.38s** | NÃO |
| **Qwen 3.5 0.8B DeltaNet** | 0.8B | 507.8 MiB | **141.89** (±0.5) | **27.64** (±1.6) | **1167.0 MiB** | **18.49s** | 59.7°C (+0.9°C) | **14.19s** | NÃO (+4.2s) | **31.92s** | NÃO |
| **Liquid LFM2 1.2B RAG** | 1.2B | 697.0 MiB | **101.31** (±2.2) | **25.92** (±3.9) | **1487.6 MiB** | **19.70s** | 62.3°C (+0.2°C) | **18.66s** | NÃO (+8.7s) | **42.00s** | NÃO |
| **Meta Llama 3.2 1B** | 1.2B | 770.3 MiB | **103.68** (±3.6) | **23.23** (±3.7) | **1690.4 MiB** | **18.53s** | 63.1°C (+6.6°C) | **18.77s** | NÃO (+8.8s) | **42.24s** | NÃO |
| **Gemma 3 1B IT** | 1.0B | 768.7 MiB | **65.15** (±2.2) | **22.88** (±3.3) | **1398.6 MiB** | **26.53s** | 62.1°C (+5.9°C) | **27.39s** | NÃO (+17.4s) | **61.64s** | NÃO |

---

## 4. Análise do Orçamento de Latência (Meta: ≤ 10s)

O orçamento de latência para a aplicação móvel da CEMIG considera:
$$T_{\text{total}} = T_{\text{prefill}} + T_{\text{decode}} = \frac{1500}{\text{prefill tok/s}} + \frac{100}{\text{decode tok/s}}$$

### Leitura Direta no Galaxy S24+
1. **Liquid LFM2.5 350M**: É o **único modelo** que cumpre o orçamento rigoroso de 10s para um contexto completo de 1500 tokens, atingindo **5.07 segundos no total** (3.74s prefill + 1.33s decode).
2. **Qwen 3 0.6B (13.06s)** e **Qwen 3.5 0.8B (14.19s)**: Ultrapassam ligeiramente o orçamento para 1500 tokens. Porém, com **contexto reduzido de RAG (500 a 800 tokens)**, ambos alcançam latência entre **6.0s e 8.5s**, cabendo confortavelmente dentro do orçamento.
3. **Liquid LFM2 1.2B RAG (18.66s)** e **Llama 3.2 1B (18.77s)**: Apresentam desempenho de prefill em torno de 100 tok/s e decode em 23-26 tok/s. Em 1500 tokens, o tempo de processamento de prompt (14.8s) consome sozinho todo o orçamento.
4. **Gemma 3 1B IT (27.39s)**: Embora tenha apresentado a melhor qualidade em Português nos testes de bancada, a arquitetura de atenção do Gemma 3 sofreu severa penalidade no pré-processamento de prompt na CPU móvel (65 tok/s), resultando em 23s apenas para ler o contexto.

### Projeção para o Galaxy S21 (Exynos 2100)
> *Nota de transparência: Esta é uma projeção analítica baseada nas especificações de largura de banda e IPC entre as plataformas Exynos 2400 (S24+) e Exynos 2100 (S21).*

- **Fator de desaceleração estimado**: ~2.0× a 2.5× (adotado **2.25×** conservador).
- **Resultado da projeção**:
  - Nenhum modelo com contexto de 1500 tokens atinge ≤10s no Galaxy S21. O **LFM2.5 350M** atinge ~11.4s (muito próximo da meta).
  - Para garantir ≤10s no S21 em produção, a arquitetura de RAG precisa limitar o contexto injetado a **≤ 500 tokens** para modelos de 350M–600M, ou recorrer a respostas em streaming de áudio token-a-token (TTS incremental com time-to-first-token).

---

## 5. Teste de Sanity em Português (llama-cli)

**Prompt avaliado**:
> *"Qual é o equipamento de proteção individual essencial para trabalho em altura segundo a NR-35? Responda de forma sucinta em uma frase."*

Resultados obtidos diretamente no dispositivo:

| Modelo | Resposta Gerada no Dispositivo | Avaliação Qualitativa |
| :--- | :--- | :--- |
| **Liquid LFM2.5 350M** | *"Equipamento de proteção individual essencial para trabalho em altura segundo a NR-35 inclui luvas, óculos de proteção e capacetes."* | Sintaxe PT correta e concisa; alucina luvas/óculos antes de cinto para altura, mas responde diretamente sem preâmbulo. |
| **Qwen 3 0.6B** | *`[Start thinking] Okay, the user is asking about the essential personal protection equipment (PPE)...`* | Vazou cadeia de raciocínio interna em inglês (`thinking tokens`) antes de responder. Requer stripping forçado de tags `<think>`. |
| **Qwen 3.5 0.8B** | *`[Start thinking] Thinking Process: 1. Analyze the Request: * Topic: Essential Personal Protective Equipment...`* | Similar ao Qwen 3: vazou planejamento interno de raciocínio em inglês em modo CLI direto. |
| **Gemma 3 1B IT** | *"A NR-35 exige o uso de equipamentos de proteção individual (EPIs) adequados, como capacete, cinto de segurança, luvas e outros, para cada tarefa realizada em altura."* | **Melhor resposta técnica**: citou explicitamente o cinto de segurança, gramática perfeita em PT-BR e respeitou o limite de uma frase. |
| **Liquid LFM2 1.2B RAG** | *"De acordo com a NR-35, o equipamento essencial de proteção individual para trabalho em altura inclui capacetes, coletes salva-vidas, luvas, botas de segurança, guarda-chuva, guarda-torso..."* | Fluência razoável, porém gerou alucinações absurdas para contexto de eletricidade ("coletes salva-vidas", "guarda-chuva"). |
| **Meta Llama 3.2 1B** | *"De acordo com a NR-35, o equipamento de proteção individual essencial para trabalho em altura é o capacete com visor de proteção, luvas, calçados de suporte e cordas de segurança."* | Boa estrutura frasal, citou cordas de segurança e capacete. |

---

## 6. Comportamento de Memória e Térmica

1. **Uso de RAM**:
   - O consumo variou de **512 MiB** (`lfm2.5-350m`) a **1.69 GiB** (`llama-3.2-1b`).
   - Todos os modelos operam confortavelmente dentro dos 12 GB do Galaxy S24+ (e cabem nos 8 GB do Galaxy S21) sem risco de disparar o OOM Killer do Android.
2. **Estabilidade Térmica**:
   - A temperatura do processador (AP) variou entre **49.2°C e 63.1°C** durante a execução intensa de 3 repetições completas com prefill de 1500 tokens.
   - A temperatura da bateria aumentou moderadamente (+6.6°C acumulados ao longo de todos os 6 testes contínuos), sem acionamento de thermal throttling agressivo do sistema.
3. **Tempo de Carga do Modelo (Flash para RAM)**:
   - A carga a frio variou de **6.8 segundos** (350M) a **26.5 segundos** (1B Gemma).
   - Isso reforça que o modelo deve ser carregado como **processo residente em background** (ou via serviço persistente com `mmap`/`mlock`), nunca carregado sob demanda no momento do clique do usuário.

---

## 7. Como Reproduzir os Testes

### Pré-requisitos
1. Aparelho Android pareado via ADB (USB ou Wi-Fi).
2. Android NDK r26b em `~/android-sdk/ndk/26.1.10909125`.
3. Repositório `llama.cpp` clonado em `~/llama.cpp`.
4. Modelos GGUF Q4_K_M baixados em `~/models-poc/`.

### Passo 1: Compilação dos Binários
```bash
./bench/device/build-android.sh
```
Gera os binários nativos estáticos em `bench/device/build-android/bin/`.

### Passo 2: Execução da Bateria Completa
```bash
./bench/device/run-device-bench.sh --serial 192.168.0.6:41073
```
Parâmetros adicionais suportados:
- `--models lfm2.5-350m qwen3.5-0.8b` (executa apenas modelos selecionados)
- `--threads 6` (configura núcleos de inferência)
- `--repetitions 3` (número de repetições)
- `--clean-only` (limpa os temporários no celular)

Resultados estruturados são salvos em `bench/device/results/summary.json` e `bench/device/results/summary.csv`.
