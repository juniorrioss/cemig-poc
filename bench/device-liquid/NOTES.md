# Mapeamento da Stack On-Device Android da Liquid AI

Data da análise: Março 2025  
Fontes oficiais consultadas:
- https://docs.liquid.ai/deployment/on-device/llama-cpp/mobile
- https://docs.liquid.ai/guides/use-case-evaluation
- https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting
- https://docs.liquid.ai/lfm/help/deprecations
- https://docs.liquid.ai/guides/migration-guide
- https://docs.liquid.ai/guides/hardware-evaluation
- https://docs.liquid.ai/lfm/key-concepts/tool-use
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF

---

## 1. Resumo Executivo da Stack Liquid para Android

A Liquid AI **não mantém um runtime C/C++ ou engine de inferência próprio** para dispositivos móveis.  
A estratégia oficial da Liquid para Android e iOS é **embedar diretamente a biblioteca `llama.cpp` upstream via C API / NDK / CMake**.

O antigo **LEAP SDK** (`ai.liquid.leap:leap-sdk`) e o formato proprietário **LEAP Bundle** (`.bundle` / `leap-bundle`) estão **OFICIALMENTE DEPRECIADOS** pela Liquid. A própria Liquid declara que o LEAP SDK era apenas uma camada fina (wrapper JNI) sobre o `llama.cpp` e recomenda explicitamente que novas aplicações usem `llama.cpp` diretamente.

| Componente | Recomendação Oficial da Liquid | Situação Atual |
| :--- | :--- | :--- |
| **Engine On-Device** | `llama.cpp` (upstream C/C++ API via Android NDK) | **Ativo e Recomendado** |
| **SDK Proprietário** | LEAP SDK (`ai.liquid.leap:leap-sdk:0.10.7`) | **DEPRECATED** (sem updates) |
| **Formato de Modelo** | GGUF padrão (`.gguf` via Hugging Face `LiquidAI/`) | **Ativo e Recomendado** |
| **Formato de Empacotamento** | LEAP Model Bundling Service (`leap-bundle`) | **DEPRECATED** |
| **Quantização Especial** | **QAD-Q4_0** (Quantization-Aware Distillation) | **Recomendado da Liquid** |

---

## 2. Evidências Textuais e Referências Oficiais

### 2.1 Depreciação Formal do LEAP SDK e Bundles
Fonte: [https://docs.liquid.ai/lfm/help/deprecations](https://docs.liquid.ai/lfm/help/deprecations)

> **Deprecated SDKs**
> - **LEAP SDK (iOS, Android, JVM, Kotlin/Native)**  
>   *Recommended replacement*: `llama.cpp` used directly — see Build with llama.cpp (Migrating from LEAP SDK).  
>   *"The LEAP SDK was a wrapper around llama.cpp. Published artifacts remain available but receive no further updates; the archived reference stays online."*
> - **LEAP Model Bundling Service / leap-bundle**  
>   *Recommended replacement*: Download GGUF files from Hugging Face and run them with `llama.cpp`.

Fonte adicional nos exemplos de Android (`leap-koog-agent`, `recipe-generator`, `slogan-generator`):
> *"This example uses the LEAP SDK, which is deprecated. It remains a useful architectural reference, but for new Android apps embed llama.cpp directly — see iOS & Android and Migrating from LEAP SDK."*

### 2.2 Recomendação Oficial de Integração no Android
Fonte: [https://docs.liquid.ai/deployment/on-device/llama-cpp/mobile#android-gradle-%2B-ndk](https://docs.liquid.ai/deployment/on-device/llama-cpp/mobile#android-gradle-%2B-ndk)

> *"llama.cpp is a dependency-free C/C++ library, so it links straight into a mobile app. Every LFM checkpoint ships as GGUF on Hugging Face (LiquidAI), and upstream llama.cpp supports the LFM2 architecture, LFM2-VL projectors, and LFM2/LFM2.5 tool-call parsing. No wrapper SDK is required.*  
> *On a phone, run the library in-process through the C API as shown here."*

A documentação da Liquid aponta para duas formas de integração:
1. **Via `examples/llama.android` do upstream**:  
   Módulo Gradle `lib` que compila `llama.cpp` via NDK, inclui kernels ARM até SME2 com detecção dinâmica e expõe a classe Kotlin `InferenceEngine` / `AiChat`.
2. **Via CMake nativo no NDK**:  
   Adicionar `llama.cpp` diretamente como `add_subdirectory` desativando alvos de desktop/servidor:
   ```cmake
   set(LLAMA_BUILD_COMMON OFF)
   set(LLAMA_BUILD_TESTS OFF)
   set(LLAMA_BUILD_EXAMPLES OFF)
   set(LLAMA_BUILD_TOOLS OFF)
   set(LLAMA_BUILD_SERVER OFF)
   add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp build-llama)
   ```

### 2.3 Recomendações de Otimização Mobile (Tuning)
Fonte: [https://docs.liquid.ai/deployment/on-device/llama-cpp/mobile#4-tune-for-mobile](https://docs.liquid.ai/deployment/on-device/llama-cpp/mobile#4-tune-for-mobile)

- **Memória & Mmap**: `use_mmap = true` (padrão do llama.cpp). Os pesos são mapeados como páginas de arquivo (*file-backed*), o que não consome RSS anônima e torna o processo imune à maioria dos triggers agressivos de Low Memory Killer (LMK).
- **Threads de CPU**: Usar exclusivamente núcleos de alta performance. Recomendação da Liquid: `n_threads = activeProcessorCount - 2`. Exceder os núcleos de performance (jogando threads nos Cortex-A520 de eficiência) degrada a vazão de decodificação. No Galaxy S24+ (Exynos 2400: 10 cores, sendo 1x X4 + 5x A720 + 4x A520), o valor ótimo é rigorosamente **6 threads**.
- **Aceleração Backend**: "CPU é o padrão seguro para Android; backends Vulkan e OpenCL existem mas exigem validação específica por aparelho."

### 2.4 Parâmetros de Inferência e Sampling Recomendados
Fonte: [https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting](https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting) e Model Card do LFM2.5-1.2B-Instruct:

- **Template**: ChatML (`<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`)
- **Temperature**: `0.1` (determinístico, ideal para tarefas técnicas de NR)
- **Top-K**: `50`
- **Repetition Penalty**: `1.05`
- **Tool-Calling**: Formato Pythonic nativo entre sentinelas `<|tool_call_start|>[retriever(query="...")]<|tool_call_end|>`

### 2.5 O que é o QAD (Quantization-Aware Distillation)?
Fonte: Hugging Face Model Card `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`

> *"The Quantization-Aware Distillation (QAD) checkpoint is available as `LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf`. This is distinct from the post-training-quantized `LFM2.5-1.2B-Instruct-Q4_0.gguf`; both use the GGUF Q4_0 format."*

No pipeline tradicional de quantização pós-treino (PTQ), o modelo FP16 é discretizado para 4 bits em blocos com fatores de escala, o que costuma causar leve perda de fidelidade léxica e raciocínio.  
No **QAD**, a Liquid treina o modelo estudante sob supervisão direta do modelo professor em ponto flutuante, aplicando a quantização durante a destilação. O resultado é um artefato em formato `Q4_0` com a velocidade máxima dos kernels Arm KleidiAI de 4 bits simples, mas com qualidade de representação superior.

---

## 3. Conclusão da Análise Arquitetural

1. **Existe um SDK ou binário novo da Liquid para testar contra o llama.cpp?**  
   **Não.** A stack recomendada pela Liquid **é** o próprio `llama.cpp`. O LEAP SDK está arquivado/depreciado e não oferece runtime diferenciado.
2. **Qual é o verdadeiro comparativo de "Stack Liquid" vs "Vanilla"?**  
   A comparação concreta proposta pelo ecossistema Liquid no hardware reside em dois eixos:
   - **Quantização & Qualidade**: `QAD-Q4_0` (Quantization-Aware Distillation da Liquid) vs `Q4_K_M` (k-quant vanilla do llama.cpp).
   - **Parâmetros de Sampling e Formatação**: Parâmetros recomendados da Liquid (`temp=0.1`, `top_k=50`, `repeat_penalty=1.05`, ChatML) vs defaults genéricos.
   - **Vazão / Latência**: Verificar se o formato QAD-Q4_0 entrega ganhos de prefill/decode em relação ao Q4_K_M no Galaxy S24+ com kernels Arm KleidiAI.
