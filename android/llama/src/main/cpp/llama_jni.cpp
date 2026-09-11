#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include <algorithm>
#include <cstring>
#include <cmath>
#include "llama.h"

#define TAG "LlamaEngineJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Defaults recomendados pelo Model Card da Liquid AI para LFM2.5 Instruct / RAG:
// - temperature: 0.1 (foco em fidelidade factual às normas técnicas)
// - top_p: 1.0 (ou 0.9 em amostragem probabilística)
// - min_p: 0.05 (filtro de cauda longa)
// - repetition_penalty: 1.05 (evita loops sem penalizar termos técnicos repetidos)
// - n_threads: 6 (pin nos 6 núcleos de alta performance Cortex-X4 / A720 do Galaxy S24+)

struct LlamaEngineContext {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    std::vector<llama_token> cached_tokens;
    std::mutex mtx;
    int32_t n_ctx = 2048;
    int32_t n_batch = 512;
};

// -----------------------------------------------------------------------------
// Supressão de reasoning (thinking-OFF) para modelos que forçam <think> no template.
//
// Motivação (task poc-engine-upgrade): o chat_template do LFM2.5-2.6B termina em
// `<|im_start|>assistant\n<think>` no `add_generation_prompt` e NÃO expõe um switch
// `enable_thinking` (verificado no GGUF). A C-API `llama_chat_apply_template` sempre
// prima o bloco de raciocínio -> 700 tok de <think>, 44-54 s/resposta (inviável p/ voz).
// O caminho `--reasoning-budget 0` do llama-server vive no path jinja do servidor,
// ausente no JNI.
//
// Fix auto-contido, réplica do `--reasoning-budget 0` do servidor (sem bump de submódulo):
//   (1) ANEXA um bloco de raciocínio VAZIO `<think></think>\n` ao prompt (nativeApplyChatTemplate);
//   (2) BANE a reabertura do token `<think>` via logit-bias na geração (nativeGenerate).
// Provado na RTX 5070 e no S24+ (0/10 respostas com <think>, todas citam a NR correta).
// Alternativas descartadas por evidência (ver README_ENGINE_UPGRADE.md): banir só o token
// falha (o modelo soletra a tag por sub-tokens); FORÇAR <think>/</think> como tokens decodificados
// separadamente fazia o 2.6B ECOAR o prompt (estado shortconv/recorrente do LFM2.5 híbrido
// difere entre decode em lote e incremental). Anexar como TEXTO evita isso.
// No 1.2B QAD (Instruct, não-reasoning) o flag é false -> no-op seguro (default do app).
//
// find_special_token permanece útil para resolver o id de `<think>`/`</think>`.
// -----------------------------------------------------------------------------
static const char * THINK_OPEN_TAG  = "<think>";
static const char * THINK_CLOSE_TAG = "</think>";

// Localiza o id de um token especial (`<think>`/`</think>`); -1 se o vocab não o tiver.
static llama_token find_special_token(const llama_vocab * vocab, const char * tag) {
    llama_token toks[8];
    // parse_special=true para casar o token especial como unidade única.
    int n = llama_tokenize(vocab, tag, (int32_t) std::strlen(tag),
                           toks, 8, /*add_special*/ false, /*parse_special*/ true);
    if (n == 1) {
        return toks[0];
    }
    return -1;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_br_org_ceia_cemigpoc_llama_LlamaBridge_nativeLoad(
        JNIEnv * env,
        jobject /* thiz */,
        jstring j_model_path,
        jint j_n_ctx,
        jint j_n_threads
) {
    if (j_model_path == nullptr) {
        LOGE("nativeLoad: modelPath nulo");
        return 0;
    }

    const char * model_path = env->GetStringUTFChars(j_model_path, nullptr);
    LOGI("nativeLoad: Carregando modelo GGUF de %s (n_ctx=%d, threads=%d)",
         model_path, j_n_ctx, j_n_threads);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    env->ReleaseStringUTFChars(j_model_path, model_path);

    if (!model) {
        LOGE("nativeLoad: Falha ao instanciar llama_model");
        return 0;
    }

    const int32_t n_ctx = j_n_ctx > 0 ? j_n_ctx : 2048;
    const int32_t n_threads = j_n_threads > 0 ? j_n_threads : 6;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOGE("nativeLoad: Falha ao instanciar llama_context");
        llama_model_free(model);
        return 0;
    }

    auto * engineCtx = new LlamaEngineContext();
    engineCtx->model = model;
    engineCtx->ctx = ctx;
    engineCtx->n_ctx = n_ctx;
    engineCtx->n_batch = cparams.n_batch;

    LOGI("nativeLoad: Modelo carregado com sucesso. Contexto = %p", engineCtx);
    return reinterpret_cast<jlong>(engineCtx);
}

JNIEXPORT jstring JNICALL
Java_br_org_ceia_cemigpoc_llama_LlamaBridge_nativeApplyChatTemplate(
        JNIEnv * env,
        jobject /* thiz */,
        jlong handle,
        jobjectArray roles_array,
        jobjectArray contents_array,
        jboolean add_assistant,
        jboolean suppress_reasoning
) {
    auto * engineCtx = reinterpret_cast<LlamaEngineContext *>(handle);
    if (!engineCtx || !engineCtx->model) {
        LOGE("nativeApplyChatTemplate: Handle inválido");
        return env->NewStringUTF("");
    }

    const char * tmpl = llama_model_chat_template(engineCtx->model, nullptr);
    if (!tmpl) {
        LOGW("nativeApplyChatTemplate: Modelo não possui chat_template embutido");
    }

    jsize n_msgs = env->GetArrayLength(roles_array);
    std::vector<llama_chat_message> chat;
    chat.reserve(n_msgs);

    std::vector<std::string> role_strings;
    std::vector<std::string> content_strings;
    role_strings.reserve(n_msgs);
    content_strings.reserve(n_msgs);

    for (jsize i = 0; i < n_msgs; i++) {
        auto j_role = (jstring) env->GetObjectArrayElement(roles_array, i);
        auto j_content = (jstring) env->GetObjectArrayElement(contents_array, i);

        const char * c_role = env->GetStringUTFChars(j_role, nullptr);
        const char * c_content = env->GetStringUTFChars(j_content, nullptr);

        role_strings.emplace_back(c_role ? c_role : "");
        content_strings.emplace_back(c_content ? c_content : "");

        env->ReleaseStringUTFChars(j_role, c_role);
        env->ReleaseStringUTFChars(j_content, c_content);
        env->DeleteLocalRef(j_role);
        env->DeleteLocalRef(j_content);
    }

    for (size_t i = 0; i < role_strings.size(); i++) {
        chat.push_back({role_strings[i].c_str(), content_strings[i].c_str()});
    }

    std::vector<char> buf(4096);
    int32_t res = llama_chat_apply_template(
            tmpl,
            chat.data(),
            chat.size(),
            add_assistant,
            buf.data(),
            buf.size()
    );

    if (res > (int32_t) buf.size()) {
        buf.resize(res + 1);
        res = llama_chat_apply_template(
                tmpl,
                chat.data(),
                chat.size(),
                add_assistant,
                buf.data(),
                buf.size()
        );
    }

    if (res < 0) {
        LOGE("nativeApplyChatTemplate: Falha ao aplicar template (código %d)", res);
        return env->NewStringUTF("");
    }

    std::string prompt(buf.data(), (size_t) res);

    // Thinking-OFF (réplica de --reasoning-budget 0): anexa um bloco de raciocínio VAZIO
    // `<think></think>\n` ao fim do prompt. A C-API `llama_chat_apply_template` NÃO usa o
    // jinja completo do GGUF (casa um template embutido por heurística) e termina o prompt
    // em `assistant\n` SEM primar `<think>`; sem o bloco vazio o 2.6B raciocínia sozinho
    // (46-50 s/resposta no S24+). Provado na RTX 5070: prompt com `<think></think>` +
    // ban do token `<think>` na geração -> resposta limpa e correta. Anexar como TEXTO (e
    // não forçar tokens em decodes separados) é crucial no LFM2.5 híbrido (camadas
    // shortconv/recorrentes): o estado de convolução difere entre decode em lote e
    // incremental, e o decode incremental fazia o modelo ECOAR o prompt no S24+.
    // No 1.2B QAD (não-reasoning) o flag é false -> no-op.
    if (suppress_reasoning) {
        // Idempotente: só anexa se ainda não houver um `<think>` ao fim.
        const std::string open_tag = THINK_OPEN_TAG;
        bool already_primed = prompt.size() >= open_tag.size() &&
            prompt.compare(prompt.size() - open_tag.size(), open_tag.size(), open_tag) == 0;
        if (already_primed) {
            prompt += THINK_CLOSE_TAG;   // jinja completo já primou `<think>` -> só fecha
            prompt += "\n";
        } else {
            prompt += THINK_OPEN_TAG;
            prompt += THINK_CLOSE_TAG;
            prompt += "\n";
        }
        LOGI("nativeApplyChatTemplate: bloco <think></think> vazio anexado (thinking-OFF)");
    }

    return env->NewStringUTF(prompt.c_str());
}

JNIEXPORT jint JNICALL
Java_br_org_ceia_cemigpoc_llama_LlamaBridge_nativeGenerate(
        JNIEnv * env,
        jobject /* thiz */,
        jlong handle,
        jstring j_prompt,
        jint j_max_tokens,
        jfloat j_temperature,
        jint j_top_k,
        jfloat j_top_p,
        jfloat j_rep_penalty,
        jboolean suppress_reasoning,
        jobject callback
) {
    auto * engineCtx = reinterpret_cast<LlamaEngineContext *>(handle);
    if (!engineCtx || !engineCtx->model || !engineCtx->ctx) {
        LOGE("nativeGenerate: Handle inválido");
        return -1;
    }

    std::lock_guard<std::mutex> lock(engineCtx->mtx);

    const char * prompt_chars = env->GetStringUTFChars(j_prompt, nullptr);
    std::string prompt(prompt_chars ? prompt_chars : "");
    env->ReleaseStringUTFChars(j_prompt, prompt_chars);

    if (prompt.empty()) {
        return 0;
    }

    const llama_vocab * vocab = llama_model_get_vocab(engineCtx->model);

    // Configuração dos parâmetros de amostragem Liquid LFM2.5
    // Defaults certificados M1: temp 0.1, top_k 50, repeat_penalty 1.05
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (j_top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(j_top_k));
    }
    if (j_rep_penalty > 1.0f) {
        int32_t n_vocab = llama_vocab_n_tokens(vocab);
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(n_vocab, 64, j_rep_penalty, 0.0f, 0.0f));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    if (j_temperature > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(j_temperature));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.0f));
    }
    if (j_top_p > 0.0f && j_top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(j_top_p, 1));
    }
    // Thinking-OFF: bane a REABERTURA do token `<think>` (-INF). O bloco vazio já foi
    // primado no prompt (ver nativeApplyChatTemplate); o ban impede o modelo de reabrir o
    // raciocínio depois. No-op se o vocab não tiver o token (modelos sem reasoning).
    if (suppress_reasoning) {
        const llama_token think_open = find_special_token(vocab, THINK_OPEN_TAG);
        if (think_open >= 0) {
            llama_logit_bias bias{think_open, -INFINITY};
            llama_sampler_chain_add(smpl,
                    llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), 1, &bias));
            LOGI("nativeGenerate: thinking-OFF ativo (token <think>=%d banido)", think_open);
        }
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Tokenização do prompt com BOS
    int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.length(), nullptr, 0, false, true);
    if (n_prompt_tokens <= 0) {
        n_prompt_tokens = 1;
    }
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    int tok_res = llama_tokenize(vocab, prompt.c_str(), prompt.length(), prompt_tokens.data(), prompt_tokens.size(), false, true);
    if (tok_res < 0) {
        LOGE("nativeGenerate: Falha na tokenização do prompt");
        llama_sampler_free(smpl);
        return -1;
    }
    prompt_tokens.resize(tok_res);

    // Para modelos híbridos/recorrentes com shortconv (LFM2.5), limpa o estado de memória
    // garantindo integridade absoluta dos tensores de convolução entre turnos
    llama_memory_clear(llama_get_memory(engineCtx->ctx), false);
    engineCtx->cached_tokens.clear();

    LOGI("nativeGenerate: Prompt tokens total=%zu, iniciando decode", prompt_tokens.size());

    // Decodifica os tokens do prompt em fatias de n_batch com posições explícitas
    size_t to_decode = prompt_tokens.size();
    if (to_decode > 0) {
        const llama_token * new_tokens = prompt_tokens.data();
        for (size_t i = 0; i < to_decode; i += engineCtx->n_batch) {
            int32_t chunk_size = std::min((size_t) engineCtx->n_batch, to_decode - i);
            llama_batch batch = llama_batch_init(chunk_size, 0, 1);
            for (int32_t j = 0; j < chunk_size; j++) {
                batch.token[j] = new_tokens[i + j];
                batch.pos[j] = (llama_pos) (i + j);
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = (i + j == to_decode - 1);
            }
            batch.n_tokens = chunk_size;

            if (llama_decode(engineCtx->ctx, batch) != 0) {
                LOGE("nativeGenerate: Falha no decode do prompt");
                llama_batch_free(batch);
                llama_sampler_free(smpl);
                return -1;
            }
            llama_batch_free(batch);
        }
    }

    engineCtx->cached_tokens = prompt_tokens;

    // Localização do método de callback Java
    jclass callback_class = env->GetObjectClass(callback);
    jmethodID mid_on_token = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)Z");

    int generated_count = 0;
    const int max_tokens = j_max_tokens > 0 ? j_max_tokens : 700;

    // Loop autoregressivo de amostragem e geração
    while (generated_count < max_tokens) {
        int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(engineCtx->ctx), 0) + 1;
        if (n_ctx_used + 1 >= engineCtx->n_ctx) {
            LOGW("nativeGenerate: Limite de contexto atingido (%d/%d)", n_ctx_used, engineCtx->n_ctx);
            break;
        }

        llama_token new_token = llama_sampler_sample(smpl, engineCtx->ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        char piece_buf[256];
        int piece_len = llama_token_to_piece(vocab, new_token, piece_buf, sizeof(piece_buf), 0, true);
        if (piece_len > 0) {
            std::string piece_str(piece_buf, piece_len);
            jstring j_piece = env->NewStringUTF(piece_str.c_str());
            jboolean keep_going = env->CallBooleanMethod(callback, mid_on_token, j_piece);
            env->DeleteLocalRef(j_piece);

            if (!keep_going) {
                LOGI("nativeGenerate: Interrompido pelo consumidor (cancelado)");
                break;
            }
        }

        // Avança o modelo com o novo token gerado (posição explícita no KV cache)
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = new_token;
        batch.pos[0] = (llama_pos) (engineCtx->cached_tokens.size());
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        batch.n_tokens = 1;

        if (llama_decode(engineCtx->ctx, batch) != 0) {
            LOGE("nativeGenerate: Falha ao decodificar token gerado");
            llama_batch_free(batch);
            break;
        }
        llama_batch_free(batch);

        engineCtx->cached_tokens.push_back(new_token);
        generated_count++;
    }

    llama_sampler_free(smpl);
    return generated_count;
}

// -----------------------------------------------------------------------------
// Modo EMBEDDING (Retrieval v3): codifica UMA query com o EmbeddingGemma-300M GGUF.
// Reusa a MESMA libllama_engine já linkada (runtime mais simples do brief). Carrega um
// contexto separado com pooling=mean e embeddings=true; retorna o vetor L2-normalizado.
// -----------------------------------------------------------------------------
struct LlamaEmbedContext {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    std::mutex mtx;
    int32_t n_embd = 0;
};

JNIEXPORT jlong JNICALL
Java_br_org_ceia_cemigpoc_llama_LlamaBridge_nativeLoadEmbedder(
        JNIEnv * env,
        jobject /* thiz */,
        jstring j_model_path,
        jint j_n_ctx,
        jint j_n_threads
) {
    if (j_model_path == nullptr) {
        LOGE("nativeLoadEmbedder: modelPath nulo");
        return 0;
    }
    const char * model_path = env->GetStringUTFChars(j_model_path, nullptr);
    LOGI("nativeLoadEmbedder: carregando encoder GGUF de %s", model_path);
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    env->ReleaseStringUTFChars(j_model_path, model_path);
    if (!model) {
        LOGE("nativeLoadEmbedder: falha ao carregar modelo de embedding");
        return 0;
    }
    const int32_t n_ctx = j_n_ctx > 0 ? j_n_ctx : 2048;
    const int32_t n_threads = j_n_threads > 0 ? j_n_threads : 6;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;    // batch físico >= tokens da query (curta)
    cparams.n_ubatch = n_ctx;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;  // igual ao índice (mean pooling)
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOGE("nativeLoadEmbedder: falha ao criar contexto de embedding");
        llama_model_free(model);
        return 0;
    }
    auto * ec = new LlamaEmbedContext();
    ec->model = model;
    ec->ctx = ctx;
    ec->n_embd = llama_model_n_embd(model);
    LOGI("nativeLoadEmbedder: encoder pronto (n_embd=%d)", ec->n_embd);
    return reinterpret_cast<jlong>(ec);
}

JNIEXPORT jfloatArray JNICALL
Java_br_org_ceia_cemigpoc_llama_LlamaBridge_nativeEmbed(
        JNIEnv * env,
        jobject /* thiz */,
        jlong handle,
        jstring j_text
) {
    auto * ec = reinterpret_cast<LlamaEmbedContext *>(handle);
    if (!ec || !ec->ctx || !ec->model) {
        LOGE("nativeEmbed: handle inválido");
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ec->mtx);
    const char * text_chars = env->GetStringUTFChars(j_text, nullptr);
    std::string text(text_chars ? text_chars : "");
    env->ReleaseStringUTFChars(j_text, text_chars);

    const llama_vocab * vocab = llama_model_get_vocab(ec->model);
    // Tokeniza com BOS (add_special=true), sem token de fim de geração.
    int n_tokens = -llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, true, true);
    if (n_tokens <= 0) n_tokens = 1;
    std::vector<llama_token> tokens(n_tokens);
    int tok_res = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), true, true);
    if (tok_res < 0) {
        LOGE("nativeEmbed: falha na tokenização");
        return nullptr;
    }
    tokens.resize(tok_res);

    llama_memory_clear(llama_get_memory(ec->ctx), true);

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;  // pooling MEAN usa todos os tokens
    }
    batch.n_tokens = (int32_t) tokens.size();

    if (llama_decode(ec->ctx, batch) != 0) {
        LOGE("nativeEmbed: falha no decode");
        llama_batch_free(batch);
        return nullptr;
    }
    llama_batch_free(batch);

    // Recupera o embedding pooled da sequência 0.
    const float * emb = llama_get_embeddings_seq(ec->ctx, 0);
    if (!emb) {
        // fallback: embedding do último token
        emb = llama_get_embeddings(ec->ctx);
    }
    if (!emb) {
        LOGE("nativeEmbed: embeddings nulos");
        return nullptr;
    }
    const int n_embd = ec->n_embd;
    // Normaliza L2 (cosseno = produto interno).
    double norm = 0.0;
    for (int i = 0; i < n_embd; i++) norm += (double) emb[i] * emb[i];
    norm = norm > 0 ? std::sqrt(norm) : 1.0;
    std::vector<float> out(n_embd);
    for (int i = 0; i < n_embd; i++) out[i] = (float) (emb[i] / norm);

    jfloatArray result = env->NewFloatArray(n_embd);
    env->SetFloatArrayRegion(result, 0, n_embd, out.data());
    return result;
}

JNIEXPORT void JNICALL
Java_br_org_ceia_cemigpoc_llama_LlamaBridge_nativeFreeEmbedder(
        JNIEnv * /* env */,
        jobject /* thiz */,
        jlong handle
) {
    auto * ec = reinterpret_cast<LlamaEmbedContext *>(handle);
    if (!ec) return;
    LOGI("nativeFreeEmbedder: liberando encoder (%p)", ec);
    if (ec->ctx) llama_free(ec->ctx);
    if (ec->model) llama_model_free(ec->model);
    delete ec;
}

JNIEXPORT void JNICALL
Java_br_org_ceia_cemigpoc_llama_LlamaBridge_nativeClearKv(
        JNIEnv * /* env */,
        jobject /* thiz */,
        jlong handle
) {
    auto * engineCtx = reinterpret_cast<LlamaEngineContext *>(handle);
    if (!engineCtx || !engineCtx->ctx) return;

    std::lock_guard<std::mutex> lock(engineCtx->mtx);
    llama_memory_clear(llama_get_memory(engineCtx->ctx), false);
    engineCtx->cached_tokens.clear();
    LOGI("nativeClearKv: Cache KV e histórico de tokens limpos com sucesso");
}

JNIEXPORT void JNICALL
Java_br_org_ceia_cemigpoc_llama_LlamaBridge_nativeFree(
        JNIEnv * /* env */,
        jobject /* thiz */,
        jlong handle
) {
    auto * engineCtx = reinterpret_cast<LlamaEngineContext *>(handle);
    if (!engineCtx) return;

    LOGI("nativeFree: Liberando recursos nativos do Llama (%p)", engineCtx);
    if (engineCtx->ctx) {
        llama_free(engineCtx->ctx);
    }
    if (engineCtx->model) {
        llama_model_free(engineCtx->model);
    }
    delete engineCtx;
}

} // extern "C"
