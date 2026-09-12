#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include <cstring>
#include "whisper.h"

#define TAG "WhisperEngineJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jlong JNICALL
Java_br_org_ceia_cemigpoc_whisper_WhisperBridge_nativeInit(
        JNIEnv * env,
        jobject /* thiz */,
        jstring j_model_path
) {
    if (j_model_path == nullptr) {
        LOGE("nativeInit: Caminho do modelo Whisper nulo");
        return 0;
    }

    const char * model_path = env->GetStringUTFChars(j_model_path, nullptr);
    LOGI("nativeInit: Carregando modelo Whisper Base Q5_1 de %s", model_path);

    whisper_context_params cparams = whisper_context_default_params();
    whisper_context * ctx = whisper_init_from_file_with_params(model_path, cparams);
    env->ReleaseStringUTFChars(j_model_path, model_path);

    if (!ctx) {
        LOGE("nativeInit: Falha ao inicializar whisper_context");
        return 0;
    }

    LOGI("nativeInit: Whisper inicializado com sucesso (%p)", ctx);
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT jstring JNICALL
Java_br_org_ceia_cemigpoc_whisper_WhisperBridge_nativeTranscribe(
        JNIEnv * env,
        jobject /* thiz */,
        jlong handle,
        jfloatArray j_audio_data,
        jint j_num_threads,
        jstring j_language,
        jstring j_initial_prompt
) {
    auto * ctx = reinterpret_cast<whisper_context *>(handle);
    if (!ctx) {
        LOGE("nativeTranscribe: Handle Whisper inválido");
        return env->NewStringUTF("");
    }

    if (j_audio_data == nullptr) {
        LOGE("nativeTranscribe: Áudio nulo");
        return env->NewStringUTF("");
    }

    jsize n_samples = env->GetArrayLength(j_audio_data);
    if (n_samples <= 0) {
        return env->NewStringUTF("");
    }

    jfloat * audio_samples = env->GetFloatArrayElements(j_audio_data, nullptr);

    const char * lang_chars = j_language ? env->GetStringUTFChars(j_language, nullptr) : "pt";
    const char * prompt_chars = j_initial_prompt ? env->GetStringUTFChars(j_initial_prompt, nullptr) : nullptr;

    const int num_threads = j_num_threads > 0 ? j_num_threads : 4;

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.translate = false;
    params.language = lang_chars;
    params.n_threads = num_threads;
    params.offset_ms = 0;
    params.no_context = true;
    params.no_timestamps = true;
    params.single_segment = false;

    // Ajuste de qualidade p/ fala natural em campo (task poc-asr-fix):
    // - temperature fallback reativado: se o greedy decode falhar nos gates de
    //   entropia/logprob, o whisper.cpp re-decodifica com temperatura crescente,
    //   reduzindo travas e repetições em fala espontânea (pausas, hesitações).
    // - no_speech_thold um pouco mais alto descarta segmentos de puro ruído/silêncio,
    //   cortando alucinações quando o operário solta o PTT sem falar.
    params.temperature = 0.0f;
    params.temperature_inc = 0.2f;
    params.entropy_thold = 2.4f;
    params.logprob_thold = -1.0f;
    params.no_speech_thold = 0.6f;
    params.suppress_blank = true;
    params.suppress_nst = true; // suprime tokens non-speech (ruídos, música)
    if (prompt_chars && strlen(prompt_chars) > 0) {
        params.initial_prompt = prompt_chars;
    }

    LOGI("nativeTranscribe: Iniciando transcrição com %d amostras (%.2fs de áudio), lang=%s, threads=%d",
         n_samples, (float) n_samples / 16000.0f, lang_chars, num_threads);

    whisper_reset_timings(ctx);
    int ret = whisper_full(ctx, params, audio_samples, n_samples);

    env->ReleaseFloatArrayElements(j_audio_data, audio_samples, JNI_ABORT);
    if (j_language) env->ReleaseStringUTFChars(j_language, lang_chars);
    if (j_initial_prompt) env->ReleaseStringUTFChars(j_initial_prompt, prompt_chars);

    if (ret != 0) {
        LOGE("nativeTranscribe: Falha na inferência do Whisper (código %d)", ret);
        return env->NewStringUTF("");
    }

    std::string result;
    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; i++) {
        const char * segment_text = whisper_full_get_segment_text(ctx, i);
        if (segment_text) {
            result += segment_text;
        }
    }

    LOGI("nativeTranscribe: Concluído. %d segmentos. Texto: '%s'", n_segments, result.c_str());
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_br_org_ceia_cemigpoc_whisper_WhisperBridge_nativeFree(
        JNIEnv * /* env */,
        jobject /* thiz */,
        jlong handle
) {
    auto * ctx = reinterpret_cast<whisper_context *>(handle);
    if (!ctx) return;

    LOGI("nativeFree: Liberando whisper_context (%p)", ctx);
    whisper_free(ctx);
}

} // extern "C"
