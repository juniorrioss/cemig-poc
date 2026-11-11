package br.org.ceia.cemigpoc.data.model

import android.content.Context
import android.os.StatFs
import android.util.Log
import br.org.ceia.cemigpoc.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

/**
 * Arquivo de modelo que NÃO cabe no APK e é baixado no primeiro uso (task lfm3-f10-dist).
 *
 * @param relPath caminho relativo ao diretório de modelos (ex.: "nemotron-…/encoder.int8.onnx").
 * @param url     download público (HF resolve/main), sem token.
 * @param size    tamanho exato em bytes (verificado).
 * @param sha256  hash esperado (verificado antes de promover o arquivo).
 */
data class RequiredModel(
    val label: String,
    val relPath: String,
    val url: String,
    val size: Long,
    val sha256: String
)

/**
 * Setup de primeiro uso: decide o que falta e baixa com retomada (HTTP Range sobre `.part`),
 * verificação de tamanho + SHA-256 e promoção atômica (rename) para `filesDir`.
 *
 * Origem dos pesos: repositórios públicos do HF (o GGUF do modelo vem do nosso repo; encoder
 * denso e ASR vêm dos repos upstream, fixados por SHA-256 — nada é carregado sem bater o hash).
 */
class ModelDownloader(private val context: Context) {

    companion object {
        private const val TAG = "ModelDownloader"
        private const val HF = "https://huggingface.co"
        private const val OUR_REPO = "$HF/juniorrios/automobil-ultra-semnexo/resolve/main"
        private const val NEMO_REPO =
            "$HF/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/resolve/main"
        private const val NEMO = ModelFileManager.NEMOTRON_DIR_NAME

        /** GGUF do LLM conforme o build (nome no aparelho -> arquivo no nosso repo HF). */
        private fun llmModel(): RequiredModel? = when (ModelFileManager.LLM_MODEL_NAME) {
            "lfm3-1.7b-v3s400-Q4_0.gguf" -> RequiredModel(
                "Modelo de linguagem LFM3 1.7B (v3s400 Q4_0)", ModelFileManager.LLM_MODEL_NAME,
                "$OUR_REPO/v1.1-rlvr/cemig-lfm3-1.7b-nr-v3s400-Q4_0.gguf", 1_049_614_880L,
                "689735f30a195be407fc56c5c4b7b27e3043c37b5ff2e58480477cda5992d6d8"
            )
            "lfm3-1.7b-f4-r32-ep2-Q4_0.gguf" -> RequiredModel(
                "Modelo de linguagem LFM3 1.7B (f4 Q4_0)", ModelFileManager.LLM_MODEL_NAME,
                "$OUR_REPO/cemig-lfm3-1.7b-nr-Q4_0.gguf", 1_049_614_880L,
                "94847d5bbcaae51a3cf63c6aed0063922e008521267d4ccdd364bf8533e85050"
            )
            "lfm3-1.7b-f4-r32-ep2-Q4_K_M.gguf" -> RequiredModel(
                "Modelo de linguagem LFM3 1.7B (f4 Q4_K_M)", ModelFileManager.LLM_MODEL_NAME,
                "$OUR_REPO/cemig-lfm3-1.7b-nr-Q4_K_M.gguf", 1_099_627_040L,
                "487c5a5530a599acc1827fffbf1b1fa17a5cdbe2f31082632fa3c572bdfcc8f3"
            )
            else -> null // builds legados LFM2.5: pesos só por adb push (não distribuídos)
        }

        /** Manifesto completo de pesos externos do build atual. */
        fun manifest(): List<RequiredModel> = buildList {
            llmModel()?.let { add(it) }
            add(
                RequiredModel(
                    "Encoder de busca EmbeddingGemma-300M", ModelFileManager.EMBED_MODEL_NAME,
                    "$HF/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF/resolve/main/embeddinggemma-300M-qat-Q4_0.gguf",
                    277_852_192L, "50d28e22432a148f6f8a86eab3700f92add5d1f54baf7790675a2a4dadbccf26"
                )
            )
            if (BuildConfig.USE_NEMOTRON_ASR) {
                add(RequiredModel("Voz Nemotron 3.5 (encoder)", "$NEMO/encoder.int8.onnx",
                    "$NEMO_REPO/encoder.int8.onnx", 657_601_403L,
                    "012e9321373af99021415e0b0eb3ec827b4be3153be6f30d9b448fe65e896e68"))
                add(RequiredModel("Voz Nemotron 3.5 (decoder)", "$NEMO/decoder.int8.onnx",
                    "$NEMO_REPO/decoder.int8.onnx", 14_978_075L,
                    "19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0"))
                add(RequiredModel("Voz Nemotron 3.5 (joiner)", "$NEMO/joiner.int8.onnx",
                    "$NEMO_REPO/joiner.int8.onnx", 9_504_438L,
                    "4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce"))
                add(RequiredModel("Voz Nemotron 3.5 (tokens)", "$NEMO/tokens.txt",
                    "$NEMO_REPO/tokens.txt", 131_440L,
                    "729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a"))
            } else {
                // Whisper só é obrigatório quando é o motor principal (build -Pcemig.asrEngine=whisper).
                add(RequiredModel("Voz Whisper Base Q5_1", ModelFileManager.ASR_MODEL_NAME,
                    "$HF/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin", 59_707_625L,
                    "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898"))
            }
        }
    }

    /** Diretório interno onde os pesos baixados ficam (apagado ao desinstalar). */
    val modelsDir: File get() = context.filesDir

    /** Um arquivo está presente se existe com o tamanho exato (interno ou override externo do adb). */
    fun isPresent(m: RequiredModel): Boolean {
        val ext = context.getExternalFilesDir(null)?.resolve(m.relPath)
        if (ext != null && ext.length() == m.size) return true
        return File(modelsDir, m.relPath).length() == m.size
    }

    fun missing(): List<RequiredModel> = manifest().filterNot { isPresent(it) }

    /** Bytes ainda a baixar (desconta os `.part` já baixados). */
    fun remainingBytes(list: List<RequiredModel>): Long =
        list.sumOf { m -> m.size - partFile(m).length().coerceAtMost(m.size) }

    fun freeBytes(): Long = StatFs(modelsDir.absolutePath).availableBytes

    private fun partFile(m: RequiredModel) = File(modelsDir, m.relPath + ".part")

    /**
     * Baixa todos os arquivos faltantes em sequência. [onProgress] recebe (índice, arquivo,
     * bytes baixados do total, total). Lança [IOException] com mensagem para o operário.
     */
    suspend fun downloadAll(
        list: List<RequiredModel>,
        onProgress: (index: Int, model: RequiredModel, doneBytes: Long, totalBytes: Long) -> Unit
    ) = withContext(Dispatchers.IO) {
        val total = list.sumOf { it.size }
        val need = remainingBytes(list)
        val free = freeBytes()
        if (free < need + 200L * 1024 * 1024) {
            throw IOException(
                "Espaço insuficiente: faltam ${fmtGb(need)} para os modelos e há só ${fmtGb(free)} livres. " +
                    "Libere espaço e toque em Tentar novamente."
            )
        }
        var doneBefore = 0L
        list.forEachIndexed { i, m ->
            downloadOne(m) { got -> onProgress(i, m, doneBefore + got, total) }
            doneBefore += m.size
        }
    }

    private suspend fun downloadOne(m: RequiredModel, onBytes: (Long) -> Unit) {
        val dest = File(modelsDir, m.relPath)
        dest.parentFile?.mkdirs()
        val part = partFile(m)
        if (part.length() > m.size) part.delete()
        var attempt = 0
        while (true) {
            coroutineContext.ensureActive()
            try {
                if (part.length() < m.size) fetch(m, part, onBytes)
                break
            } catch (e: IOException) {
                attempt++
                Log.w(TAG, "Falha no download de ${m.relPath} (tentativa $attempt): ${e.message}")
                if (attempt >= 5) throw IOException(
                    "Falha ao baixar ${m.label}: ${e.message}. Verifique a internet e toque em " +
                        "Tentar novamente (o download continua de onde parou).", e
                )
                kotlinx.coroutines.delay(2000L * attempt)
            }
        }
        onBytes(m.size)
        val hash = sha256(part)
        if (!hash.equals(m.sha256, ignoreCase = true)) {
            part.delete()
            throw IOException("Arquivo ${m.label} chegou corrompido (SHA-256 não confere). Toque em Tentar novamente.")
        }
        if (!part.renameTo(dest)) throw IOException("Não foi possível gravar ${dest.absolutePath}")
        Log.i(TAG, "OK ${m.relPath} (${m.size} bytes, sha256 verificado)")
    }

    /** Baixa (retomando via Range) para o `.part`. */
    private suspend fun fetch(m: RequiredModel, part: File, onBytes: (Long) -> Unit) {
        var have = part.length()
        val conn = (URL(m.url).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = true
            connectTimeout = 20_000
            readTimeout = 60_000
            setRequestProperty("User-Agent", "cemig-assistente/${BuildConfig.VERSION_NAME}")
            if (have > 0) setRequestProperty("Range", "bytes=$have-")
        }
        try {
            val code = conn.responseCode
            val append = when (code) {
                HttpURLConnection.HTTP_PARTIAL -> true
                HttpURLConnection.HTTP_OK -> false // servidor ignorou o Range: recomeça
                416 -> { part.delete(); throw IOException("faixa inválida (HTTP 416), recomeçando") }
                else -> throw IOException("HTTP $code")
            }
            if (!append) have = 0
            conn.inputStream.use { input ->
                FileOutputStream(part, append).use { out ->
                    val buf = ByteArray(256 * 1024)
                    var lastReport = 0L
                    while (true) {
                        coroutineContext.ensureActive()
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        have += n
                        if (have - lastReport > 2L * 1024 * 1024) {
                            lastReport = have
                            onBytes(have)
                        }
                    }
                }
            }
            if (have != m.size) throw IOException("conexão interrompida ($have de ${m.size} bytes)")
        } finally {
            conn.disconnect()
        }
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        FileInputStream(f).use { input ->
            val buf = ByteArray(1024 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}

fun fmtGb(bytes: Long): String =
    if (bytes >= 1_000_000_000L) "%.2f GB".format(bytes / 1e9) else "%.0f MB".format(bytes / 1e6)
