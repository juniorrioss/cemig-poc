package br.org.ceia.cemigpoc.data.model

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/**
 * Gerenciador de ciclo de vida dos arquivos de pesos dos modelos e índices locais.
 *
 * Precedência de resolução:
 * 1. Override externo: `context.getExternalFilesDir(null)/<arquivo>` (injetável via adb push sem reinstalação)
 * 2. Armazenamento interno: `context.filesDir/<arquivo>`
 * 3. Assets embutidos: cópia automática do APK na primeira execução
 */
class ModelFileManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelFileManager"
        // Consolidação POC: 1.2B QAD-Q4_0 (Instruct, NÃO-reasoning) mantido como sintetizador.
        // O experimento judge151 aprovou o 2.6B thinking-OFF por NÚMERO na GPU (gate 9.9%
        // ≈ 3x do 1.2B, 178 tok), MAS o engine nativo do app (commit 434ddbb, C-API
        // llama_chat_apply_template com add_assistant=true) FORÇA thinking ON no 2.6B: no
        // aparelho gerou 700 tok de <think> e 44-54s de latência (inviável p/ voz). Não há
        // --reasoning-budget no caminho JNI e /no_think não suprime. Fallback do brief -> 1.2B.
        // Ver android/README_CONSOLIDACAO.md (seção "2.6B: aprovado na GPU, barrado no engine").
        const val LLM_MODEL_NAME = "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf"
        const val ASR_MODEL_NAME = "ggml-base-q5_1.bin"
        const val FTS5_DB_NAME = "index.db"
    }

    suspend fun getLlmModelFile(onProgress: ((Float) -> Unit)? = null): File = withContext(Dispatchers.IO) {
        resolveOrCopy(LLM_MODEL_NAME, onProgress)
    }

    suspend fun getAsrModelFile(onProgress: ((Float) -> Unit)? = null): File = withContext(Dispatchers.IO) {
        resolveOrCopy(ASR_MODEL_NAME, onProgress)
    }

    suspend fun getFts5DatabaseFile(): File = withContext(Dispatchers.IO) {
        resolveOrCopy(FTS5_DB_NAME, null)
    }

    private fun resolveOrCopy(fileName: String, onProgress: ((Float) -> Unit)?): File {
        // 1. Verifica override externo (sdcard)
        val externalFile = context.getExternalFilesDir(null)?.resolve(fileName)
        if (externalFile != null && externalFile.exists() && externalFile.length() > 0) {
            Log.i(TAG, "Utilizando arquivo sobrescrito externo: ${externalFile.absolutePath} (${externalFile.length()} bytes)")
            onProgress?.invoke(1.0f)
            return externalFile
        }

        // 2. Verifica se já foi copiado para o armazenamento interno
        val internalFile = File(context.filesDir, fileName)
        if (internalFile.exists() && internalFile.length() > 0) {
            Log.i(TAG, "Utilizando arquivo local interno: ${internalFile.absolutePath} (${internalFile.length()} bytes)")
            onProgress?.invoke(1.0f)
            return internalFile
        }

        // 3. Copia dos assets embutidos do APK
        Log.i(TAG, "Copiando $fileName dos assets para ${internalFile.absolutePath}...")
        try {
            val assetFd = try {
                context.assets.openFd(fileName)
            } catch (_: Exception) {
                null
            }
            val totalBytes = assetFd?.length ?: -1L
            assetFd?.close()

            context.assets.open(fileName).use { input ->
                FileOutputStream(internalFile).use { output ->
                    val buffer = ByteArray(1024 * 1024) // 1MB buffer
                    var bytesCopied = 0L
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        bytesCopied += read
                        if (totalBytes > 0) {
                            val progress = bytesCopied.toFloat() / totalBytes.toFloat()
                            onProgress?.invoke(progress)
                        }
                    }
                    output.flush()
                }
            }
            Log.i(TAG, "Cópia concluída: ${internalFile.absolutePath} (${internalFile.length()} bytes)")
            onProgress?.invoke(1.0f)
            return internalFile
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao copiar asset $fileName", e)
            throw e
        }
    }
}
