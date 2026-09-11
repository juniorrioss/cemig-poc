package br.org.ceia.cemigpoc.data.model

import android.content.Context
import android.util.Log
import br.org.ceia.cemigpoc.BuildConfig
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
        //
        // task poc-engine-upgrade: o engine agora suporta thinking-OFF no 2.6B (supressão do
        // <think> via template + logit-bias). O modelo é selecionável por build
        // (-Pcemig.synthModel=2.6b); o DEFAULT continua o 1.2B QAD até o treino DPO fechar.
        // O nome vem de BuildConfig.LLM_MODEL_NAME (definido em app/build.gradle.kts).
        val LLM_MODEL_NAME: String = BuildConfig.LLM_MODEL_NAME
        const val ASR_MODEL_NAME = "ggml-base-q5_1.bin"
        const val FTS5_DB_NAME = "index.db"
        // Retrieval v3: encoder de embeddings on-device (EmbeddingGemma-300M QAT-Q4_0) e os
        // dois índices densos binários (formato DVEC1). O index.db agora é o EXPANDIDO
        // (5 campos FTS5, com coluna expansion) — substituído nos assets.
        const val EMBED_MODEL_NAME = "embeddinggemma-300M-qat-Q4_0.gguf"
        const val DENSE_TEXT_BIN = "dense_text.bin"
        const val DENSE_EXPONLY_BIN = "dense_exponly.bin"

        // Versão dos assets copiados: ao subir (ex.: index.db 4->5 campos do Retrieval v3),
        // invalida a cópia interna obsoleta em filesDir (que NÃO é sobrescrita por install).
        // Sem isto, um upgrade do app continuaria usando o índice antigo já copiado.
        private const val ASSET_VERSION = 3  // v3: índice FTS5 expandido + índices densos
        // Arquivos versionados: recopiar do APK se a versão do asset mudou.
        private val VERSIONED_ASSETS = setOf(FTS5_DB_NAME, DENSE_TEXT_BIN, DENSE_EXPONLY_BIN)
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

    suspend fun getEmbedModelFile(onProgress: ((Float) -> Unit)? = null): File = withContext(Dispatchers.IO) {
        resolveOrCopy(EMBED_MODEL_NAME, onProgress)
    }

    /**
     * Invalida a cópia interna de assets versionados quando a ASSET_VERSION muda (ex.: novo
     * índice v3). O marcador fica em filesDir/asset_version.txt. Idempotente.
     */
    private fun invalidateStaleAssetsIfNeeded() {
        val marker = File(context.filesDir, "asset_version.txt")
        val current = if (marker.exists()) marker.readText().trim().toIntOrNull() ?: -1 else -1
        if (current == ASSET_VERSION) return
        for (name in VERSIONED_ASSETS) {
            val f = File(context.filesDir, name)
            if (f.exists()) {
                Log.i(TAG, "Asset versionado obsoleto (v$current -> v$ASSET_VERSION): removendo ${f.name}")
                f.delete()
            }
        }
        marker.writeText(ASSET_VERSION.toString())
    }

    private fun resolveOrCopy(fileName: String, onProgress: ((Float) -> Unit)?): File {
        // 0. Invalida cópias internas obsoletas de assets versionados (upgrade de índice).
        if (fileName in VERSIONED_ASSETS) invalidateStaleAssetsIfNeeded()

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
