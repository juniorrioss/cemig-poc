package br.org.ceia.cemigpoc.data.retriever

import android.content.Context
import android.util.Log
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.llama.LlamaEmbedder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.DataInputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Busca vetorial densa on-device (Retrieval v3). Carrega dois índices de vetores
 * EmbeddingGemma-300M (mesmo espaço 768-d do encoder GGUF) e faz KNN exato por produto
 * interno (cosseno, vetores L2-normalizados) em memória — brute-force sobre 2202 vetores
 * é trivial (~poucos ms) e dispensa dependência nativa de vector store.
 *
 * Índices (formato "DVEC1", ver retrieval3/export_dense_bin.py):
 *   - dense_text.bin     : texto normativo + expansão coloquial (query↔passagem)
 *   - dense_exponly.bin  : só a expansão coloquial              (query↔query)
 *
 * A metadata (doc/section/title/text) NÃO está no .bin: é hidratada pelo Fts5Retriever
 * (join por id) para montar os Chunk finais na fusão RRF.
 *
 * Precedência de arquivos idêntica ao ModelFileManager: externalFilesDir -> filesDir ->
 * assets.
 */
class DenseRetriever(
    private val context: Context,
    private val embedder: LlamaEmbedder
) {

    companion object {
        private const val TAG = "DenseRetriever"
        const val TEXT_BIN = "dense_text.bin"
        const val EXPONLY_BIN = "dense_exponly.bin"
        private const val MAGIC = "DVEC1"
    }

    /** Um índice denso carregado: ids alinhados a uma matriz linear (count × dim). */
    private class DenseIndex(val ids: IntArray, val dim: Int, val data: FloatArray) {
        val count: Int get() = ids.size

        /** KNN por produto interno; retorna pares (id, score) ordenados desc. */
        fun search(query: FloatArray, k: Int): List<Pair<Int, Float>> {
            val scores = FloatArray(count)
            var base = 0
            for (i in 0 until count) {
                var s = 0f
                var d = 0
                while (d < dim) {
                    s += data[base + d] * query[d]
                    d++
                }
                scores[i] = s
                base += dim
            }
            // top-k parcial por seleção simples (k pequeno).
            val idx = (0 until count).sortedByDescending { scores[it] }.take(k)
            return idx.map { ids[it] to scores[it] }
        }
    }

    @Volatile
    private var textIndex: DenseIndex? = null

    @Volatile
    private var expIndex: DenseIndex? = null

    val isReady: Boolean
        get() = textIndex != null && expIndex != null

    /** Carrega os dois índices .bin (idempotente). */
    suspend fun load(): Boolean = withContext(Dispatchers.IO) {
        if (isReady) return@withContext true
        try {
            textIndex = loadBin(resolveFile(TEXT_BIN))
            expIndex = loadBin(resolveFile(EXPONLY_BIN))
            Log.i(TAG, "Índices densos carregados: text=${textIndex?.count} exp=${expIndex?.count} dim=${textIndex?.dim}")
            isReady
        } catch (e: Exception) {
            Log.e(TAG, "Falha ao carregar índices densos", e)
            false
        }
    }

    /**
     * Codifica a query no encoder GGUF e busca os top-k de cada índice denso.
     * Retorna dois rankings de ids (texto+exp, exp-only) para a fusão RRF.
     */
    suspend fun search(rawQuestion: String, k: Int): DenseResult? = withContext(Dispatchers.IO) {
        val ti = textIndex; val ei = expIndex
        if (ti == null || ei == null) {
            Log.w(TAG, "search chamado sem índices carregados")
            return@withContext null
        }
        val qv = embedder.embedQuery(rawQuestion)
        if (qv == null || qv.size != ti.dim) {
            Log.w(TAG, "encode da query falhou (dim=${qv?.size} esperado=${ti.dim})")
            return@withContext null
        }
        DenseResult(
            textRanking = ti.search(qv, k),
            expRanking = ei.search(qv, k)
        )
    }

    data class DenseResult(
        val textRanking: List<Pair<Int, Float>>,
        val expRanking: List<Pair<Int, Float>>
    )

    private fun loadBin(file: File): DenseIndex {
        DataInputStream(file.inputStream().buffered()).use { input ->
            val magic = ByteArray(5)
            input.readFully(magic)
            check(String(magic) == MAGIC) { "Formato inválido em ${file.name}" }
            // Cabeçalho little-endian: dim, count.
            val header = ByteArray(8)
            input.readFully(header)
            val hb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
            val dim = hb.int
            val count = hb.int
            val ids = IntArray(count)
            val data = FloatArray(count.toLong().toInt() * dim)
            val recBytes = 4 + dim * 4
            val rec = ByteArray(recBytes)
            for (i in 0 until count) {
                input.readFully(rec)
                val rb = ByteBuffer.wrap(rec).order(ByteOrder.LITTLE_ENDIAN)
                ids[i] = rb.int
                val off = i * dim
                for (d in 0 until dim) data[off + d] = rb.float
            }
            return DenseIndex(ids, dim, data)
        }
    }

    private fun resolveFile(name: String): File {
        val external = context.getExternalFilesDir(null)?.resolve(name)
        if (external != null && external.exists() && external.length() > 0) {
            Log.i(TAG, "Usando índice denso externo: ${external.absolutePath}")
            return external
        }
        val internal = File(context.filesDir, name)
        if (!internal.exists() || internal.length() == 0L) {
            Log.i(TAG, "Copiando $name dos assets para ${internal.absolutePath}")
            context.assets.open(name).use { i -> FileOutputStream(internal).use { o -> i.copyTo(o) } }
        }
        return internal
    }
}
