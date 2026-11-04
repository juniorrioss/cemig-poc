package br.org.ceia.cemigpoc.data.retriever

import android.util.Log
import br.org.ceia.cemigpoc.domain.model.Chunk
import org.sqlite.database.sqlite.SQLiteDatabase
import java.io.File

/**
 * Os 9 MANUAIS como "aba separada" do RRF (F6, ponto de operação thr0.7_f2+man1.75+margin).
 * Porte de `bench/lfm3_f5/manuals.py::ManualIndex`.
 *
 * Nunca entram no índice das normas (lição v2/docai: despejar manuais no BM25 dilui R@5 −23 p.p.).
 * Base FTS5 PRÓPRIA (`manuals.db`, tabela mchunks + mchunks_fts, mesmo tokenizer unicode61
 * remove_diacritics 2, mesma query sanitizada do app, pesos 1.5/3/2/1) + denso PRÓPRIO
 * (`manuals_dense.bin`, DVEC1, EmbeddingGemma com prompt de documento). Ids = 1_000_000+idx.
 *
 * O carregamento do .db usa o SQLite empacotado com FTS5 (sqliteX), já carregado pelo
 * Fts5Retriever.
 */
class ManualIndex(
    private val dbFile: File,
    val dense: DenseMatrix,
    private val sanitize: (String) -> String
) {
    companion object {
        private const val TAG = "ManualIndex"
        const val DB_NAME = "manuals.db"
        const val DENSE_BIN = "manuals_dense.bin"
        const val ID_OFFSET = 1_000_000L
        const val POOL = 60
    }

    private val meta: Map<Long, Chunk> by lazy { loadMeta() }

    private fun open(): SQLiteDatabase =
        SQLiteDatabase.openDatabase(dbFile.absolutePath, null, SQLiteDatabase.OPEN_READONLY)

    private fun loadMeta(): Map<Long, Chunk> {
        val out = LinkedHashMap<Long, Chunk>()
        open().use { db ->
            db.rawQuery("SELECT id, doc, section, title, page, text FROM mchunks", null).use { c ->
                while (c.moveToNext()) {
                    val id = c.getLong(0)
                    out[id] = Chunk(
                        id = id, doc = c.getString(1) ?: "", section = c.getString(2) ?: "",
                        title = c.getString(3) ?: "", page = c.getInt(4), content = c.getString(5) ?: ""
                    )
                }
            }
        }
        Log.i(TAG, "manuals.db: ${out.size} chunks")
        return out
    }

    fun chunk(id: Long): Chunk? = meta[id]

    /** BM25 (== ManualIndex.bm25 do espelho): ranking de ids. */
    fun bm25(rawQuestion: String, pool: Int = POOL): List<Long> {
        val fq = sanitize(rawQuestion)
        if (fq.isBlank()) return emptyList()
        val out = ArrayList<Long>()
        try {
            open().use { db ->
                db.rawQuery(
                    "SELECT m.id, bm25(mchunks_fts, 1.5, 3.0, 2.0, 1.0) AS score FROM mchunks_fts " +
                        "JOIN mchunks m ON m.id = mchunks_fts.rowid WHERE mchunks_fts MATCH ? " +
                        "ORDER BY score LIMIT ?",
                    arrayOf(fq, pool.toString())
                ).use { c -> while (c.moveToNext()) out.add(c.getLong(0)) }
            }
        } catch (e: Exception) {
            Log.w(TAG, "bm25 manual falhou ('$fq')", e)
            return emptyList()
        }
        return out
    }

    /** KNN denso (== ManualIndex.dense): ranking de ids. */
    fun denseRanking(qv: FloatArray, pool: Int = POOL): List<Long> =
        dense.topK(qv, pool).map { it.first.toLong() }

    /** Melhor similaridade densa de manual (gate por margem). */
    fun maxSim(qv: FloatArray): Float = dense.maxSim(qv)
}
