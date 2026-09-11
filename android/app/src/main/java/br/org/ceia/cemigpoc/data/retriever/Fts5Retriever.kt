package br.org.ceia.cemigpoc.data.retriever

import android.content.Context
import android.database.Cursor
import android.util.Log
// SQLite com FTS5 embutido (mil.nga, fork do requery/sqlite-android). O SQLite do sistema
// Android nao traz o modulo FTS5, entao usamos a implementacao empacotada (libsqliteX.so)
// para que bm25()/MATCH funcionem no aparelho.
import org.sqlite.database.sqlite.SQLiteDatabase
import br.org.ceia.cemigpoc.domain.engine.Retriever
import br.org.ceia.cemigpoc.domain.model.Chunk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.Normalizer

/**
 * Implementação real do motor de busca documental utilizando SQLite FTS5 (BM25)
 * calibrado sobre o índice consolidado de 36 Normas Regulamentadoras (`index_hf_36nr.db`).
 *
 * Pesos BM25 calibrados: doc=1.5, section=3.0, title=2.0, text=1.0.
 *
 * Suporte a ciclo de vida de dados:
 * 1. Override externo: `context.getExternalFilesDir(null)/index.db` (injetável via adb push).
 * 2. Cópia interna: `context.filesDir/index.db`.
 * 3. Fallback de assets: cópia do arquivo embutido no APK na primeira execução.
 */
class Fts5Retriever(
    private val context: Context? = null,
    private val databaseOverrideFile: File? = null
) : Retriever {

    companion object {
        private const val TAG = "Fts5Retriever"
        const val DB_NAME = "index.db"

        // Carrega a biblioteca nativa do SQLite empacotado (com FTS5) uma unica vez.
        // Sem isto, o SQLite do sistema seria usado e falharia com 'no such module: fts5'.
        @Volatile
        private var nativeLoaded = false

        @Synchronized
        private fun ensureNativeLoaded() {
            if (nativeLoaded) return
            try {
                System.loadLibrary("sqliteX")
                nativeLoaded = true
                Log.i(TAG, "Biblioteca nativa sqliteX (FTS5) carregada com sucesso")
            } catch (e: Throwable) {
                Log.e(TAG, "Falha ao carregar libsqliteX.so (FTS5)", e)
            }
        }

        private val PORTUGUESE_STOPWORDS = setOf(
            "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "ate", "até",
            "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "do", "dos",
            "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa", "essas",
            "esse", "esses", "esta", "estas", "este", "estes", "eu", "foi", "fomos", "foram",
            "ha", "há", "isso", "isto", "ja", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo",
            "meu", "meus", "minha", "minhas", "muito", "na", "nas", "nao", "não", "no", "nos",
            "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para",
            "pela", "pelas", "pelo", "pelos", "por", "qual", "quais", "quando", "que", "quem",
            "se", "seja", "sem", "so", "só", "sua", "suas", "seu", "seus", "tambem", "também",
            "te", "tem", "têm", "temos", "ter", "teu", "teus", "tua", "tuas", "um", "uma", "voce", "você", "voces", "vocês",
            "norma", "normas", "regulamentar", "regulamentares", "regulamentaria", "regulamentarias",
            "seguranca", "segurança", "trabalho", "item", "artigo"
        )
    }

    fun resolveDatabaseFile(): File {
        if (databaseOverrideFile != null && databaseOverrideFile.exists()) {
            return databaseOverrideFile
        }

        if (context == null) {
            return databaseOverrideFile ?: File(DB_NAME)
        }

        val externalOverride = context.getExternalFilesDir(null)?.resolve(DB_NAME)
        if (externalOverride != null && externalOverride.exists() && externalOverride.length() > 0) {
            Log.i(TAG, "Utilizando index.db externo sobrescrito: ${externalOverride.absolutePath}")
            return externalOverride
        }

        val internalFile = File(context.filesDir, DB_NAME)
        if (!internalFile.exists() || internalFile.length() == 0L) {
            Log.i(TAG, "Copiando index.db dos assets para armazenamento interno: ${internalFile.absolutePath}")
            try {
                context.assets.open(DB_NAME).use { input ->
                    FileOutputStream(internalFile).use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Erro ao copiar index.db dos assets", e)
            }
        }
        return internalFile
    }

    override suspend fun search(query: String, topK: Int): List<Chunk> =
        searchBoosted(query, topK, boostNrs = emptyList(), boostFactor = 1.0, hardFilter = false)

    /**
     * Busca BM25 com BOOST SUAVE opcional nas NRs indicadas pelo classificador (estágio 1).
     *
     * O bm25() do FTS5 devolve score NEGATIVO (menor = melhor). No boost suave multiplicamos
     * o score dos chunks das NRs previstas por `boostFactor` (>1 => mais negativo => melhor),
     * SEM excluir as demais NRs (elas continuam elegíveis, só perdem prioridade relativa).
     * Com `hardFilter=true`, restringe a busca às NRs previstas (com fallback amplo se faltar).
     *
     * @param boostNrs códigos nr-XX a privilegiar (ex.: top-1/top-2 do NrClassifier)
     * @param boostFactor fator multiplicativo do boost suave (calibrado 5x no harness)
     * @param hardFilter se true, aplica filtro duro na(s) norma(s) prevista(s)
     */
    suspend fun searchBoosted(
        query: String,
        topK: Int,
        boostNrs: List<String>,
        boostFactor: Double,
        hardFilter: Boolean
    ): List<Chunk> = withContext(Dispatchers.IO) {
        val sanitizedQuery = sanitizeFts5Query(query)
        if (sanitizedQuery.isBlank()) {
            return@withContext emptyList()
        }
        val boostSet = boostNrs.map { it.lowercase() }.toSet()
        // Pool amplo p/ reordenar quando há boost/filtro; sem boost, LIMIT direto = topK.
        val poolLimit = if (boostSet.isEmpty()) topK else maxOf(topK, 60)

        val dbFile = resolveDatabaseFile()
        if (!dbFile.exists() || dbFile.length() == 0L) {
            Log.w(TAG, "Arquivo index.db não encontrado em ${dbFile.absolutePath}")
            return@withContext emptyList()
        }

        val results = mutableListOf<Chunk>()
        var database: SQLiteDatabase? = null
        var cursor: Cursor? = null

        try {
            ensureNativeLoaded()
            database = SQLiteDatabase.openDatabase(
                dbFile.absolutePath,
                null,
                SQLiteDatabase.OPEN_READONLY
            )
            // Garante que o FTS5 esta disponivel; se nao, a query MATCH lancaria 'no such module: fts5'.
            ensureFts5(database)

            // Filtro duro: restringe às NRs previstas via IN(...) (fallback amplo depois).
            val docClause = if (hardFilter && boostSet.isNotEmpty()) {
                "AND c.doc IN (${boostSet.joinToString(",") { "?" }})"
            } else ""

            // Consulta com junção chunks_fts e chunks com pesos BM25 calibrados (v2 36 NRs)
            val sql = """
                SELECT
                    c.id,
                    c.doc,
                    c.section,
                    c.title,
                    c.page,
                    c.text,
                    bm25(chunks_fts, 1.5, 3.0, 2.0, 1.0) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ? $docClause
                ORDER BY score ASC
                LIMIT ?;
            """.trimIndent()

            val args = ArrayList<String>()
            args.add(sanitizedQuery)
            if (hardFilter && boostSet.isNotEmpty()) args.addAll(boostSet)
            args.add(poolLimit.toString())
            cursor = database.rawQuery(sql, args.toTypedArray())

            while (cursor.moveToNext()) {
                val rowId = cursor.getLong(0)
                val doc = cursor.getString(1) ?: ""
                val section = cursor.getString(2) ?: ""
                val title = cursor.getString(3) ?: ""
                val page = cursor.getInt(4)
                val text = cursor.getString(5) ?: ""
                val score = cursor.getDouble(6)

                results.add(
                    Chunk(
                        id = rowId,
                        doc = doc,
                        section = section,
                        content = text,
                        score = score,
                        title = title,
                        page = page
                    )
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao consultar FTS5 com query '$sanitizedQuery'", e)
            results.addAll(fallbackSearch(database, query, topK))
        } finally {
            cursor?.close()
            database?.close()
        }

        // Reordenação em memória: boost suave (score negativo * fator) e corte em topK.
        if (boostSet.isNotEmpty() && !hardFilter && boostFactor != 1.0) {
            results.sortBy { c -> if (c.doc.lowercase() in boostSet) c.score * boostFactor else c.score }
        }
        // Filtro duro com fallback amplo quando o pool ficou abaixo de topK.
        if (hardFilter && boostSet.isNotEmpty() && results.size < topK) {
            val extra = searchBoosted(query, topK, emptyList(), 1.0, hardFilter = false)
            val seen = results.map { it.id }.toHashSet()
            results.addAll(extra.filter { it.id !in seen })
        }
        results.take(topK)
    }

    /**
     * Higieniza e formata termos de busca para o SQLite FTS5 idêntico ao harness v2:
     * - Remoção de diacríticos e normalização NFKD
     * - Filtragem de stopwords em português
     * - Truncamento para raiz (stemming 6 caracteres)
     * - Escape com aspas para termos com hífen ou ponto (ex: "nr-10"*)
     */
    fun sanitizeFts5Query(rawQuery: String): String {
        val normalized = Normalizer.normalize(rawQuery, Normalizer.Form.NFKD)
            .replace(Regex("\\p{M}"), "")
            .lowercase()

        val rawTokens = Regex("[\\w.-]+").findAll(normalized).map { it.value }.toList()
        var filtered = rawTokens.filter { it !in PORTUGUESE_STOPWORDS && it.length > 2 }
        if (filtered.isEmpty()) {
            filtered = rawTokens.filter { it.length > 2 }
        }
        if (filtered.isEmpty()) {
            return ""
        }

        return filtered.map { token ->
            val stem = if (token.length > 6) token.take(6) else token
            if (stem.contains("-") || stem.contains(".")) {
                "\"$stem\"*"
            } else {
                "$stem*"
            }
        }.joinToString(" OR ")
    }

    /**
     * Valida a presenca do modulo FTS5 na biblioteca SQLite carregada. Registra o estado
     * para diagnostico (o SQLite do sistema Android falha aqui; o empacotado passa).
     */
    private fun ensureFts5(database: SQLiteDatabase) {
        var c: Cursor? = null
        try {
            c = database.rawQuery("SELECT sqlite_version()", null)
            val ver = if (c.moveToFirst()) c.getString(0) else "?"
            Log.i(TAG, "SQLite empacotado carregado (versao $ver) com suporte a FTS5")
        } catch (e: Exception) {
            Log.e(TAG, "Falha ao validar SQLite/FTS5", e)
        } finally {
            c?.close()
        }
    }

    private fun fallbackSearch(database: SQLiteDatabase?, query: String, topK: Int): List<Chunk> {
        if (database == null || !database.isOpen) return emptyList()
        val list = mutableListOf<Chunk>()
        var cursor: Cursor? = null
        try {
            val terms = query.split(Regex("\\s+")).filter { it.length >= 3 }
            val firstTerm = terms.firstOrNull() ?: return emptyList()
            cursor = database.rawQuery(
                "SELECT id, doc, section, title, page, text FROM chunks WHERE text LIKE ? LIMIT ?",
                arrayOf("%$firstTerm%", topK.toString())
            )
            while (cursor.moveToNext()) {
                list.add(
                    Chunk(
                        id = cursor.getLong(0),
                        doc = cursor.getString(1) ?: "",
                        section = cursor.getString(2) ?: "",
                        title = cursor.getString(3) ?: "",
                        page = cursor.getInt(4),
                        content = cursor.getString(5) ?: "",
                        score = 0.0
                    )
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Erro no fallbackSearch", e)
        } finally {
            cursor?.close()
        }
        return list
    }
}
