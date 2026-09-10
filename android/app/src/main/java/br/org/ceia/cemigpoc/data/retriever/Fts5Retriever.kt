package br.org.ceia.cemigpoc.data.retriever

import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import br.org.ceia.cemigpoc.domain.engine.Retriever
import br.org.ceia.cemigpoc.domain.model.Chunk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/**
 * Implementação real do motor de busca documental utilizando SQLite FTS5 (BM25) nativo do Android.
 *
 * Suporte a ciclo de vida de dados:
 * 1. Primeiro boot: copia `index.db` embutido nos assets para `context.filesDir/index.db`.
 * 2. Override externo: se existir um `index.db` em `context.getExternalFilesDir(null)` (injetável via
 *    `adb push index.db /sdcard/Android/data/br.org.ceia.cemigpoc/files/index.db`), ele tem precedência
 *    para permitir iteração rápida com novos corpuses da trilha T1 sem reinstalação do app.
 */
class Fts5Retriever(
    private val context: Context,
    private val databaseOverrideFile: File? = null
) : Retriever {

    companion object {
        private const val TAG = "Fts5Retriever"
        private const val DB_NAME = "index.db"
    }

    /**
     * Garante que o arquivo do banco de dados FTS5 exista no disco local e retorna o caminho ativo.
     */
    fun resolveDatabaseFile(): File {
        if (databaseOverrideFile != null && databaseOverrideFile.exists()) {
            return databaseOverrideFile
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

    override suspend fun search(query: String, topK: Int): List<Chunk> = withContext(Dispatchers.IO) {
        val sanitizedQuery = sanitizeFts5Query(query)
        if (sanitizedQuery.isBlank()) {
            return@withContext emptyList()
        }

        val dbFile = resolveDatabaseFile()
        if (!dbFile.exists() || dbFile.length() == 0L) {
            Log.w(TAG, "Arquivo index.db não encontrado em ${dbFile.absolutePath}")
            return@withContext emptyList()
        }

        val results = mutableListOf<Chunk>()
        var database: SQLiteDatabase? = null
        var cursor: Cursor? = null

        try {
            database = SQLiteDatabase.openDatabase(
                dbFile.absolutePath,
                null,
                SQLiteDatabase.OPEN_READONLY
            )

            // Consulta FTS5 com cálculo de relevância BM25 integrado
            val sql = """
                SELECT rowid, doc, section, content, bm25(chunks) as score 
                FROM chunks 
                WHERE chunks MATCH ? 
                ORDER BY score ASC 
                LIMIT ?
            """.trimIndent()

            cursor = database.rawQuery(sql, arrayOf(sanitizedQuery, topK.toString()))

            while (cursor.moveToNext()) {
                val rowId = cursor.getLong(0)
                val doc = cursor.getString(1) ?: ""
                val section = cursor.getString(2) ?: ""
                val content = cursor.getString(3) ?: ""
                val score = cursor.getDouble(4)

                results.add(
                    Chunk(
                        id = rowId,
                        doc = doc,
                        section = section,
                        content = content,
                        score = score
                    )
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Erro ao consultar FTS5 com query '$sanitizedQuery'", e)
            // Tenta busca de fallback por termo simples caso a sintaxe FTS5 tenha falhado
            results.addAll(fallbackSearch(database, query, topK))
        } finally {
            cursor?.close()
            database?.close()
        }

        results
    }

    /**
     * Sanitiza a consulta para a sintaxe FTS5, tratando caracteres reservados e criando termos OR/AND.
     */
    private fun sanitizeFts5Query(rawQuery: String): String {
        val cleaned = rawQuery.replace(Regex("[^a-zA-Z0-9À-ÿ\\s]"), " ")
        val tokens = cleaned.split(Regex("\\s+"))
            .filter { it.length >= 2 }
            .map { "$it*" } // busca por prefixo no FTS5

        return tokens.joinToString(" OR ")
    }

    private fun fallbackSearch(database: SQLiteDatabase?, query: String, topK: Int): List<Chunk> {
        if (database == null || !database.isOpen) return emptyList()
        val list = mutableListOf<Chunk>()
        var cursor: Cursor? = null
        try {
            val terms = query.split(Regex("\\s+")).filter { it.length >= 3 }
            val firstTerm = terms.firstOrNull() ?: return emptyList()
            cursor = database.rawQuery(
                "SELECT rowid, doc, section, content FROM chunks WHERE content LIKE ? LIMIT ?",
                arrayOf("%$firstTerm%", topK.toString())
            )
            while (cursor.moveToNext()) {
                list.add(
                    Chunk(
                        id = cursor.getLong(0),
                        doc = cursor.getString(1) ?: "",
                        section = cursor.getString(2) ?: "",
                        content = cursor.getString(3) ?: "",
                        score = 0.0
                    )
                )
            }
        } catch (_: Exception) {
        } finally {
            cursor?.close()
        }
        return list
    }
}
