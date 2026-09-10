package br.org.ceia.cemigpoc.domain.engine

import br.org.ceia.cemigpoc.domain.model.Chunk

/**
 * Interface do motor de busca documental (BM25 / SQLite-FTS5).
 */
interface Retriever {
    /**
     * Realiza a busca léxica por palavras-chave nas normas técnicas indexadas.
     *
     * @param query Termos de consulta fornecidos pelo modelo ou usuário
     * @param topK Quantidade máxima de trechos a retornar (padrão 3)
     * @return Lista dos chunks mais relevantes ordenados por score BM25
     */
    suspend fun search(query: String, topK: Int = 3): List<Chunk>
}
