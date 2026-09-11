package br.org.ceia.cemigpoc.domain.model

/**
 * Representa um trecho recuperado de uma norma técnica (NR-10, NR-35, etc.).
 *
 * @property id Identificador único do chunk no índice
 * @property doc Identificador da norma (ex: "NR-10")
 * @property section Título ou número da seção/item (ex: "Item 10.4.1")
 * @property content Texto integral ou trecho relevante da norma
 * @property score Pontuação de relevância BM25 atribuída pelo FTS5
 */
data class Chunk(
    val id: Long = 0,
    val doc: String,
    val section: String,
    val content: String,
    val score: Double = 0.0,
    val title: String = "",
    val page: Int = 0,
    // Telemetria de fusão RRF (Retrieval v3) para o Modo Engenharia. -1 = ausente no sinal.
    val rankBm25: Int = -1,
    val rankDenseText: Int = -1,
    val rankDenseExp: Int = -1,
    val rrfScore: Double = 0.0
)
