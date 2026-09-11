package br.org.ceia.cemigpoc.data.retriever

/**
 * Reciprocal Rank Fusion (RRF) — funde múltiplos rankings de ids num único ranking.
 *
 * score(d) = Σ_sinal  peso_sinal / (k + rank_sinal(d))
 *
 * Config da entrega v3 (retrieval3/README.md): 3 sinais (BM25-gated-expandido,
 * denso texto+expansão, denso expansão-only), k=30, pesos iguais. Espelha exatamente
 * `retrieval3/rrf.py::rrf_fuse` (rank começa em 1). Kotlin puro, sem dependências.
 */
object RrfFusion {

    const val DEFAULT_K = 30.0

    /** Um ranking = lista ordenada de ids (posição 0 = melhor). */
    data class Signal(val ids: List<Long>, val weight: Double = 1.0)

    /** Resultado por id: score RRF + posição (1-based) em cada sinal (-1 se ausente). */
    data class Fused(
        val id: Long,
        val rrfScore: Double,
        val ranks: IntArray  // paralelo à ordem dos sinais fornecidos
    )

    /**
     * Funde os sinais e retorna os ids ordenados por score RRF desc (limitado a [limit]).
     * Os `ranks` de cada id preservam a ordem dos sinais passados (útil p/ telemetria).
     */
    fun fuse(signals: List<Signal>, k: Double = DEFAULT_K, limit: Int = 5): List<Fused> {
        val scores = HashMap<Long, Double>()
        val ranks = HashMap<Long, IntArray>()
        signals.forEachIndexed { si, signal ->
            signal.ids.forEachIndexed { pos, id ->
                val rank1 = pos + 1
                scores[id] = (scores[id] ?: 0.0) + signal.weight / (k + rank1)
                val arr = ranks.getOrPut(id) { IntArray(signals.size) { -1 } }
                if (arr[si] == -1) arr[si] = rank1
            }
        }
        return scores.entries
            .sortedByDescending { it.value }
            .take(limit)
            .map { Fused(it.key, it.value, ranks[it.key] ?: IntArray(signals.size) { -1 }) }
    }
}
