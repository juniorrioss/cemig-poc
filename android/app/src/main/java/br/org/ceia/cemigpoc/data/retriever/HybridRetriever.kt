package br.org.ceia.cemigpoc.data.retriever

import android.util.Log
import br.org.ceia.cemigpoc.data.classifier.NrClassifier
import br.org.ceia.cemigpoc.domain.engine.Retriever
import br.org.ceia.cemigpoc.domain.model.Chunk

/**
 * Retriever híbrido de 2 estágios (Opção B aprovada pelo capitão).
 *
 * Estágio 1: NrClassifier (TF-IDF + LogReg em Kotlin puro) prevê top-1/top-2 NR a partir
 * da FALA BRUTA do trabalhador (medido melhor que reescrita para este boost — ver
 * classifier/README.md). Estágio 2: BM25 (Fts5Retriever) com BOOST SUAVE nas NRs previstas.
 *
 * Política CONFIANÇA-GATED (melhor R@2 no harness das 151 reais: 29.8% vs baseline 13.9%):
 *   - se prob(top-1) >= confThreshold  -> filtro DURO na top-1 (classificador confiante);
 *   - senão                            -> boost SUAVE (fator 5x) nas top-2 (mais seguro);
 *   - se top-1 == 'nenhuma'            -> busca SEM boost (fora de escopo das NRs).
 *
 * IMPORTANTE: o estágio 1 usa a FALA BRUTA (rawQuestion), não as keywords reescritas; o
 * BM25 continua recebendo as keywords do Turno 1 do AskPipeline via search(). Quando o
 * chamador dispõe da fala bruta, deve usar searchWithRaw() para ativar o estágio 1.
 */
class HybridRetriever(
    private val delegate: Fts5Retriever,
    private val classifier: NrClassifier?,
    private val confThreshold: Float = 0.5f,
    private val boostFactor: Double = 5.0,
    private val topNr: Int = 2
) : Retriever {

    companion object {
        private const val TAG = "HybridRetriever"
        const val NONE_LABEL = "nenhuma"
        // Detecta menção explícita de norma na fala (ex.: "segundo a NR-10", "pela nr 35").
        private val EXPLICIT_NR = Regex("\\bnr[\\s._-]*(\\d{1,2})\\b", RegexOption.IGNORE_CASE)

        /** Extrai a norma explícita citada na fala, normalizada em nr-XX; null se ausente. */
        fun detectExplicitNr(text: String): String? {
            val m = EXPLICIT_NR.find(text) ?: return null
            val n = m.groupValues[1].toIntOrNull() ?: return null
            return "nr-%02d".format(n)
        }
    }

    /** Última decisão do estágio 1, para telemetria/modo engenharia. */
    @Volatile
    var lastDecision: Decision? = null
        private set

    data class Decision(
        val rawQuestion: String,
        val top1: String,
        val top1Prob: Float,
        val top2: String,
        val boostNrs: List<String>,
        val mode: String // "hard" | "soft" | "none"
    )

    /**
     * Busca compatível com a interface Retriever. Sem a fala bruta separada, o estágio 1
     * classifica a própria query recebida (keywords). Preferir searchWithRaw() quando houver
     * a fala original disponível.
     */
    override suspend fun search(query: String, topK: Int): List<Chunk> =
        searchWithRaw(bm25Query = query, rawQuestion = query, topK = topK)

    /**
     * Busca híbrida com fala bruta explícita para o estágio 1 e query BM25 (keywords) separada.
     */
    suspend fun searchWithRaw(bm25Query: String, rawQuestion: String, topK: Int): List<Chunk> {
        val clf = classifier
        if (clf == null) {
            // Sem classificador: degrada graciosamente para BM25 puro (comportamento atual).
            return delegate.search(bm25Query, topK)
        }

        // Override 1: se o trabalhador cita a norma explicitamente, confiamos nela (filtro duro).
        val explicit = detectExplicitNr(rawQuestion)
        if (explicit != null) {
            val boost = listOf(explicit)
            lastDecision = Decision(rawQuestion, explicit, 1.0f, explicit, boost, "explicit")
            Log.i(TAG, "Estágio 1: NR explícita na fala '$explicit' -> filtro duro")
            return delegate.searchBoosted(bm25Query, topK, boost, boostFactor, hardFilter = true)
        }

        val preds = clf.classify(rawQuestion)
        val top1 = preds.getOrNull(0)?.nr ?: NONE_LABEL
        val top1Prob = preds.getOrNull(0)?.prob ?: 0f
        val top2 = preds.getOrNull(1)?.nr ?: NONE_LABEL

        val decision: Decision
        val chunks: List<Chunk>
        when {
            top1 == NONE_LABEL -> {
                // Fora de escopo das NRs -> busca ampla sem boost.
                decision = Decision(rawQuestion, top1, top1Prob, top2, emptyList(), "none")
                chunks = delegate.searchBoosted(bm25Query, topK, emptyList(), 1.0, hardFilter = false)
            }
            top1Prob >= confThreshold -> {
                // Classificador confiante -> filtro DURO na top-1.
                val boost = listOf(top1)
                decision = Decision(rawQuestion, top1, top1Prob, top2, boost, "hard")
                chunks = delegate.searchBoosted(bm25Query, topK, boost, boostFactor, hardFilter = true)
            }
            else -> {
                // Incerto -> boost SUAVE nas top-2 (não exclui outras NRs).
                val boost = listOf(top1, top2).filter { it != NONE_LABEL }
                decision = Decision(rawQuestion, top1, top1Prob, top2, boost, "soft")
                chunks = delegate.searchBoosted(bm25Query, topK, boost, boostFactor, hardFilter = false)
            }
        }
        lastDecision = decision
        Log.i(
            TAG,
            "Estágio 1: top1=%s (%.2f) top2=%s -> modo=%s boost=%s".format(
                top1, top1Prob, top2, decision.mode, decision.boostNrs
            )
        )
        return chunks
    }
}
