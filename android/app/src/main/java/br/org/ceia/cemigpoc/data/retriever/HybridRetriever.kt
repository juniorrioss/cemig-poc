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
    private val topNr: Int = 2,
    // Retrieval v3: busca densa opcional (fusão RRF 3-sinais). Se null, degrada para o
    // caminho BM25-gated legado (comportamento anterior preservado).
    private val denseRetriever: DenseRetriever? = null,
    // Retrieval v4: k e pesos da fusão calibrados no dev-set (nunca no holdout). O combo
    // vencedor é BM25(2) + denso-texto(1) + denso-exp(2), k=10 (ver retrieval4/README.md).
    private val rrfK: Double = 10.0,
    private val weightBm25: Double = 2.0,
    private val weightDenseText: Double = 1.0,
    private val weightDenseExp: Double = 2.0,
    private val fusionPool: Int = 60
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
        val mode: String, // "hard" | "soft" | "none" | "explicit"
        // Retrieval v3: rótulo da estratégia de recuperação efetivamente usada.
        val retrieval: String = "bm25" // "bm25" | "rrf3" | "rrf3-fallback-bm25"
    )

    /** true se a busca densa (fusão RRF 3-sinais) está disponível. */
    val hasDense: Boolean
        get() = denseRetriever != null

    /** Custo do encode denso on-device do último searchV3 (ms), para telemetria. */
    @Volatile
    var lastDenseEncodeMs: Long = 0L
        private set

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

    /**
     * Retrieval v3 (fusão RRF 3-sinais). Estende o caminho gated com dois sinais densos
     * (EmbeddingGemma) fundidos por RRF. Se o denso não estiver disponível/carregado, ou
     * falhar, retorna o top-k BM25-gated (fallback honesto, marcado na decisão).
     *
     * Etapas:
     *   s1 = BM25 gated-expandido (pool)      -> ids ordenados
     *   sT = denso texto+expansão (pool)      -> ids ordenados
     *   sE = denso expansão-only (pool)       -> ids ordenados
     *   RRF(k) -> top-k -> hidrata metadata por id (Fts5Retriever.fetchByIds)
     */
    suspend fun searchV3(rawQuestion: String, topK: Int): List<Chunk> {
        val dense = denseRetriever
        if (dense == null || !dense.isReady) {
            // Sem denso: caminho BM25-gated normal.
            val r = searchWithRaw(bm25Query = rawQuestion, rawQuestion = rawQuestion, topK = topK)
            lastDecision = lastDecision?.copy(retrieval = "rrf3-fallback-bm25")
            return r
        }

        // s1: pool BM25 gated-expandido (reusa a política gated, mas com pool grande).
        val clf = classifier
        val explicit = detectExplicitNr(rawQuestion)
        val bm25Pool: List<Chunk>
        val decision: Decision
        when {
            clf == null -> {
                bm25Pool = delegate.searchBoosted(rawQuestion, fusionPool, emptyList(), 1.0, hardFilter = false)
                decision = Decision(rawQuestion, NONE_LABEL, 0f, NONE_LABEL, emptyList(), "none", "rrf3")
            }
            explicit != null -> {
                bm25Pool = delegate.searchBoosted(rawQuestion, fusionPool, listOf(explicit), boostFactor, hardFilter = true)
                decision = Decision(rawQuestion, explicit, 1.0f, explicit, listOf(explicit), "explicit", "rrf3")
            }
            else -> {
                val preds = clf.classify(rawQuestion)
                val top1 = preds.getOrNull(0)?.nr ?: NONE_LABEL
                val top1Prob = preds.getOrNull(0)?.prob ?: 0f
                val top2 = preds.getOrNull(1)?.nr ?: NONE_LABEL
                when {
                    top1 == NONE_LABEL -> {
                        bm25Pool = delegate.searchBoosted(rawQuestion, fusionPool, emptyList(), 1.0, hardFilter = false)
                        decision = Decision(rawQuestion, top1, top1Prob, top2, emptyList(), "none", "rrf3")
                    }
                    top1Prob >= confThreshold -> {
                        val boost = listOf(top1)
                        bm25Pool = delegate.searchBoosted(rawQuestion, fusionPool, boost, boostFactor, hardFilter = true)
                        decision = Decision(rawQuestion, top1, top1Prob, top2, boost, "hard", "rrf3")
                    }
                    else -> {
                        val boost = listOf(top1, top2).filter { it != NONE_LABEL }
                        bm25Pool = delegate.searchBoosted(rawQuestion, fusionPool, boost, boostFactor, hardFilter = false)
                        decision = Decision(rawQuestion, top1, top1Prob, top2, boost, "soft", "rrf3")
                    }
                }
            }
        }
        lastDecision = decision

        // sT, sE: rankings densos (1 encode serve os dois índices).
        val denseRes = dense.search(rawQuestion, fusionPool)
        lastDenseEncodeMs = denseRes?.encodeMs ?: 0L
        if (denseRes == null) {
            // encode falhou -> fallback BM25 top-k.
            lastDecision = decision.copy(retrieval = "rrf3-fallback-bm25")
            return bm25Pool.take(topK)
        }

        val s1Ids = bm25Pool.map { it.id }
        val sTIds = denseRes.textRanking.map { it.first.toLong() }
        val sEIds = denseRes.expRanking.map { it.first.toLong() }

        val fused = RrfFusion.fuse(
            listOf(
                RrfFusion.Signal(s1Ids, weightBm25),
                RrfFusion.Signal(sTIds, weightDenseText),
                RrfFusion.Signal(sEIds, weightDenseExp)
            ),
            k = rrfK, limit = topK
        )

        // Hidrata metadata por id: usa o pool BM25 quando presente, senão busca no índice.
        val poolById = bm25Pool.associateBy { it.id }
        val missing = fused.map { it.id }.filter { it !in poolById }
        val fetched = if (missing.isNotEmpty()) delegate.fetchByIds(missing) else emptyMap()

        return fused.mapNotNull { f ->
            val base = poolById[f.id] ?: fetched[f.id] ?: return@mapNotNull null
            base.copy(
                score = f.rrfScore,
                rankBm25 = f.ranks.getOrElse(0) { -1 },
                rankDenseText = f.ranks.getOrElse(1) { -1 },
                rankDenseExp = f.ranks.getOrElse(2) { -1 },
                rrfScore = f.rrfScore
            )
        }
    }
}
