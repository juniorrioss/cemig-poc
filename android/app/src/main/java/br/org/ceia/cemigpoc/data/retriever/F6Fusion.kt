package br.org.ceia.cemigpoc.data.retriever

/**
 * Fusão do PONTO DE OPERAÇÃO da F6 (`thr0.7_f2+man1.75+margin`) — Kotlin puro, porte de
 * `bench/lfm3_f5/retriever_f5.py::F5Retriever.search` com `Cfg(conf_thr=0.7, factor=2,
 * manual_w=1.75, manual_gate="margin", manual_margin=0)`.
 *
 *   sinais = [BM25 gated (normas), denso-texto, denso-exp]          pesos (2, 1, 2)
 *          + [BM25 manual, denso manual]  SE  max cos(manual) >= cos(top-1 denso-texto) + m
 *                                                                    pesos (1.75, 1.75)
 *   RRF k=10 -> top-5
 *
 * O controle `v4app` (gate 0.5 / boost 5x, sem manuais) usa os mesmos densos e só serve para
 * o ORÇAMENTO da compressão (50% dos tokens do top-2 cru dele).
 */
object F6Fusion {
    const val RRF_K = 10.0
    const val W_BM25 = 2.0
    const val W_TEXT = 1.0
    const val W_EXP = 2.0
    const val MANUAL_W = 1.75
    const val MANUAL_MARGIN = 0.0f
    const val CONF_THR_OP = 0.7f
    const val BOOST_OP = 2.0
    const val CONF_THR_CTRL = 0.5f
    const val BOOST_CTRL = 5.0
    const val TOP_K = 5

    /** Gate por margem densa (m=0): o melhor manual é tão parecido quanto o melhor de norma. */
    fun manualWanted(manualMaxSim: Float, textTop1Sim: Float, margin: Float = MANUAL_MARGIN): Boolean =
        manualMaxSim >= textTop1Sim + margin

    fun fuse(
        bm25: List<Long>,
        denseText: List<Long>,
        denseExp: List<Long>,
        manualBm25: List<Long>? = null,
        manualDense: List<Long>? = null,
        limit: Int = TOP_K
    ): List<RrfFusion.Fused> {
        val signals = mutableListOf(
            RrfFusion.Signal(bm25, W_BM25),
            RrfFusion.Signal(denseText, W_TEXT),
            RrfFusion.Signal(denseExp, W_EXP)
        )
        if (manualBm25 != null && manualDense != null) {
            signals.add(RrfFusion.Signal(manualBm25, MANUAL_W))
            signals.add(RrfFusion.Signal(manualDense, MANUAL_W))
        }
        return RrfFusion.fuse(signals, k = RRF_K, limit = maxOf(limit, 5))
    }
}
