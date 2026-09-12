package br.org.ceia.cemigpoc

import br.org.ceia.cemigpoc.data.retriever.RrfFusion
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Trava da configuração de fusão do Retrieval v4 (retrieval4/README.md):
 * k=10 e pesos (BM25=2, denso-texto=1, denso-exp=2). Confere a paridade numérica com a
 * fórmula RRF ponderada e a preferência esperada por ids reforçados nos sinais de peso 2.
 */
class RrfFusionV4Test {

    @Test
    fun fusaoPonderada_k10_bateFormula() {
        // s1 (BM25, peso 2): id 100 em 1º; s2 (texto, peso 1): id 200 em 1º;
        // s3 (exp, peso 2): id 100 em 1º. id 100 aparece em 2 sinais de peso 2.
        val s1 = RrfFusion.Signal(listOf(100L, 200L, 300L), weight = 2.0)
        val s2 = RrfFusion.Signal(listOf(200L, 100L, 400L), weight = 1.0)
        val s3 = RrfFusion.Signal(listOf(100L, 300L, 200L), weight = 2.0)

        val fused = RrfFusion.fuse(listOf(s1, s2, s3), k = 10.0, limit = 5)

        // score(100) = 2/(10+1) + 1/(10+2) + 2/(10+1) = 0.181818 + 0.083333 + 0.181818
        val expected100 = 2.0 / 11 + 1.0 / 12 + 2.0 / 11
        val got100 = fused.first { it.id == 100L }.rrfScore
        assertEquals(expected100, got100, 1e-9)

        // id 100 deve ser o 1º (reforçado nos dois sinais de peso 2).
        assertEquals(100L, fused.first().id)
    }

    @Test
    fun pesoMaior_venceEmpateDePosicao() {
        // Dois ids empatam na 1ª posição, mas em sinais de pesos diferentes.
        val forte = RrfFusion.Signal(listOf(10L), weight = 2.0)
        val fraco = RrfFusion.Signal(listOf(20L), weight = 1.0)
        val fused = RrfFusion.fuse(listOf(forte, fraco), k = 10.0, limit = 2)
        assertEquals(10L, fused.first().id)
        assertTrue(fused.first { it.id == 10L }.rrfScore > fused.first { it.id == 20L }.rrfScore)
    }
}
