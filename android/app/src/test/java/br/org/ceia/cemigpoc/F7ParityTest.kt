package br.org.ceia.cemigpoc

import br.org.ceia.cemigpoc.data.classifier.NrClassifier
import br.org.ceia.cemigpoc.data.retriever.ContextCompressor
import br.org.ceia.cemigpoc.data.retriever.DenseMatrix
import br.org.ceia.cemigpoc.data.retriever.F6Fusion
import br.org.ceia.cemigpoc.data.retriever.Fts5Retriever
import br.org.ceia.cemigpoc.data.retriever.HybridRetriever
import br.org.ceia.cemigpoc.domain.model.Chunk
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/**
 * PARIDADE app (Kotlin) × espelho Python do ponto de operação da F6 (task lfm3-f7).
 *
 * Fixtures: `bench/lfm3_f7/make_fixtures.py --pipeline` grava, para 40 perguntas do recorte
 * (21 com manuais ligados), o vetor da query, a saída do classificador, os rankings de cada
 * sinal, o top-5 fundido, o orçamento e o CONTEXTO COMPRIMIDO que o espelho entregou ao
 * modelo na régua da F6 (conferido == prompts_recorte.jsonl op_d50).
 *
 * O que é provado aqui (a parte BM25/FTS5 não roda na JVM — o SQLite com FTS5 é nativo
 * Android; o BM25 de normas é o mesmo código já em produção desde a v3):
 *   1. sanitizeFts5Query == app_fts_query (consulta FTS5 das normas e dos manuais);
 *   2. classificador -> gate (hard/soft/none/explicit) com o limiar 0,7 do ponto op;
 *   3. KNN denso dos manuais e gate por margem (m=0) == espelho;
 *   4. fusão RRF 5 sinais (pesos 2,1,2 + 1,75,1,75; k=10) -> top-5 == espelho;
 *   5. compressão (janelas fp16, lexical 0,3, prior 0,05, gulosa até 50%) == contexto do espelho
 *      BYTE A BYTE.
 */
class F7ParityTest {

    companion object {
        private lateinit var cases: JSONArray
        private lateinit var comp: ContextCompressor
        private lateinit var manDense: DenseMatrix
        private val chunkMeta = HashMap<Long, Chunk>()

        @BeforeClass
        @JvmStatic
        fun load() {
            cases = JSONArray(File("src/test/resources/f7_parity/cases.json").readText())
            val assets = File("src/main/assets")
            comp = ContextCompressor.load(File(assets, "units.bin"), File(assets, "units_win3.f16"))
            manDense = DenseMatrix.load(File(assets, "manuals_dense.bin"))
        }

        private fun ids(a: JSONArray): List<Long> = (0 until a.length()).map { a.getLong(it) }
        private fun vec(a: JSONArray): FloatArray = FloatArray(a.length()) { a.getDouble(it).toFloat() }
    }

    @Test
    fun consultaFts5_bateEspelho() {
        val r = Fts5Retriever()
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            assertEquals(c.getString("id"), c.getString("fts_query"), r.sanitizeFts5Query(c.getString("question")))
        }
    }

    @Test
    fun classificadorEGate_bateEspelho() {
        val clf = NrClassifier.loadFromFile(File("src/test/resources/nr_classifier.bin"))
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val q = c.getString("question")
            val preds = clf.classify(q)
            val top = c.getJSONArray("clf_top")
            assertEquals(c.getString("id"), top.getString(0), preds[0].nr)
            assertEquals(c.getDouble("clf_p1"), preds[0].prob.toDouble(), 1e-5)
            val mode = when {
                HybridRetriever.detectExplicitNr(q) != null -> "explicit"
                preds[0].nr == HybridRetriever.NONE_LABEL -> "none"
                preds[0].prob >= F6Fusion.CONF_THR_OP -> "hard"
                else -> "soft"
            }
            assertEquals(c.getString("id"), c.getString("gate_mode"), mode)
        }
    }

    @Test
    fun manuaisDensoEGateMargem_batemEspelho() {
        var on = 0
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val qv = vec(c.getJSONArray("qvec"))
            val maxSim = manDense.maxSim(qv)
            assertEquals(c.getDouble("manual_max_sim"), maxSim.toDouble(), 1e-5)
            assertEquals(c.getString("id"), ids(c.getJSONArray("manual_dense_ids")),
                manDense.topK(qv, 60).map { it.first.toLong() })
            val wanted = F6Fusion.manualWanted(maxSim, c.getDouble("text_top1_sim").toFloat())
            assertEquals(c.getString("id"), c.getBoolean("manual_on"), wanted)
            if (wanted) on++
        }
        assertTrue("fixtures devem exercitar o gate ligado", on >= 10)
    }

    @Test
    fun fusaoTop5_bateEspelho() {
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val on = c.getBoolean("manual_on")
            val fused = F6Fusion.fuse(
                ids(c.getJSONArray("bm25_ids")), ids(c.getJSONArray("dense_text_ids")),
                ids(c.getJSONArray("dense_exp_ids")),
                if (on) ids(c.getJSONArray("manual_bm25_ids")) else null,
                if (on) ids(c.getJSONArray("manual_dense_ids")) else null
            ).take(5).map { it.id }
            assertEquals(c.getString("id"), ids(c.getJSONArray("fused_top5")), fused)
        }
    }

    @Test
    fun compressao_bateContextoDoEspelhoByteAByte() {
        var same = 0
        val diffs = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val r = comp.compress(
                vec(c.getJSONArray("qvec")), c.getString("question"),
                ids(c.getJSONArray("fused_top5")), c.getInt("budget")
            )
            if (r.text == c.getString("context")) same++ else diffs.add(c.getString("id"))
        }
        assertEquals("contextos divergentes: $diffs", cases.length(), same)
    }
}
