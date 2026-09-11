package br.org.ceia.cemigpoc

import br.org.ceia.cemigpoc.data.classifier.NrClassifier
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Valida o classificador de NR em Kotlin puro contra o gabarito exportado do sklearn
 * (classifier/export_kotlin.py grava paridade Δprob < 1e-7). O fixture
 * kotlin_test_fixture.json traz a top-2 e probs de referência do sklearn.
 */
class NrClassifierTest {

    private fun modelFile(): File {
        val f = File("src/test/resources/nr_classifier.bin")
        assertTrue("modelo de teste ausente: ${f.absolutePath}", f.exists())
        return f
    }

    @Test
    fun loadsModelWith36ClassesAnd37Labels() {
        val clf = NrClassifier.loadFromFile(modelFile())
        // 36 NRs + rótulo 'nenhuma' = 37 classes
        assertEquals(37, clf.numClasses)
        assertTrue("features esperadas > 20000", clf.numFeatures > 20000)
    }

    @Test
    fun predictsElectricalQuestionAsNr10() {
        val clf = NrClassifier.loadFromFile(modelFile())
        val preds = clf.classify(
            "Tô na subestação e não dá pra desligar o circuito, posso usar só a luva de borracha?"
        )
        assertEquals("nr-10", preds[0].nr)
        assertTrue("prob top-1 deve ser alta", preds[0].prob > 0.5f)
    }

    @Test
    fun predictsHeightQuestionWithNr35AndNr10InTop2() {
        val clf = NrClassifier.loadFromFile(modelFile())
        val preds = clf.classify("Vou subir num poste de 8 metros pra trocar o transformador, preciso de cinto?")
        val top2 = setOf(preds[0].nr, preds[1].nr)
        assertTrue("top-2 deve conter nr-35 e nr-10: $top2", top2.containsAll(listOf("nr-35", "nr-10")))
    }

    @Test
    fun predictsOutOfScopeAsNenhuma() {
        val clf = NrClassifier.loadFromFile(modelFile())
        val preds = clf.classify("Como preencher o ponto pra receber hora extra?")
        assertEquals("nenhuma", preds[0].nr)
    }

    @Test
    fun matchesSklearnReferenceWithinTolerance() {
        val clf = NrClassifier.loadFromFile(modelFile())
        val fixture = File("src/test/resources/kotlin_test_fixture.json").readText()
        // parsing mínimo do JSON (evita dependência): extrai blocos por "text"
        val regex = Regex("\"text\":\\s*\"(.*?)\",\\s*\"top2\":\\s*\\[\\s*\\[\\s*\"(nr-\\d+|nenhuma)\",\\s*([0-9.]+)")
        var checked = 0
        for (m in regex.findAll(fixture.replace("\\n", " "))) {
            val text = m.groupValues[1].replace("\\\"", "\"")
            val expNr = m.groupValues[2]
            val expProb = m.groupValues[3].toFloat()
            val preds = clf.classify(text)
            assertEquals("top-1 NR difere p/ '$text'", expNr, preds[0].nr)
            assertTrue(
                "prob top-1 difere > 0.02 p/ '$text' (esperado $expProb, obtido ${preds[0].prob})",
                Math.abs(preds[0].prob - expProb) < 0.02f
            )
            checked++
        }
        assertTrue("nenhum caso verificado do fixture", checked >= 3)
    }
}
