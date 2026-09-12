package br.org.ceia.cemigpoc.whisper

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sin

/**
 * Testes da higiene de áudio determinística (recorte de silêncio + normalização de ganho).
 */
class AudioPreprocessingTest {

    private val sr = AudioPreprocessing.SAMPLE_RATE

    /** Gera um tom senoidal (fala simulada) de dada amplitude e duração em amostras. */
    private fun tone(amp: Float, n: Int, freq: Float = 220f): FloatArray {
        return FloatArray(n) { i -> amp * sin(2.0 * Math.PI * freq * i / sr).toFloat() }
    }

    @Test
    fun normalizeGain_amplificaFalaFraca() {
        val weak = tone(0.1f, sr) // 1 s de tom fraco (pico ~0.1)
        val out = AudioPreprocessing.normalizeGain(weak)
        val peak = out.maxOf { abs(it) }
        // Deve subir para próximo do alvo 0.95 sem clipar.
        assertTrue("pico após normalização deveria ~0.95, foi $peak", peak in 0.90f..0.96f)
    }

    @Test
    fun normalizeGain_naoAmplificaSilencioRuido() {
        val noise = tone(0.005f, sr) // abaixo do piso MIN_PEAK_TO_NORMALIZE
        val out = AudioPreprocessing.normalizeGain(noise)
        val peak = out.maxOf { abs(it) }
        assertTrue("ruído não deve ser amplificado, pico $peak", peak < 0.02f)
    }

    @Test
    fun trimSilence_removePontasSilenciosas() {
        // 0,5 s de silêncio + 1 s de fala + 0,5 s de silêncio.
        val silenceHead = FloatArray(sr / 2) { 0f }
        val speech = tone(0.6f, sr)
        val silenceTail = FloatArray(sr / 2) { 0f }
        val full = silenceHead + speech + silenceTail

        val out = AudioPreprocessing.trimSilence(full)
        // Deve encolher (removeu boa parte do silêncio), mas preservar a fala + padding.
        assertTrue("deveria encolher: ${full.size} -> ${out.size}", out.size < full.size)
        assertTrue("deve manter ao menos ~1 s de fala", out.size >= sr)
    }

    @Test
    fun process_curtoRetornaOriginal() {
        val tiny = FloatArray(100) { 0.1f }
        val out = AudioPreprocessing.process(tiny)
        assertEquals(tiny.size, out.size)
    }
}
