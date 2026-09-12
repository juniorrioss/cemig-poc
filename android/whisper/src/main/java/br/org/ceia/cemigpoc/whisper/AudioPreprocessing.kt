package br.org.ceia.cemigpoc.whisper

import android.util.Log
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Higiene de áudio barata (CPU-only, sem dependências) aplicada antes do Whisper.
 *
 * Motivação (task poc-asr-fix): o capitão relatou que a transcrição só acerta com fala
 * lenta e pausada. Fala natural em campo traz silêncio nas pontas (dedo no PTT antes de
 * falar / solta depois), ganho baixo (microfone longe, luva) e variação de nível. Estes
 * três passos determinísticos aproximam o sinal do que o Whisper Base espera:
 *
 * 1. Recorte de silêncio nas pontas (leading/trailing) por limiar de energia RMS.
 * 2. Normalização de ganho de pico (evita clipping, sobe fala fraca a um alvo).
 * 3. O sinal já chega 16 kHz mono float [-1,1] do AudioRecordRecorder (confirmado).
 *
 * Tudo O(n) sobre o array de amostras; custo desprezível vs. a inferência do Whisper.
 */
object AudioPreprocessing {

    private const val TAG = "AudioPreprocessing"

    const val SAMPLE_RATE = 16000

    // Alvo de pico após normalização. 0.95 dá folga para não clipar em 1.0.
    private const val PEAK_TARGET = 0.95f

    // Piso de pico abaixo do qual consideramos "silêncio/ruído" e NÃO amplificamos
    // (evita estourar ruído de fundo quando não houve fala real).
    private const val MIN_PEAK_TO_NORMALIZE = 0.02f

    // Janela de análise de energia para o recorte de silêncio (20 ms @ 16 kHz).
    private const val FRAME_SIZE = 320

    // Fração da energia RMS máxima usada como limiar de voz. Frames abaixo disso nas
    // pontas são cortados; o miolo é sempre preservado.
    private const val SILENCE_RMS_FRACTION = 0.08f

    // Margem preservada em torno da fala detectada (100 ms), evita cortar ataque/cauda.
    private const val PADDING_SAMPLES = SAMPLE_RATE / 10

    /**
     * Aplica recorte de silêncio nas pontas + normalização de ganho.
     * Retorna o próprio array se estiver vazio ou curto demais para valer a pena.
     */
    fun process(samples: FloatArray): FloatArray {
        if (samples.size < FRAME_SIZE * 2) return samples
        val trimmed = trimSilence(samples)
        return normalizeGain(trimmed)
    }

    /**
     * Recorta silêncio de baixa energia nas pontas usando RMS por frame com limiar
     * relativo ao pico de energia do enunciado. Mantém o miolo intacto.
     */
    internal fun trimSilence(samples: FloatArray): FloatArray {
        val nFrames = samples.size / FRAME_SIZE
        if (nFrames < 2) return samples

        val rms = FloatArray(nFrames)
        var maxRms = 0f
        for (f in 0 until nFrames) {
            var sum = 0.0
            val start = f * FRAME_SIZE
            for (i in start until start + FRAME_SIZE) {
                val s = samples[i]
                sum += (s * s).toDouble()
            }
            val r = sqrt(sum / FRAME_SIZE).toFloat()
            rms[f] = r
            if (r > maxRms) maxRms = r
        }

        if (maxRms <= 0f) return samples
        val threshold = maxRms * SILENCE_RMS_FRACTION

        var firstVoiced = 0
        while (firstVoiced < nFrames && rms[firstVoiced] < threshold) firstVoiced++
        var lastVoiced = nFrames - 1
        while (lastVoiced > firstVoiced && rms[lastVoiced] < threshold) lastVoiced--

        // Nenhuma fala detectada acima do limiar -> devolve original (deixa o Whisper decidir).
        if (firstVoiced >= lastVoiced) return samples

        val startSample = (firstVoiced * FRAME_SIZE - PADDING_SAMPLES).coerceAtLeast(0)
        val endSample = ((lastVoiced + 1) * FRAME_SIZE + PADDING_SAMPLES).coerceAtMost(samples.size)
        if (startSample >= endSample) return samples

        val outLen = endSample - startSample
        // Se o recorte é irrelevante (<5% removido), não copia à toa.
        if (outLen >= samples.size - FRAME_SIZE) return samples

        Log.i(TAG, "Recorte de silêncio: ${samples.size} -> $outLen amostras (%.2fs -> %.2fs)"
            .format(samples.size / 16000.0f, outLen / 16000.0f))
        return samples.copyOfRange(startSample, endSample)
    }

    /**
     * Normalização de pico: escala o sinal para que o pico absoluto atinja PEAK_TARGET.
     * Não amplifica quando o pico está abaixo de MIN_PEAK_TO_NORMALIZE (provável só ruído).
     */
    internal fun normalizeGain(samples: FloatArray): FloatArray {
        if (samples.isEmpty()) return samples
        var peak = 0f
        for (s in samples) {
            val a = abs(s)
            if (a > peak) peak = a
        }
        if (peak < MIN_PEAK_TO_NORMALIZE || peak <= 0f) return samples

        val gain = PEAK_TARGET / peak
        // Já está próximo do alvo: evita trabalho e ruído numérico.
        if (gain in 0.98f..1.02f) return samples

        val out = FloatArray(samples.size)
        for (i in samples.indices) {
            out[i] = (samples[i] * gain).coerceIn(-1.0f, 1.0f)
        }
        Log.i(TAG, "Normalização de ganho: pico %.3f -> alvo %.2f (ganho %.2fx)".format(peak, PEAK_TARGET, gain))
        return out
    }
}
