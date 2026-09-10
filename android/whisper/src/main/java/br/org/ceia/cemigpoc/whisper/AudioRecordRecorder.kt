package br.org.ceia.cemigpoc.whisper

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Gravador de áudio via AudioRecord configurado para 16 kHz Mono PCM 16-bit.
 * Formato padrão consumido pelo Whisper.
 */
class AudioRecordRecorder {

    companion object {
        private const val TAG = "AudioRecordRecorder"
        const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    }

    private var audioRecord: AudioRecord? = null
    @Volatile
    private var isRecording = false
    private val bufferStream = ByteArrayOutputStream()

    val recording: Boolean
        get() = isRecording

    @SuppressLint("MissingPermission")
    fun start() {
        if (isRecording) return

        val minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        val bufferSize = maxOf(minBufferSize, SAMPLE_RATE * 2) // Pelo menos 1 segundo de buffer

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord não foi inicializado corretamente")
                audioRecord?.release()
                audioRecord = null
                return
            }

            bufferStream.reset()
            isRecording = true
            audioRecord?.startRecording()
            Log.i(TAG, "Gravação iniciada (16 kHz mono)")

            Thread {
                val tempBuffer = ByteArray(2048)
                while (isRecording && audioRecord != null) {
                    val read = audioRecord?.read(tempBuffer, 0, tempBuffer.size) ?: -1
                    if (read > 0) {
                        synchronized(bufferStream) {
                            bufferStream.write(tempBuffer, 0, read)
                        }
                    }
                }
            }.start()

        } catch (e: Exception) {
            Log.e(TAG, "Erro ao iniciar AudioRecord", e)
            isRecording = false
            audioRecord?.release()
            audioRecord = null
        }
    }

    /**
     * Encerra a gravação e converte os bytes PCM 16-bit para array de floats normalizados [-1.0, 1.0].
     */
    fun stop(): FloatArray {
        if (!isRecording && bufferStream.size() == 0) return FloatArray(0)

        isRecording = false
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            Log.w(TAG, "Exceção ao parar AudioRecord", e)
        }
        audioRecord = null

        val pcmBytes: ByteArray
        synchronized(bufferStream) {
            pcmBytes = bufferStream.toByteArray()
            bufferStream.reset()
        }

        if (pcmBytes.isEmpty()) {
            return FloatArray(0)
        }

        // Converte PCM 16-bit little endian para Float [-1.0f, 1.0f]
        val shortBuffer = ByteBuffer.wrap(pcmBytes)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()

        val floatArray = FloatArray(shortBuffer.remaining())
        for (i in floatArray.indices) {
            floatArray[i] = shortBuffer.get() / 32768.0f
        }

        Log.i(TAG, "Gravação finalizada: ${floatArray.size} amostras (%.2fs)".format(floatArray.size / 16000.0f))
        return floatArray
    }
}
