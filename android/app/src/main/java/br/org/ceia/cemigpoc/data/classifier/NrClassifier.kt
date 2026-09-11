package br.org.ceia.cemigpoc.data.classifier

import android.content.Context
import android.util.Log
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer

/**
 * Classificador leve de Norma Regulamentadora (NR) — estágio 1 do pipeline híbrido.
 *
 * Reimplementa em Kotlin puro (sem runtime ONNX) o pipeline vencedor do shootout
 * (classifier/train_classic.py): TF-IDF híbrido (char_wb 2-5 + word 1-2, sublinear_tf,
 * norma L2 POR BLOCO) seguido de LogisticRegression multinomial + softmax. O algoritmo
 * é validado com paridade Δprob < 1e-7 contra o sklearn em classifier/export_kotlin.py.
 *
 * Artefato: `nr_classifier.bin` (formato NRC1, ~4 MB), embarcado nos assets ou
 * sobrescrito via getExternalFilesDir(null)/nr_classifier.bin (injetável via adb push).
 *
 * Uso: `classify(fala)` -> lista ordenada de (nr, prob). A top-1/top-2 alimenta o boost
 * suave do Fts5Retriever; o gate de confiança decide filtro-duro vs boost-suave.
 */
class NrClassifier private constructor(
    private val classes: Array<String>,
    private val nChar: Int,
    private val nWord: Int,
    private val charVocab: HashMap<String, Int>,
    private val wordVocab: HashMap<String, Int>,
    private val idf: FloatArray,
    private val coef: Array<FloatArray>,      // [nClasses][nFeat]
    private val intercept: FloatArray
) {
    val numClasses: Int get() = classes.size
    val numFeatures: Int get() = idf.size

    data class Prediction(val nr: String, val prob: Float)

    companion object {
        private const val TAG = "NrClassifier"
        const val ASSET_NAME = "nr_classifier.bin"
        private const val MAGIC = "NRC2"

        /** Carrega o modelo priorizando override externo (adb push) sobre assets. */
        fun load(context: Context): NrClassifier? {
            val external = context.getExternalFilesDir(null)?.resolve(ASSET_NAME)
            return try {
                if (external != null && external.exists() && external.length() > 0) {
                    Log.i(TAG, "Carregando classificador NR externo: ${external.absolutePath}")
                    DataInputStream(FileInputStream(external).buffered()).use { readModel(it) }
                } else {
                    Log.i(TAG, "Carregando classificador NR dos assets: $ASSET_NAME")
                    DataInputStream(context.assets.open(ASSET_NAME).buffered()).use { readModel(it) }
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Falha ao carregar $ASSET_NAME; estágio 1 desativado (fallback busca sem boost)", e)
                null
            }
        }

        /** Carrega diretamente de um arquivo (testes JVM). */
        fun loadFromFile(file: File): NrClassifier =
            DataInputStream(FileInputStream(file).buffered()).use { readModel(it) }

        /**
         * Lê o formato binário NRC2 auto-contido (little-endian):
         *   [magic 4B][nClasses u16][nFeat u32][nChar u32]
         *   classes: nClasses x { [len u16][utf8] }
         *   vocab:   nFeat  x { [len u16][ngram utf8][idf f32] }  (char em [0,nChar), depois word)
         *   coef: nClasses*nFeat f32 (linha-maior) ; intercept: nClasses f32
         */
        private fun readModel(din: DataInputStream): NrClassifier {
            val magic = ByteArray(4).also { din.readFully(it) }.toString(Charsets.US_ASCII)
            require(magic == MAGIC) { "Formato inválido: magic=$magic" }
            val nClasses = readU16(din)
            val nFeat = readU32(din)
            val nChar = readU32(din)

            val classes = Array(nClasses) {
                val len = readU16(din)
                String(ByteArray(len).also { b -> din.readFully(b) }, Charsets.UTF_8)
            }
            val charVocab = HashMap<String, Int>(nChar)
            val wordVocab = HashMap<String, Int>(nFeat - nChar)
            val idf = FloatArray(nFeat)
            for (col in 0 until nFeat) {
                val len = readU16(din)
                val gram = String(ByteArray(len).also { b -> din.readFully(b) }, Charsets.UTF_8)
                idf[col] = readF32(din)
                if (col < nChar) charVocab[gram] = col else wordVocab[gram] = col
            }
            val coef = Array(nClasses) { FloatArray(nFeat) }
            val buf = ByteArray(4 * nFeat)
            for (r in 0 until nClasses) {
                din.readFully(buf)
                val bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN)
                for (c in 0 until nFeat) coef[r][c] = bb.float
            }
            val intercept = FloatArray(nClasses)
            run {
                val ib = ByteArray(4 * nClasses).also { din.readFully(it) }
                val bb = ByteBuffer.wrap(ib).order(ByteOrder.LITTLE_ENDIAN)
                for (r in 0 until nClasses) intercept[r] = bb.float
            }
            return NrClassifier(
                classes = classes,
                nChar = nChar, nWord = nFeat - nChar,
                charVocab = charVocab, wordVocab = wordVocab,
                idf = idf, coef = coef, intercept = intercept
            )
        }

        private fun readU16(d: DataInputStream): Int {
            val b0 = d.readUnsignedByte(); val b1 = d.readUnsignedByte()
            return b0 or (b1 shl 8)
        }

        private fun readU32(d: DataInputStream): Int {
            val b0 = d.readUnsignedByte(); val b1 = d.readUnsignedByte()
            val b2 = d.readUnsignedByte(); val b3 = d.readUnsignedByte()
            return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
        }

        private fun readF32(d: DataInputStream): Float {
            val b = ByteArray(4).also { d.readFully(it) }
            return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).float
        }
    }

    /** Sanitização idêntica a text_utils.preprocess (strip_accents NFKD + lowercase + trim). */
    private fun preprocess(text: String): String {
        val nfkd = Normalizer.normalize(text, Normalizer.Form.NFKD).replace(Regex("\\p{M}"), "")
        return nfkd.lowercase().trim()
    }

    /** char_wb n-grams idêntico ao sklearn (_char_wb_ngrams): padding ' ', break em palavra curta. */
    private fun charWbNgrams(text: String, nmin: Int, nmax: Int): List<String> {
        val out = ArrayList<String>()
        for (token in text.split(Regex("\\s+"))) {
            if (token.isEmpty()) continue
            val w = " $token "
            val wLen = w.length
            for (n in nmin..nmax) {
                var offset = 0
                out.add(w.substring(offset, minOf(offset + n, wLen)))
                while (offset + n < wLen) {
                    offset += 1
                    out.add(w.substring(offset, offset + n))
                }
                if (offset == 0) break // palavra curta: conta 1x e para
            }
        }
        return out
    }

    /** word n-grams idêntico ao token_pattern padrão do sklearn (\b\w\w+\b). */
    private fun wordNgrams(text: String, nmin: Int, nmax: Int): List<String> {
        val tokens = Regex("\\b\\w\\w+\\b").findAll(text).map { it.value }.toList()
        val out = ArrayList<String>()
        for (n in nmin..nmax) {
            for (i in 0..(tokens.size - n)) {
                out.add(tokens.subList(i, i + n).joinToString(" "))
            }
        }
        return out
    }

    /**
     * Featuriza a fala em vetor TF-IDF esparso (map col->valor), com L2 POR BLOCO
     * (char e word normalizados separadamente, replicando os dois TfidfVectorizer + hstack).
     */
    private fun featurize(rawText: String): HashMap<Int, Float> {
        val text = preprocess(rawText)
        val counts = HashMap<Int, Float>()
        for (ng in charWbNgrams(text, 2, 5)) charVocab[ng]?.let { counts[it] = (counts[it] ?: 0f) + 1f }
        for (ng in wordNgrams(text, 1, 2)) wordVocab[ng]?.let { counts[it] = (counts[it] ?: 0f) + 1f }
        // sublinear_tf + idf
        var sumSqChar = 0.0
        var sumSqWord = 0.0
        for ((col, tf) in counts) {
            val v = (1.0f + Math.log(tf.toDouble()).toFloat()) * idf[col]
            counts[col] = v
            if (col < nChar) sumSqChar += (v * v).toDouble() else sumSqWord += (v * v).toDouble()
        }
        val normChar = Math.sqrt(sumSqChar).toFloat()
        val normWord = Math.sqrt(sumSqWord).toFloat()
        for ((col, v) in counts) {
            val nrm = if (col < nChar) normChar else normWord
            if (nrm > 0f) counts[col] = v / nrm
        }
        return counts
    }

    /**
     * Classifica a fala e retorna as predições ordenadas por probabilidade (softmax).
     */
    fun classify(rawText: String): List<Prediction> {
        val feats = featurize(rawText)
        val logits = FloatArray(classes.size)
        for (r in classes.indices) {
            var acc = intercept[r]
            val row = coef[r]
            for ((col, v) in feats) acc += row[col] * v
            logits[r] = acc
        }
        // softmax
        var maxLogit = Float.NEGATIVE_INFINITY
        for (l in logits) if (l > maxLogit) maxLogit = l
        var sum = 0.0
        val exps = DoubleArray(classes.size)
        for (i in logits.indices) {
            val e = Math.exp((logits[i] - maxLogit).toDouble()); exps[i] = e; sum += e
        }
        val preds = ArrayList<Prediction>(classes.size)
        for (i in classes.indices) preds.add(Prediction(classes[i], (exps[i] / sum).toFloat()))
        preds.sortByDescending { it.prob }
        return preds
    }
}
