package br.org.ceia.cemigpoc.data.retriever

import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Matriz densa em memória (ids alinhados a count × dim, fp32) com KNN exato por produto
 * interno — vetores L2-normalizados => cosseno. Kotlin puro (testável na JVM).
 *
 * Formato DVEC1 (retrieval3/export_dense_bin.py, bench/lfm3_f7/export_assets.py):
 *   "DVEC1" + dim(i32 LE) + count(i32 LE) + count × [id(i32 LE) + dim × f32 LE]
 */
class DenseMatrix(val ids: IntArray, val dim: Int, val data: FloatArray) {
    val count: Int get() = ids.size

    fun dot(row: Int, q: FloatArray): Float {
        var s = 0f
        val base = row * dim
        var d = 0
        while (d < dim) {
            s += data[base + d] * q[d]
            d++
        }
        return s
    }

    /** top-k (id, score) desc; empates pela ordem das linhas (sort estável). */
    fun topK(q: FloatArray, k: Int): List<Pair<Int, Float>> {
        val scores = FloatArray(count) { dot(it, q) }
        return (0 until count).sortedByDescending { scores[it] }.take(k).map { ids[it] to scores[it] }
    }

    fun maxSim(q: FloatArray): Float {
        var m = Float.NEGATIVE_INFINITY
        for (i in 0 until count) {
            val s = dot(i, q)
            if (s > m) m = s
        }
        return m
    }

    companion object {
        private const val MAGIC = "DVEC1"

        fun load(file: File): DenseMatrix {
            DataInputStream(file.inputStream().buffered(1 shl 16)).use { input ->
                val magic = ByteArray(5)
                input.readFully(magic)
                check(String(magic) == MAGIC) { "Formato inválido em ${file.name}" }
                val header = ByteArray(8)
                input.readFully(header)
                val hb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
                val dim = hb.int
                val count = hb.int
                val ids = IntArray(count)
                val data = FloatArray(count * dim)
                val rec = ByteArray(4 + dim * 4)
                for (i in 0 until count) {
                    input.readFully(rec)
                    val rb = ByteBuffer.wrap(rec).order(ByteOrder.LITTLE_ENDIAN)
                    ids[i] = rb.int
                    val off = i * dim
                    for (d in 0 until dim) data[off + d] = rb.float
                }
                return DenseMatrix(ids, dim, data)
            }
        }
    }
}
