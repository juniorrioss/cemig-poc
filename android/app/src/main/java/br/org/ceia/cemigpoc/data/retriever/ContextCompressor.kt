package br.org.ceia.cemigpoc.data.retriever

import br.org.ceia.cemigpoc.domain.model.Chunk
import java.io.DataInputStream
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import java.text.Normalizer

/**
 * COMPRESSÃO de contexto por relevância (F5 frente 3 + F6 manuais) — porte fiel de
 * `bench/lfm3_f5/compress.py::Compressor.compress` (mode=hybrid, lam=0.3, prior=0.05).
 *
 * Índice pré-computado nos assets (bench/lfm3_f7/export_assets.py):
 *   - units.bin (UNIT1): por chunk (2202 normas + 1092 manuais) o rótulo de fonte, o prefixo,
 *     o custo em tokens do cabeçalho; por unidade o texto, o item, starts_item, os radicais
 *     lexicais e o custo em tokens (= ntok(_unit_text)+3 do espelho).
 *   - units_win3.f16 (HVEC1): embedding EmbeddingGemma da JANELA-3 de cada unidade (fp16;
 *     fp16 reproduz 344/344 contextos do espelho, int8 só 332/344). Mapeado em memória.
 *
 * score(unidade) = cos(query, janela) + 0.3·sobreposição lexical (radical-6) + 0.05/(rank+1).
 * Seleção GULOSA global até o orçamento; render na ordem do ranking por chunk, unidades em
 * ordem de documento, "[…]" nas lacunas, "(item X)" quando a unidade começa no meio de um item.
 * Orçamento = 50% dos tokens do top-2 CRU do controle v4app da mesma fala (F6).
 */
class ContextCompressor private constructor(
    private val chunkIndex: Map<Long, ChunkUnits>,
    private val vecs: ShortBuffer,
    val dim: Int
) {
    class Unit(val item: String, val startsItem: Boolean, val tokFull: Int, val text: String, val lex: Set<String>) {
        /** == Compressor._unit_text */
        val fullText: String get() = if (!startsItem && item.isNotEmpty()) "(item $item) $text" else text
    }

    class ChunkUnits(val label: String, val prefix: String, val hdrTok: Int, val base: Int, val units: List<Unit>)

    data class Result(val text: String, val selected: Map<Long, List<Int>>, val usedEstimate: Int)

    companion object {
        const val LAMBDA = 0.3
        const val PRIOR = 0.05
        const val RATIO = 0.5

        // == compress._lex_tokens (stopwords do app; \w+ Unicode; deaccent; len>2)
        private val WORD_U = java.util.regex.Pattern.compile("\\w+", java.util.regex.Pattern.UNICODE_CHARACTER_CLASS)
        private val MARKS = Regex("\\p{Mn}+")

        fun deaccentLower(s: String): String =
            MARKS.replace(Normalizer.normalize(s, Normalizer.Form.NFKD), "").lowercase()

        fun lexStems(text: String): Set<String> {
            val out = LinkedHashSet<String>()
            val m = WORD_U.matcher(text)
            while (m.find()) {
                val t = deaccentLower(m.group())
                if (t.length > 2 && t !in Fts5Retriever.APP_STOPWORDS) out.add(if (t.length > 6) t.substring(0, 6) else t)
            }
            return out
        }

        fun load(unitsBin: File, vecsF16: File): ContextCompressor {
            val idx = LinkedHashMap<Long, ChunkUnits>()
            var row = 0
            DataInputStream(unitsBin.inputStream().buffered(1 shl 16)).use { d ->
                val magic = ByteArray(5); d.readFully(magic)
                check(String(magic) == "UNIT1") { "units.bin inválido" }
                val nChunks = le32(d)
                repeat(nChunks) {
                    val cid = le64(d)
                    val label = str(d)
                    val prefix = str(d)
                    val hdr = le32(d)
                    val n = le32(d)
                    val base = row
                    val units = ArrayList<Unit>(n)
                    repeat(n) {
                        val item = str(d)
                        val starts = d.readByte().toInt() != 0
                        val tok = le32(d)
                        val text = str(d)
                        val lex = str(d)
                        units.add(Unit(item, starts, tok, text,
                            if (lex.isEmpty()) emptySet() else lex.split(' ').toHashSet()))
                        row++
                    }
                    idx[cid] = ChunkUnits(label, prefix, hdr, base, units)
                }
            }
            val raf = RandomAccessFile(vecsF16, "r")
            val ch = raf.channel
            val head = ByteBuffer.allocate(13).order(ByteOrder.LITTLE_ENDIAN)
            ch.read(head, 0)
            head.flip()
            val magic = ByteArray(5); head.get(magic)
            check(String(magic) == "HVEC1") { "units_win3.f16 inválido" }
            val dim = head.int
            val count = head.int
            check(count == row) { "janelas ($count) != unidades ($row)" }
            val mm = ch.map(FileChannel.MapMode.READ_ONLY, 13, count.toLong() * dim * 2)
                .order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            return ContextCompressor(idx, mm, dim)
        }

        private fun le32(d: DataInputStream): Int {
            val b = ByteArray(4); d.readFully(b)
            return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).int
        }

        private fun le64(d: DataInputStream): Long {
            val b = ByteArray(8); d.readFully(b)
            return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).long
        }

        private fun str(d: DataInputStream): String {
            val n = le32(d)
            val b = ByteArray(n); d.readFully(b)
            return String(b, Charsets.UTF_8)
        }

        /** IEEE-754 half -> float (sem depender de API 26+ Half). */
        fun halfToFloat(h: Short): Float {
            val bits = h.toInt() and 0xFFFF
            val sign = (bits ushr 15) and 1
            val exp = (bits ushr 10) and 0x1F
            val mant = bits and 0x3FF
            val f = when {
                exp == 0 -> if (mant == 0) 0f else (mant / 1024f) * (1f / 16384f)
                exp == 31 -> if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                else -> java.lang.Float.intBitsToFloat(((exp - 15 + 127) shl 23) or (mant shl 13))
            }
            return if (sign == 1) -f else f
        }
    }

    fun has(cid: Long): Boolean = chunkIndex.containsKey(cid)

    /** cos(query, janela) de cada unidade do chunk (float32, como o espelho). */
    fun denseScores(qv: FloatArray, cid: Long): FloatArray {
        val cu = chunkIndex[cid] ?: return FloatArray(0)
        val out = FloatArray(cu.units.size)
        for (j in cu.units.indices) {
            var s = 0f
            val base = (cu.base + j) * dim
            for (d in 0 until dim) s += halfToFloat(vecs.get(base + d)) * qv[d]
            out[j] = s
        }
        return out
    }

    /** == Compressor.lex: |radicais(q) ∩ radicais(janela-3)| / |radicais(q)|. */
    fun lexScores(qStems: Set<String>, cid: Long): DoubleArray {
        val us = chunkIndex[cid]?.units ?: return DoubleArray(0)
        if (qStems.isEmpty()) return DoubleArray(us.size)
        return DoubleArray(us.size) { i ->
            val w = HashSet<String>()
            for (k in maxOf(0, i - 1) until minOf(us.size, i + 2)) w.addAll(us[k].lex)
            qStems.count { it in w }.toDouble() / qStems.size
        }
    }

    /** == Compressor.render_raw (top-2 cru: base do orçamento). */
    fun renderRaw(chunks: List<Chunk>): String =
        chunks.mapIndexed { i, c -> "[${i + 1}] (${chunkIndex[c.id]?.label ?: labelOf(c)}):\n${c.content}" }
            .joinToString("\n\n")

    private fun labelOf(c: Chunk): String =
        if (c.title.isNotEmpty()) "${c.doc} - ${c.section} - ${c.title}" else "${c.doc} - ${c.section}"

    /** == Compressor.render_selected */
    fun renderSelected(sel: Map<Long, List<Int>>, order: List<Long>): String {
        val parts = ArrayList<String>()
        var i = 0
        for (cid in order) {
            val js = sel[cid]?.sorted() ?: continue
            if (js.isEmpty()) continue
            i++
            val cu = chunkIndex.getValue(cid)
            val pieces = ArrayList<String>()
            var prev: Int? = null
            for (j in js) {
                if (prev != null && j != prev + 1) pieces.add("[…]")
                pieces.add(if (prev == null || j != prev + 1) cu.units[j].fullText else cu.units[j].text)
                prev = j
            }
            parts.add("[$i] (${cu.label}):\n${cu.prefix} — ${pieces.joinToString(" ")}")
        }
        return parts.joinToString("\n\n")
    }

    /**
     * Seleção gulosa (== Compressor.compress mode=hybrid). `countTokens` confere o render final
     * (laço de corte do espelho; nunca disparou nos 344 do recorte). Chunks fora do índice são
     * ignorados (não deve ocorrer: o índice cobre todas as normas e manuais).
     */
    fun compress(
        qv: FloatArray,
        question: String,
        cids: List<Long>,
        budget: Int,
        countTokens: ((String) -> Int)? = null
    ): Result {
        val qStems = lexStems(question)
        data class Cand(val s: Double, val r: Int, val cid: Long, val j: Int)
        val cand = ArrayList<Cand>()
        val valid = cids.filter { has(it) }
        valid.forEachIndexed { r, cid ->
            val ds = denseScores(qv, cid)
            val ls = lexScores(qStems, cid)
            for (j in ds.indices) cand.add(Cand(ds[j].toDouble() + LAMBDA * ls[j] + PRIOR / (r + 1), r, cid, j))
        }
        if (cand.isEmpty()) return Result("", emptyMap(), 0)
        val sorted = cand.sortedByDescending { it.s }  // estável: empates na ordem (rank, unidade)
        val sel = LinkedHashMap<Long, MutableList<Int>>()
        val picked = ArrayList<Cand>()
        var used = 0
        for (c in sorted) {
            val cu = chunkIndex.getValue(c.cid)
            val cost = cu.units[c.j].tokFull + (if (c.cid !in sel) cu.hdrTok else 0)
            if (used + cost > budget) continue
            sel.getOrPut(c.cid) { ArrayList() }.add(c.j)
            used += cost
            picked.add(c)
        }
        if (sel.isEmpty()) {
            val c = sorted.first()
            sel[c.cid] = mutableListOf(c.j)
            picked.add(c)
        }
        var text = renderSelected(sel, valid)
        if (countTokens != null) {
            while (countTokens(text) > budget && picked.size > 1) {
                val c = picked.removeAt(picked.size - 1)
                sel[c.cid]?.remove(c.j)
                if (sel[c.cid].isNullOrEmpty()) sel.remove(c.cid)
                text = renderSelected(sel, valid)
            }
        }
        return Result(text, sel, used)
    }
}
