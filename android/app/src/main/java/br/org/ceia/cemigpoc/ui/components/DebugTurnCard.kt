package br.org.ceia.cemigpoc.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.TurnMetrics
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue500
import br.org.ceia.cemigpoc.ui.theme.CeiaGray100
import br.org.ceia.cemigpoc.ui.theme.CeiaGray200
import br.org.ceia.cemigpoc.ui.theme.CeiaGray700
import br.org.ceia.cemigpoc.ui.theme.CeiaGray900
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy950
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusMd
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusSm
import br.org.ceia.cemigpoc.ui.theme.CeiaSuccess
import br.org.ceia.cemigpoc.ui.theme.CeiaWhite

private val TimelineAsrColor = Color(0xFF3B82F6)      // Azul
private val TimelineSearchColor = Color(0xFF14B8A6)   // Teal (classificação + BM25)
private val TimelineTtftColor = Color(0xFFF59E0B)     // Âmbar
private val TimelineDecodeColor = Color(0xFF10B981)   // Verde

/**
 * Card de diagnóstico e telemetria inline para o Modo Engenharia / Debug solicitado pelo Capitão.
 *
 * Exibe breakdown minucioso do pipeline consolidado (sem Turno 1 de rewrite):
 * (1) Transcrição ASR + tempo
 * (2) Classificação NR + gate acionado (top-1/top-2, probs, modo do gate, NRs boostadas)
 * (3) Chunks recuperados (doc+seção+score BM25) + tempo busca
 * (4) Tokens de contexto, TTFT, tok/s decode + tempo total
 * (5) Barra visual proporcional do tempo total da pergunta
 */
@Composable
fun DebugTurnCard(
    metrics: TurnMetrics,
    chunks: List<Chunk>,
    transcription: String,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(CeiaRadiusMd))
            .background(Color(0xFF0F172A)) // Slate escuro técnico
            .border(1.dp, Color(0xFF334155), RoundedCornerShape(CeiaRadiusMd))
            .padding(12.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            // Cabeçalho técnico
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "TELEMETRIA DE ENGENHARIA",
                    color = Color(0xFF94A3B8),
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 0.8.sp
                )
                Text(
                    text = "TOTAL: ${metrics.totalMs} ms (%.2fs)".format(metrics.totalMs / 1000.0),
                    color = Color(0xFF38BDF8),
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace
                )
            }

            // Barra visual de timeline proporcional com breakdown das etapas
            TimelineBar(metrics = metrics)

            // Legenda das cores da timeline
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                TimelineLegendItem("ASR", TimelineAsrColor)
                TimelineLegendItem("Classif+BM25", TimelineSearchColor)
                TimelineLegendItem("TTFT", TimelineTtftColor)
                TimelineLegendItem("Decode", TimelineDecodeColor)
            }

            Spacer(modifier = Modifier.height(2.dp))

            // (1) ASR + Tempo
            if (metrics.asrMs > 0 || transcription.isNotBlank()) {
                DebugRow(
                    label = "1. ASR (Whisper)",
                    time = "${metrics.asrMs} ms",
                    detail = if (transcription.isNotBlank()) "\"$transcription\"" else "Áudio sintetizado / digitado"
                )
            }

            // (2) Classificação NR + gate acionado (o capítão quer ver a decisão do estágio 1)
            val gateLabel = when (metrics.gateMode) {
                "hard" -> "FILTRO-DURO (prob ≥ 0.5)"
                "soft" -> "BOOST-SUAVE 5x (top-2)"
                "explicit" -> "NR EXPLÍCITA NA FALA"
                "none" -> "SEM BOOST (fora de escopo)"
                "reuso" -> "REUSO JACCARD > 0.7 (sem nova classificação)"
                else -> metrics.gateMode.ifBlank { "-" }
            }
            val classDetail = if (metrics.gateMode == "reuso") {
                "Chunks reaproveitados do turno anterior"
            } else {
                "top1=${metrics.nrTop1} (${"%.2f".format(metrics.nrTop1Prob)}) · top2=${metrics.nrTop2}" +
                    if (metrics.boostNrs.isNotBlank()) " · boost=[${metrics.boostNrs}]" else ""
            }
            DebugRow(
                label = "2. Classificador NR + Gate",
                time = "[$gateLabel]",
                detail = classDetail
            )

            // (3) Recuperação: Retrieval v3 (fusão RRF 3-sinais) ou BM25-gated legado.
            val isRrf3 = metrics.retrievalMode == "rrf3"
            val retrievalTitle = when (metrics.retrievalMode) {
                "rrf3" -> "3. Fusão RRF 3-sinais (BM25+Denso×2)"
                "rrf3-fallback-bm25" -> "3. BM25-gated (denso indisponível)"
                else -> "3. Busca BM25 Top-2 (fala bruta)"
            }
            val chunksSummary = if (chunks.isNotEmpty()) {
                if (isRrf3) {
                    // Mostra os ranks por sinal e o score RRF (o capitão quer ver a fusão).
                    chunks.joinToString("  |  ") { c ->
                        val rb = if (c.rankBm25 > 0) "bm25#${c.rankBm25}" else "bm25–"
                        val rt = if (c.rankDenseText > 0) "dTxt#${c.rankDenseText}" else "dTxt–"
                        val re = if (c.rankDenseExp > 0) "dExp#${c.rankDenseExp}" else "dExp–"
                        "${c.doc} ${c.section} [$rb $rt $re rrf=${"%.4f".format(c.rrfScore)}]"
                    }
                } else {
                    chunks.joinToString(", ") { "${it.doc} ${it.section} (score: ${"%.2f".format(it.score)})" }
                }
            } else {
                "Nenhum chunk recuperado"
            }
            val retrievalTime = when {
                metrics.chunksReused -> "0 ms (reuso)"
                isRrf3 && metrics.denseEncodeMs > 0 -> "${metrics.searchMs} ms (inclui encode denso)"
                else -> "${metrics.searchMs} ms"
            }
            DebugRow(
                label = retrievalTitle,
                time = retrievalTime,
                detail = chunksSummary
            )

            // (4) Tokens de contexto, TTFT, velocidade decode e total
            DebugRow(
                label = "4. Síntese (LFM2.5)",
                time = "${metrics.ttftMs + metrics.decodeMs} ms",
                detail = "Contexto: ~${metrics.contextTokens} tok · TTFT: ${metrics.ttftMs} ms · Decode: ${metrics.decodeMs} ms (${metrics.completionTokens} tok @ ${metrics.tokPerSec} t/s)"
            )
        }
    }
}

@Composable
private fun TimelineBar(metrics: TurnMetrics) {
    val asr = maxOf(0L, metrics.asrMs)
    val search = maxOf(0L, metrics.searchMs)
    val ttft = maxOf(0L, metrics.ttftMs)
    val decode = maxOf(0L, metrics.decodeMs)

    val sum = asr + search + ttft + decode
    val total = if (sum > 0) sum.toFloat() else 1.0f

    val wAsr = asr.toFloat() / total
    val wSearch = search.toFloat() / total
    val wTtft = ttft.toFloat() / total
    val wDecode = decode.toFloat() / total

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(8.dp)
            .clip(RoundedCornerShape(4.dp))
            .background(Color(0xFF1E293B))
    ) {
        if (wAsr > 0f) Box(modifier = Modifier.weight(wAsr).height(8.dp).background(TimelineAsrColor))
        if (wSearch > 0f) Box(modifier = Modifier.weight(wSearch).height(8.dp).background(TimelineSearchColor))
        if (wTtft > 0f) Box(modifier = Modifier.weight(wTtft).height(8.dp).background(TimelineTtftColor))
        if (wDecode > 0f) Box(modifier = Modifier.weight(wDecode).height(8.dp).background(TimelineDecodeColor))
    }
}

@Composable
private fun TimelineLegendItem(label: String, color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .width(8.dp)
                .height(8.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(color)
        )
        Spacer(modifier = Modifier.width(3.dp))
        Text(
            text = label,
            color = Color(0xFF94A3B8),
            fontSize = 9.sp,
            fontFamily = FontFamily.Monospace
        )
    }
}

@Composable
private fun DebugRow(
    label: String,
    time: String,
    detail: String
) {
    Column(verticalArrangement = Arrangement.spacedBy(1.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = label,
                color = Color(0xFFCBD5E1),
                fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = time,
                color = Color(0xFFFCD34D),
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Medium
            )
        }
        Text(
            text = detail,
            color = Color(0xFF94A3B8),
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace,
            lineHeight = 13.sp
        )
    }
}
