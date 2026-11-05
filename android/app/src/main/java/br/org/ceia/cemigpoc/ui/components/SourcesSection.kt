package br.org.ceia.cemigpoc.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.ui.theme.CemigBorder
import br.org.ceia.cemigpoc.ui.theme.CemigCard
import br.org.ceia.cemigpoc.ui.theme.CemigForeground
import br.org.ceia.cemigpoc.ui.theme.CemigManual
import br.org.ceia.cemigpoc.ui.theme.CemigMuted
import br.org.ceia.cemigpoc.ui.theme.CemigMutedFg
import br.org.ceia.cemigpoc.ui.theme.CemigPrimary
import br.org.ceia.cemigpoc.ui.theme.CemigSecondary
import br.org.ceia.cemigpoc.ui.theme.CemigSecondaryFg
import br.org.ceia.cemigpoc.ui.theme.InterFontFamily

/**
 * Citação de fontes obrigatória em cada resposta (requisito do capitão): selo da norma/manual
 * + item/seção, cada trecho expansível com o texto integral.
 *
 * F8: visual do cartão "Fontes" do app de referência (fundo muted, ícone de livro, linhas
 * compactas), mantendo a expansão por trecho da nossa UI.
 *
 * @param reused true quando o modelo NÃO buscou (reuso do contexto / resposta direta).
 */
@Composable
fun SourcesSection(
    chunks: List<Chunk>,
    modifier: Modifier = Modifier,
    reused: Boolean = false
) {
    // Controla estado de expansão de cada chunk individual
    val expandedStates = remember { mutableStateMapOf<Long, Boolean>() }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(CemigMuted)
            .border(1.dp, CemigBorder, RoundedCornerShape(16.dp))
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.MenuBook,
                contentDescription = null,
                tint = CemigPrimary,
                modifier = Modifier.size(14.dp)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = if (chunks.isEmpty()) "Fontes" else "Fontes consultadas · ${chunks.size}",
                color = CemigMutedFg,
                fontFamily = InterFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 12.sp
            )
        }

        if (chunks.isEmpty()) {
            Text(
                text = if (reused) "Sem nova busca: o modelo respondeu com o contexto já consultado na conversa."
                else "Nenhuma fonte consultada para esta mensagem (resposta direta do assistente).",
                color = CemigMutedFg,
                fontFamily = InterFontFamily,
                fontSize = 11.sp,
                lineHeight = 16.sp
            )
        } else {
            chunks.forEachIndexed { idx, chunk ->
                val isExpanded = expandedStates[chunk.id] ?: false
                SourceItem(
                    index = idx + 1,
                    chunk = chunk,
                    isExpanded = isExpanded,
                    onToggle = { expandedStates[chunk.id] = !isExpanded }
                )
            }
        }
    }
}

/** Trechos de manual vêm do manuals.db com section == "manual" (doc = NR associada). */
private fun isManual(chunk: Chunk): Boolean = chunk.section.equals("manual", ignoreCase = true)

@Composable
private fun SourceItem(
    index: Int,
    chunk: Chunk,
    isExpanded: Boolean,
    onToggle: () -> Unit
) {
    val manual = isManual(chunk)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(if (isExpanded) CemigCard else CemigMuted)
            .border(1.dp, if (isExpanded) CemigPrimary.copy(alpha = 0.4f) else CemigMuted, RoundedCornerShape(12.dp))
            .clickable { onToggle() }
            .padding(horizontal = 6.dp, vertical = 5.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = "[$index]",
                color = CemigMutedFg,
                fontFamily = InterFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 11.sp
            )
            Spacer(modifier = Modifier.width(6.dp))
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(999.dp))
                    .background(if (manual) CemigManual.copy(alpha = 0.12f) else CemigSecondary)
                    .padding(horizontal = 8.dp, vertical = 2.dp)
            ) {
                Text(
                    text = (if (manual) "MANUAL " else "") + chunk.doc.uppercase(),
                    color = if (manual) CemigManual else CemigSecondaryFg,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 11.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                // Norma: "item 10.2.1"; manual: título do guia.
                text = if (manual) chunk.title.ifBlank { "Manual comentado" }
                else chunk.section.let { if (it.isNotBlank() && it.first().isDigit()) "item $it" else it.ifBlank { chunk.title } },
                color = CemigForeground,
                fontFamily = InterFontFamily,
                fontWeight = FontWeight.SemiBold,
                fontSize = 12.sp,
                maxLines = if (isExpanded) 3 else 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
            Icon(
                imageVector = if (isExpanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                contentDescription = if (isExpanded) "Recolher trecho" else "Ver trecho",
                tint = CemigMutedFg,
                modifier = Modifier.size(18.dp)
            )
        }

        AnimatedVisibility(
            visible = isExpanded,
            enter = fadeIn() + expandVertically(),
            exit = fadeOut() + shrinkVertically()
        ) {
            Column(modifier = Modifier.fillMaxWidth().padding(start = 4.dp, end = 4.dp, top = 8.dp, bottom = 4.dp)) {
                Text(
                    text = chunk.content,
                    color = CemigForeground,
                    fontFamily = InterFontFamily,
                    fontSize = 13.sp,
                    lineHeight = 19.sp
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Score: ${"%.4f".format(chunk.score)}" + if (chunk.page > 0) " · pág. ${chunk.page}" else "",
                    color = CemigMutedFg,
                    fontFamily = InterFontFamily,
                    fontSize = 11.sp
                )
            }
        }
    }
}
