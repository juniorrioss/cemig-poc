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
import androidx.compose.material.icons.filled.Bookmark
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Verified
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue100
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue500
import br.org.ceia.cemigpoc.ui.theme.CeiaGray200
import br.org.ceia.cemigpoc.ui.theme.CeiaGray400
import br.org.ceia.cemigpoc.ui.theme.CeiaGray50
import br.org.ceia.cemigpoc.ui.theme.CeiaGray700
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy800
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy950
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusMd
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusSm
import br.org.ceia.cemigpoc.ui.theme.CeiaTeal100
import br.org.ceia.cemigpoc.ui.theme.CeiaTeal600
import br.org.ceia.cemigpoc.ui.theme.CeiaWhite
import br.org.ceia.cemigpoc.ui.theme.InterFontFamily

/**
 * Seção de citação de fontes obrigatórias (Norma + Seção expansível).
 * Requisito estrito do Capitão: "citação de fonte obrigatória em cada resposta".
 */
@Composable
fun SourcesSection(
    chunks: List<Chunk>,
    modifier: Modifier = Modifier
) {
    // Controla estado de expansão de cada chunk individual
    val expandedStates = remember { mutableStateMapOf<Long, Boolean>() }

    Column(modifier = modifier.fillMaxWidth()) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(bottom = 8.dp)
        ) {
            Icon(
                imageVector = Icons.Filled.Verified,
                contentDescription = null,
                tint = CeiaTeal600,
                modifier = Modifier.size(18.dp)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = "FONTES NORMATIVAS CONSULTADAS (OBRIGATÓRIO)",
                color = CeiaNavy950,
                fontFamily = InterFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 12.sp,
                letterSpacing = 0.5.sp
            )
            Spacer(modifier = Modifier.width(6.dp))
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(CeiaRadiusSm))
                    .background(if (chunks.isNotEmpty()) CeiaTeal100 else CeiaGray200)
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            ) {
                Text(
                    text = "${chunks.size} doc(s)",
                    color = if (chunks.isNotEmpty()) CeiaTeal600 else CeiaGray700,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 11.sp
                )
            }
        }

        if (chunks.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(CeiaRadiusMd))
                    .background(CeiaWhite)
                    .border(1.dp, CeiaGray200, RoundedCornerShape(CeiaRadiusMd))
                    .padding(12.dp)
            ) {
                Text(
                    text = "Nenhuma fonte externa consultada para a mensagem atual (resposta direta do assistente).",
                    color = CeiaGray400,
                    fontFamily = InterFontFamily,
                    fontSize = 12.sp
                )
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                chunks.forEach { chunk ->
                    val isExpanded = expandedStates[chunk.id] ?: false
                    SourceItemCard(
                        chunk = chunk,
                        isExpanded = isExpanded,
                        onToggle = {
                            expandedStates[chunk.id] = !isExpanded
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun SourceItemCard(
    chunk: Chunk,
    isExpanded: Boolean,
    onToggle: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(CeiaRadiusMd))
            .background(CeiaWhite)
            .border(1.dp, if (isExpanded) CeiaBlue500 else CeiaGray200, RoundedCornerShape(CeiaRadiusMd))
            .clickable { onToggle() }
            .padding(12.dp)
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(CeiaRadiusSm))
                        .background(CeiaBlue100)
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = chunk.doc,
                        color = CeiaNavy800,
                        fontFamily = InterFontFamily,
                        fontWeight = FontWeight.Bold,
                        fontSize = 11.sp
                    )
                }

                Spacer(modifier = Modifier.width(8.dp))

                Text(
                    text = chunk.section,
                    color = CeiaNavy950,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 13.sp,
                    modifier = Modifier.weight(1f)
                )

                Icon(
                    imageVector = if (isExpanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                    contentDescription = if (isExpanded) "Recolher" else "Expandir",
                    tint = CeiaBlue500,
                    modifier = Modifier.size(20.dp)
                )
            }

            AnimatedVisibility(
                visible = isExpanded,
                enter = fadeIn() + expandVertically(),
                exit = fadeOut() + shrinkVertically()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 10.dp)
                        .clip(RoundedCornerShape(CeiaRadiusSm))
                        .background(CeiaGray50)
                        .padding(10.dp)
                ) {
                    Text(
                        text = chunk.content,
                        color = CeiaGray700,
                        fontFamily = InterFontFamily,
                        fontSize = 13.sp,
                        lineHeight = 19.sp
                    )

                    Spacer(modifier = Modifier.height(6.dp))

                    Text(
                        text = "Score BM25: ${"%.4f".format(chunk.score)}",
                        color = CeiaGray400,
                        fontFamily = InterFontFamily,
                        fontSize = 11.sp
                    )
                }
            }
        }
    }
}
