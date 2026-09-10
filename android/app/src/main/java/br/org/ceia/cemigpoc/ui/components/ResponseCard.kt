package br.org.ceia.cemigpoc.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.ElectricBolt
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.org.ceia.cemigpoc.domain.model.ToolCall
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue100
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue500
import br.org.ceia.cemigpoc.ui.theme.CeiaGray200
import br.org.ceia.cemigpoc.ui.theme.CeiaGray500
import br.org.ceia.cemigpoc.ui.theme.CeiaGray900
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy950
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusMd
import br.org.ceia.cemigpoc.ui.theme.CeiaSuccess
import br.org.ceia.cemigpoc.ui.theme.CeiaWhite
import br.org.ceia.cemigpoc.ui.theme.InterFontFamily

/**
 * Card para exibição de resposta do modelo SLM com streaming de texto em tempo real.
 */
@Composable
fun ResponseCard(
    responseText: String,
    toolCall: ToolCall?,
    totalDurationMs: Long?,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(CeiaRadiusMd))
            .background(CeiaWhite)
            .border(1.dp, CeiaGray200, RoundedCornerShape(CeiaRadiusMd))
            .padding(16.dp)
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Filled.ElectricBolt,
                    contentDescription = null,
                    tint = CeiaBlue500,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "RESPOSTA DAS NORMAS TÉCNICAS",
                    color = CeiaNavy950,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 13.sp,
                    letterSpacing = 0.5.sp
                )

                Spacer(modifier = Modifier.weight(1f))

                if (totalDurationMs != null) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Filled.Timer,
                            contentDescription = null,
                            tint = CeiaSuccess,
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "${totalDurationMs}ms (≤10s)",
                            color = CeiaSuccess,
                            fontFamily = InterFontFamily,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 11.sp
                        )
                    }
                }
            }

            // Exibe indicador se o LLM acionou o retriever via tool-calling
            if (toolCall != null) {
                Spacer(modifier = Modifier.height(10.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(6.dp))
                        .background(CeiaBlue100.copy(alpha = 0.5f))
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Filled.Build,
                            contentDescription = null,
                            tint = CeiaBlue500,
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "Tool executada: ${toolCall.toolName}(\"${toolCall.query}\")",
                            color = CeiaNavy950,
                            fontFamily = InterFontFamily,
                            fontWeight = FontWeight.Medium,
                            fontSize = 11.sp
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            if (responseText.isNotBlank()) {
                Text(
                    text = responseText,
                    color = CeiaGray900,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Normal,
                    fontSize = 15.sp,
                    lineHeight = 24.sp
                )
            } else {
                Text(
                    text = "Aguardando pergunta. A resposta será apresentada aqui estritamente embasada nas Normas Regulamentadoras com citação de fonte.",
                    color = CeiaGray500,
                    fontFamily = InterFontFamily,
                    fontSize = 13.sp,
                    lineHeight = 18.sp
                )
            }
        }
    }
}
