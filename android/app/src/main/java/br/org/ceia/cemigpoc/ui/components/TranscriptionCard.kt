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
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue500
import br.org.ceia.cemigpoc.ui.theme.CeiaGray200
import br.org.ceia.cemigpoc.ui.theme.CeiaGray500
import br.org.ceia.cemigpoc.ui.theme.CeiaGray900
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy800
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusMd
import br.org.ceia.cemigpoc.ui.theme.CeiaWhite
import br.org.ceia.cemigpoc.ui.theme.InterFontFamily

/**
 * Card de conferência da transcrição de fala do operário.
 */
@Composable
fun TranscriptionCard(
    transcription: String,
    isListening: Boolean,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(CeiaRadiusMd))
            .background(CeiaWhite)
            .border(1.dp, CeiaGray200, RoundedCornerShape(CeiaRadiusMd))
            .padding(14.dp)
    ) {
        Column {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Filled.RecordVoiceOver,
                    contentDescription = null,
                    tint = CeiaBlue500,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "TRANSCRIÇÃO DE VOZ (CONFERÊNCIA)",
                    color = CeiaNavy800,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp,
                    letterSpacing = 0.5.sp
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            if (transcription.isNotBlank()) {
                Text(
                    text = "\"$transcription\"",
                    color = CeiaGray900,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Medium,
                    fontSize = 15.sp,
                    lineHeight = 22.sp,
                    fontStyle = if (isListening) FontStyle.Italic else FontStyle.Normal
                )
            } else {
                Text(
                    text = if (isListening) "Ouvindo... Fale agora." else "Nenhuma instrução gravada ainda. Segure o botão abaixo para perguntar.",
                    color = CeiaGray500,
                    fontFamily = InterFontFamily,
                    fontStyle = FontStyle.Italic,
                    fontSize = 13.sp
                )
            }
        }
    }
}
