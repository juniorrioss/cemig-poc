package br.org.ceia.cemigpoc.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
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
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicNone
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.org.ceia.cemigpoc.ui.theme.ActayWideFontFamily
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue500
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue600
import br.org.ceia.cemigpoc.ui.theme.CeiaDanger
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy950
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusLg
import br.org.ceia.cemigpoc.ui.theme.CeiaWhite
import br.org.ceia.cemigpoc.ui.theme.InterFontFamily

/**
 * Botão Push-To-Talk gigante projetado especificamente para operários de campo (uso com luvas).
 *
 * Interação:
 * - Ao tocar e segurar: inicia captura de voz no ASR.
 * - Ao soltar: envia transcrição final para o pipeline.
 */
@Composable
fun PushToTalkButton(
    isListening: Boolean,
    onStartPress: () -> Unit,
    onRelease: () -> Unit,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition(label = "ptt_pulse")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1.0f,
        targetValue = 1.04f,
        animationSpec = infiniteRepeatable(
            animation = tween(600),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse_scale"
    )

    val currentScale = if (isListening) pulseScale else 1.0f

    val backgroundColor by animateColorAsState(
        targetValue = if (isListening) CeiaDanger else CeiaBlue500,
        animationSpec = tween(150),
        label = "ptt_color"
    )

    Box(
        modifier = modifier
            .fillMaxWidth()
            .scale(currentScale)
            .shadow(
                elevation = if (isListening) 12.dp else 4.dp,
                shape = RoundedCornerShape(CeiaRadiusLg)
            )
            .clip(RoundedCornerShape(CeiaRadiusLg))
            .background(backgroundColor)
            .pointerInput(Unit) {
                detectTapGestures(
                    onPress = {
                        onStartPress()
                        tryAwaitRelease()
                        onRelease()
                    }
                )
            }
            .padding(vertical = 20.dp, horizontal = 16.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            Icon(
                imageVector = if (isListening) Icons.Filled.Mic else Icons.Filled.MicNone,
                contentDescription = "Microfone Push-To-Talk",
                tint = CeiaWhite,
                modifier = Modifier.size(36.dp)
            )

            Spacer(modifier = Modifier.width(14.dp))

            Column {
                Text(
                    text = if (isListening) "SOLTE PARA ENVIAR" else "SEGURE PARA FALAR",
                    color = CeiaWhite,
                    fontFamily = ActayWideFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 17.sp,
                    letterSpacing = 0.5.sp
                )
                Text(
                    text = if (isListening) "Gravando áudio do operador..." else "Assistente por voz 100% offline",
                    color = CeiaWhite.copy(alpha = 0.85f),
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Normal,
                    fontSize = 12.sp
                )
            }
        }
    }
}
