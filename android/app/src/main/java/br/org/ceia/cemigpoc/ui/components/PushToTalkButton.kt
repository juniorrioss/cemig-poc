package br.org.ceia.cemigpoc.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import br.org.ceia.cemigpoc.ui.theme.CemigDestructive
import br.org.ceia.cemigpoc.ui.theme.CemigPrimary
import br.org.ceia.cemigpoc.ui.theme.CemigPrimaryDisabled

/**
 * Botão principal CONTEXTUAL (ajuste de UI 4, pedido do capitão), no formato circular do
 * botão de envio do app de referência (F8).
 *
 * - Campo de texto VAZIO -> MICROFONE: segurar para falar (Push-To-Talk), soltar para
 *   transcrever. Fica grande (56dp) para uso com luvas de eletricista.
 * - Campo COM conteúdo (transcrito ou digitado) -> ENVIAR: um TOQUE envia ao pipeline.
 *   Limpar o campo volta ao microfone (permite regravar).
 * - Pipeline ocupado -> spinner no lugar da seta (como o app de referência); envio bloqueado.
 *
 * Gravando: vermelho com pulso. O modo enviar usa clique simples (não gesto de segurar),
 * evitando envio acidental por toque de gravação.
 */
@Composable
fun PushToTalkButton(
    isListening: Boolean,
    hasText: Boolean,
    sendEnabled: Boolean,
    onStartPress: () -> Unit,
    onRelease: () -> Unit,
    onSend: () -> Unit,
    modifier: Modifier = Modifier,
    busy: Boolean = false
) {
    val infiniteTransition = rememberInfiniteTransition(label = "ptt_pulse")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1.0f,
        targetValue = 1.12f,
        animationSpec = infiniteRepeatable(
            animation = tween(600),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse_scale"
    )

    // Modo ENVIAR quando há texto e não estamos gravando; senão modo MICROFONE.
    val sendMode = hasText && !isListening
    val currentScale = if (isListening) pulseScale else 1.0f

    val backgroundColor by animateColorAsState(
        targetValue = when {
            isListening -> CemigDestructive
            // Cores sólidas (alpha + shadow gera artefato poligonal no Android)
            sendMode && !sendEnabled -> CemigPrimaryDisabled
            busy -> CemigPrimaryDisabled
            else -> CemigPrimary
        },
        animationSpec = tween(150),
        label = "ptt_color"
    )

    // Modificador de interação: TOQUE (enviar) ou SEGURAR (gravar), conforme o modo.
    val interactionModifier = if (sendMode) {
        Modifier.clickable(enabled = sendEnabled) { onSend() }
    } else {
        Modifier.pointerInput(Unit) {
            detectTapGestures(
                onPress = {
                    onStartPress()
                    tryAwaitRelease()
                    onRelease()
                }
            )
        }
    }

    Box(
        modifier = modifier
            .size(56.dp)
            .scale(currentScale)
            .shadow(elevation = if (isListening) 10.dp else if (busy) 0.dp else 2.dp, shape = CircleShape)
            .clip(CircleShape)
            .background(backgroundColor)
            .then(interactionModifier)
            .semantics {
                contentDescription = when {
                    isListening -> "Gravando. Solte para transcrever"
                    sendMode -> "Enviar pergunta"
                    else -> "Microfone Push-To-Talk: segure para falar"
                }
            },
        contentAlignment = Alignment.Center
    ) {
        if (busy && sendMode) {
            CircularProgressIndicator(modifier = Modifier.size(22.dp), color = Color.White, strokeWidth = 2.5.dp)
        } else {
            Icon(
                imageVector = when {
                    isListening -> Icons.Filled.Mic
                    sendMode -> Icons.Filled.ArrowUpward
                    else -> Icons.Filled.Mic
                },
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(if (sendMode) 26.dp else 28.dp)
            )
        }
    }
}
