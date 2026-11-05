package br.org.ceia.cemigpoc.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ElectricBolt
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.RestartAlt
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.org.ceia.cemigpoc.R
import br.org.ceia.cemigpoc.domain.model.Chunk
import br.org.ceia.cemigpoc.domain.model.ConversationTurn
import br.org.ceia.cemigpoc.domain.model.PipelineStage
import br.org.ceia.cemigpoc.domain.model.TurnMetrics
import br.org.ceia.cemigpoc.ui.components.DebugTurnCard
import br.org.ceia.cemigpoc.ui.components.PushToTalkButton
import br.org.ceia.cemigpoc.ui.components.SourcesSection
import br.org.ceia.cemigpoc.ui.theme.CemigBackground
import br.org.ceia.cemigpoc.ui.theme.CemigBorder
import br.org.ceia.cemigpoc.ui.theme.CemigCard
import br.org.ceia.cemigpoc.ui.theme.CemigDestructive
import br.org.ceia.cemigpoc.ui.theme.CemigDisabled
import br.org.ceia.cemigpoc.ui.theme.CemigErrorBg
import br.org.ceia.cemigpoc.ui.theme.CemigErrorBorder
import br.org.ceia.cemigpoc.ui.theme.CemigErrorText
import br.org.ceia.cemigpoc.ui.theme.CemigForeground
import br.org.ceia.cemigpoc.ui.theme.CemigManual
import br.org.ceia.cemigpoc.ui.theme.CemigMuted
import br.org.ceia.cemigpoc.ui.theme.CemigMutedFg
import br.org.ceia.cemigpoc.ui.theme.CemigPlaceholder
import br.org.ceia.cemigpoc.ui.theme.CemigPrimary
import br.org.ceia.cemigpoc.ui.theme.CemigSecondary
import br.org.ceia.cemigpoc.ui.theme.CemigSecondaryFg
import br.org.ceia.cemigpoc.ui.theme.CemigThinking
import br.org.ceia.cemigpoc.ui.theme.CemigThinkingBg
import br.org.ceia.cemigpoc.ui.theme.CemigThinkingBorder
import br.org.ceia.cemigpoc.ui.theme.CemigThinkingText
import br.org.ceia.cemigpoc.ui.theme.InterFontFamily
import kotlinx.coroutines.delay

/**
 * Interface conversacional multiturno de campo para operários do setor elétrico.
 *
 * F8: layout aproximado do app de referência "Assistente CEMIG" (github.com/vinyteles/cemig-mobile,
 * tela de chat em src/app/index.tsx) — cabeçalho com selo verde, bolhas de chat, bolha
 * "Pensando" durante a busca, rodapé de fontes/tempos e composer arredondado — RECRIADO em
 * Compose (nenhum arquivo copiado). Todas as funções da UI anterior foram mantidas: voz
 * push-to-talk com revisão da transcrição, perguntas-exemplo, citação de fontes expansível,
 * indicação de busca (tool-calling), streaming, Modo Engenharia e Nova Conversa.
 */
@Composable
fun MainScreen(
    viewModel: MainViewModel,
    modifier: Modifier = Modifier
) {
    val uiState by viewModel.uiState.collectAsState()
    val listState = rememberLazyListState()
    var confirmRestart by remember { mutableStateOf(false) }

    // Rola automaticamente para o fim da conversa ao receber novas mensagens / tokens
    LaunchedEffect(uiState.history.size, uiState.currentStreamingAnswer, uiState.stage, uiState.currentChunks.size) {
        val total = listState.layoutInfo.totalItemsCount
        if (total > 0) listState.animateScrollToItem(total - 1)
    }

    val pipelineBusy = uiState.stage == PipelineStage.CLASSIFYING ||
        uiState.stage == PipelineStage.RESPONDING
    val turnActive = uiState.currentStreamingAnswer.isNotEmpty() ||
        uiState.stage == PipelineStage.CLASSIFYING || uiState.stage == PipelineStage.RESPONDING

    Box(modifier = modifier.fillMaxSize().background(CemigCard)) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
        ) {
            ChatHeader(
                isDebugMode = uiState.isDebugMode,
                restartEnabled = (uiState.history.isNotEmpty() || uiState.currentQuestionInput.isNotEmpty()) && !pipelineBusy,
                onToggleDebug = { viewModel.toggleDebugMode() },
                onRestart = { confirmRestart = true }
            )
            StatusStrip(
                stage = uiState.stage,
                isModelReady = uiState.isModelReady,
                modelStatus = uiState.modelStatusMessage,
                isDebugMode = uiState.isDebugMode
            )

            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                if (!uiState.isModelReady) {
                    item { LoadingCard(message = uiState.modelStatusMessage) }
                }

                if (uiState.history.isEmpty() && !turnActive) {
                    item {
                        EmptyState(
                            examples = uiState.exampleQuestions,
                            enabled = uiState.isModelReady && !pipelineBusy,
                            onExample = { viewModel.processQuestion(it) }
                        )
                    }
                }

                items(uiState.history, key = { it.id }) { turn ->
                    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                        UserBubble(text = turn.question)
                        AssistantMessage(turn = turn, isDebugMode = uiState.isDebugMode)
                    }
                }

                if (turnActive) {
                    item {
                        ActiveTurn(
                            stage = uiState.stage,
                            toolCalled = uiState.toolCalled,
                            searchInProgress = uiState.searchInProgress,
                            streamingAnswer = uiState.currentStreamingAnswer,
                            chunks = uiState.currentChunks
                        )
                    }
                }

                if (uiState.stage == PipelineStage.ERROR) {
                    item { ErrorBanner(message = uiState.errorMessage ?: "Falha operacional") }
                }
            }

            Composer(
                input = uiState.currentQuestionInput,
                isListening = uiState.isListening,
                isTranscribing = uiState.stage == PipelineStage.TRANSCRIBING,
                awaitingConfirmation = uiState.awaitingConfirmation,
                pipelineBusy = pipelineBusy,
                onInputChange = { viewModel.onQuestionInputChanged(it) },
                onStartPress = { viewModel.startPushToTalk() },
                onRelease = { viewModel.stopPushToTalk() },
                onSend = { viewModel.submitQuestion() }
            )
        }

        SplashOverlay()
    }

    // Confirmação antes de apagar a conversa (como no app de referência).
    if (confirmRestart) {
        AlertDialog(
            onDismissRequest = { confirmRestart = false },
            title = { Text("Nova conversa?", fontFamily = InterFontFamily, fontWeight = FontWeight.Bold) },
            text = {
                Text(
                    "As mensagens desta conversa serão apagadas. Essa ação não pode ser desfeita.",
                    fontFamily = InterFontFamily
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    confirmRestart = false
                    viewModel.newConversation()
                }) { Text("Reiniciar", color = CemigDestructive, fontWeight = FontWeight.Bold) }
            },
            dismissButton = {
                TextButton(onClick = { confirmRestart = false }) {
                    Text("Cancelar", color = CemigForeground)
                }
            },
            containerColor = CemigCard,
            // O tema CEIA define extraLarge=999dp (pílula); o diálogo precisa de canto fixo.
            shape = RoundedCornerShape(24.dp)
        )
    }
}

// -------------------------------------------------------------------------
// Cabeçalho: selo verde com raio + título "Assistente CEMIG" + ações
// -------------------------------------------------------------------------
@Composable
private fun ChatHeader(
    isDebugMode: Boolean,
    restartEnabled: Boolean,
    onToggleDebug: () -> Unit,
    onRestart: () -> Unit
) {
    Column(modifier = Modifier.fillMaxWidth().background(CemigCard)) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            BrandTile(size = 44)
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "CEIA · 100% offline · LFM3",
                    color = CemigMutedFg,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 12.sp,
                    maxLines = 1
                )
                Text(
                    text = "Assistente CEMIG",
                    color = CemigForeground,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp,
                    letterSpacing = (-0.3).sp,
                    maxLines = 1
                )
            }
            // Modo Engenharia / Debug (ghost; preenchido em verde quando ativo)
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(if (isDebugMode) CemigPrimary else Color.Transparent)
                    .clickable(onClick = onToggleDebug)
                    .semantics { contentDescription = if (isDebugMode) "Desligar Modo Engenharia" else "Ligar Modo Engenharia" },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.BugReport,
                    contentDescription = null,
                    tint = if (isDebugMode) Color.White else CemigPrimary,
                    modifier = Modifier.size(21.dp)
                )
            }
            Spacer(modifier = Modifier.width(4.dp))
            // Nova conversa (outline; pede confirmação)
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(CemigCard)
                    .border(1.dp, CemigBorder, CircleShape)
                    .clickable(enabled = restartEnabled, onClick = onRestart)
                    .semantics { contentDescription = "Nova conversa" },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.RestartAlt,
                    contentDescription = null,
                    tint = if (restartEnabled) CemigPrimary else CemigDisabled,
                    modifier = Modifier.size(21.dp)
                )
            }
        }
        Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(CemigBorder))
    }
}

/** Selo da marca recriado do app de referência: quadrado verde arredondado com raio branco. */
@Composable
private fun BrandTile(size: Int) {
    Box(
        modifier = Modifier
            .size(size.dp)
            .shadow(2.dp, RoundedCornerShape((size * 0.36f).dp))
            .clip(RoundedCornerShape((size * 0.36f).dp))
            .background(CemigPrimary),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = Icons.Filled.ElectricBolt,
            contentDescription = null,
            tint = Color.White,
            modifier = Modifier.size((size * 0.5f).dp)
        )
    }
}

// -------------------------------------------------------------------------
// Faixa de status sob o cabeçalho (no lugar do monitor de recursos deles):
// etapa viva do pipeline — o operário vê que o app está trabalhando.
// -------------------------------------------------------------------------
@Composable
private fun StatusStrip(
    stage: PipelineStage,
    isModelReady: Boolean,
    modelStatus: String,
    isDebugMode: Boolean
) {
    val (dot, label) = when {
        !isModelReady -> CemigWarningDot to "Carregando modelos no aparelho…"
        else -> when (stage) {
            PipelineStage.IDLE -> CemigPrimary to "Pronto · 100% offline"
            PipelineStage.LISTENING -> CemigDestructive to "Gravando sua voz…"
            PipelineStage.TRANSCRIBING -> CemigThinking to "Transcrevendo (Nemotron 3.5)…"
            PipelineStage.CLASSIFYING -> CemigThinking to "Decidindo: buscar nas normas ou reusar…"
            PipelineStage.RESPONDING -> CemigThinking to "Gerando resposta (LFM3 1.7B)…"
            PipelineStage.DONE -> CemigPrimary to "Resposta concluída"
            PipelineStage.ERROR -> CemigDestructive to "Falha operacional"
        }
    }
    Column(modifier = Modifier.fillMaxWidth().background(CemigMuted)) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(modifier = Modifier.size(8.dp).clip(CircleShape).background(dot))
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = label,
                color = CemigMutedFg,
                fontFamily = InterFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 12.sp,
                modifier = Modifier.weight(1f),
                maxLines = 1
            )
            if (isDebugMode) {
                Text(
                    text = "MODO ENGENHARIA",
                    color = CemigPrimary,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 10.sp
                )
            } else if (isModelReady && stage == PipelineStage.IDLE) {
                Text(
                    text = modelStatus.removePrefix("Pronto ").trim('(', ')'),
                    color = CemigMutedFg,
                    fontFamily = InterFontFamily,
                    fontSize = 10.sp,
                    maxLines = 1
                )
            }
        }
        Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(CemigBorder))
    }
}

private val CemigWarningDot = Color(0xFFD97706)

@Composable
private fun LoadingCard(message: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(CemigSecondary)
            .padding(14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        CircularProgressIndicator(modifier = Modifier.size(18.dp), color = CemigPrimary, strokeWidth = 2.dp)
        Spacer(modifier = Modifier.width(10.dp))
        Text(
            text = message,
            color = CemigSecondaryFg,
            fontFamily = InterFontFamily,
            fontWeight = FontWeight.SemiBold,
            fontSize = 13.sp
        )
    }
}

// -------------------------------------------------------------------------
// Estado vazio: cartão explicativo + perguntas-exemplo (chips) — o app de referência
// não tem chips; eles entram aqui, no slot do estado vazio, para não perder a função.
// -------------------------------------------------------------------------
@Composable
private fun EmptyState(
    examples: List<ExampleQuestion>,
    enabled: Boolean,
    onExample: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(CemigMuted)
                .border(1.dp, CemigBorder, RoundedCornerShape(16.dp))
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                text = "Pergunte sobre as normas e manuais",
                color = CemigForeground,
                fontFamily = InterFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 15.sp
            )
            Text(
                text = "Segure o microfone para perguntar por voz ou digite. Cada pergunta busca trechos " +
                    "nas 36 NRs e 9 manuais e responde citando a norma e o item. Tudo roda neste aparelho.",
                color = CemigMutedFg,
                fontFamily = InterFontFamily,
                fontSize = 12.sp,
                lineHeight = 18.sp
            )
            Spacer(modifier = Modifier.height(4.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Image(
                    painter = painterResource(id = R.drawable.ceia_logo_navy),
                    contentDescription = "CEIA",
                    modifier = Modifier.height(22.dp),
                    contentScale = ContentScale.Fit
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "LFM3 1.7B tool-calling · Nemotron 3.5",
                    color = CemigMutedFg,
                    fontFamily = InterFontFamily,
                    fontSize = 11.sp
                )
            }
        }

        if (examples.isNotEmpty()) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(start = 4.dp)) {
                Icon(Icons.Filled.AutoAwesome, contentDescription = null, tint = CemigPrimary, modifier = Modifier.size(14.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "Perguntas de exemplo",
                    color = CemigMutedFg,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp
                )
            }
            examples.forEach { ex ->
                val isManual = ex.kind == "manual"
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(CemigCard)
                        .border(1.dp, CemigBorder, RoundedCornerShape(16.dp))
                        .clickable(enabled = enabled) { onExample(ex.question) }
                        .padding(horizontal = 14.dp, vertical = 10.dp)
                ) {
                    // Selo da fonte esperada: norma (verde) ou manual (âmbar).
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(999.dp))
                            .background(if (isManual) CemigManual.copy(alpha = 0.12f) else CemigSecondary)
                            .padding(horizontal = 8.dp, vertical = 2.dp)
                    ) {
                        Text(
                            text = (if (isManual) "MANUAL · " else "NORMA · ") + ex.source,
                            color = if (isManual) CemigManual else CemigSecondaryFg,
                            fontFamily = InterFontFamily,
                            fontWeight = FontWeight.Bold,
                            fontSize = 10.sp
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = ex.question,
                        color = if (enabled) CemigForeground else CemigMutedFg,
                        fontFamily = InterFontFamily,
                        fontSize = 13.sp,
                        lineHeight = 19.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Bolhas de chat
// -------------------------------------------------------------------------
@Composable
private fun UserBubble(text: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
        Box(
            modifier = Modifier
                .widthIn(max = 320.dp)
                .clip(RoundedCornerShape(topStart = 24.dp, topEnd = 8.dp, bottomStart = 24.dp, bottomEnd = 24.dp))
                .background(CemigPrimary)
                .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
            Text(
                text = text,
                color = Color.White,
                fontFamily = InterFontFamily,
                fontWeight = FontWeight.Medium,
                fontSize = 15.sp,
                lineHeight = 22.sp
            )
        }
    }
}

@Composable
private fun AssistantLabel(text: String = "Assistente", thinking: Boolean = false) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(start = 4.dp)) {
        Box(
            modifier = Modifier
                .size(28.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(if (thinking) CemigThinkingBorder else CemigSecondary),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = if (thinking) Icons.Filled.Psychology else Icons.Filled.AutoAwesome,
                contentDescription = null,
                tint = if (thinking) CemigThinking else CemigPrimary,
                modifier = Modifier.size(15.dp)
            )
        }
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = text,
            color = CemigMutedFg,
            fontFamily = InterFontFamily,
            fontWeight = FontWeight.Bold,
            fontSize = 12.sp
        )
    }
}

private val AssistantBubbleShape = RoundedCornerShape(topStart = 6.dp, topEnd = 16.dp, bottomStart = 16.dp, bottomEnd = 16.dp)

@Composable
private fun AnswerBubble(text: String, muted: Boolean = false) {
    Box(
        modifier = Modifier
            .fillMaxWidth(0.94f)
            .clip(AssistantBubbleShape)
            .background(CemigCard)
            .border(1.dp, CemigBorder, AssistantBubbleShape)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        Text(
            text = text,
            color = if (muted) CemigMutedFg else CemigForeground,
            fontFamily = InterFontFamily,
            fontWeight = FontWeight.Medium,
            fontSize = 15.sp,
            lineHeight = 23.sp
        )
    }
}

@Composable
private fun AssistantMessage(turn: ConversationTurn, isDebugMode: Boolean) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        AssistantLabel()
        AnswerBubble(text = turn.answer)
        Column(modifier = Modifier.fillMaxWidth(0.94f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            // Citação obrigatória: norma + item de cada trecho usado (expansível).
            SourcesSection(chunks = turn.chunks, reused = turn.chunks.isEmpty() && !turn.calledTool)
            TimingFooter(metrics = turn.metrics)
            if (isDebugMode) {
                DebugTurnCard(metrics = turn.metrics, chunks = turn.chunks, transcription = turn.question)
            }
        }
    }
}

/** Linha de tempos clicável + painel de detalhes por etapa (como o rodapé do app de referência). */
@Composable
private fun TimingFooter(metrics: TurnMetrics) {
    var open by rememberSaveable { mutableStateOf(false) }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .clip(RoundedCornerShape(8.dp))
                .clickable { open = !open }
                .padding(start = 4.dp, end = 4.dp, top = 2.dp, bottom = 2.dp)
        ) {
            val tps = if (metrics.tokPerSec > 0) "%.1f".format(metrics.tokPerSec) else "—"
            Text(
                text = "${fmtMs(metrics.totalMs)} · 1º token ${fmtMs(metrics.ttftMs)} · $tps tok/s",
                color = CemigMutedFg,
                fontFamily = InterFontFamily,
                fontSize = 11.sp
            )
            Icon(
                imageVector = if (open) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                contentDescription = if (open) "Recolher tempos" else "Ver tempos por etapa",
                tint = CemigMutedFg,
                modifier = Modifier.size(15.dp)
            )
        }
        AnimatedVisibility(visible = open) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(CemigCard)
                    .border(1.dp, CemigBorder, RoundedCornerShape(16.dp))
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                DetailRow("Voz (ASR)", if (metrics.asrMs > 0) fmtMs(metrics.asrMs) else "— (digitada)")
                DetailRow("Decisão do modelo", (if (metrics.calledTool) "buscou · " else "sem busca · ") + fmtMs(metrics.reasonMs))
                DetailRow("Busca nas normas", if (metrics.calledTool) fmtMs(metrics.searchMs) else "—")
                DetailRow("1º token", fmtMs(metrics.ttftMs))
                DetailRow("Geração", fmtMs(metrics.decodeMs))
                DetailRow("Tokens (prompt/resposta)", "${metrics.promptTokens.takeIf { it > 0 } ?: metrics.contextTokens} / ${metrics.completionTokens}")
                DetailRow("Total", fmtMs(metrics.totalMs))
            }
        }
    }
}

@Composable
private fun DetailRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, color = CemigMutedFg, fontFamily = InterFontFamily, fontSize = 11.sp)
        Spacer(modifier = Modifier.width(12.dp))
        Text(value, color = CemigForeground, fontFamily = InterFontFamily, fontWeight = FontWeight.SemiBold, fontSize = 11.sp)
    }
}

private fun fmtMs(ms: Long): String =
    if (ms >= 1000) "%.1f s".format(ms / 1000.0) else "$ms ms"

/**
 * Turno em andamento. Bolha "Pensando" (violeta, como no app de referência) enquanto o modelo
 * decide e busca; quando chegam os trechos, as FONTES ficam EM CIMA e o texto em streaming
 * cresce ABAIXO (ajuste de UI 1 do capitão, preservado).
 */
@Composable
private fun ActiveTurn(
    stage: PipelineStage,
    toolCalled: Boolean?,
    searchInProgress: Boolean,
    streamingAnswer: String,
    chunks: List<Chunk>
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (streamingAnswer.isEmpty()) {
            val label = when {
                searchInProgress -> "Buscando trechos nas normas e manuais"
                toolCalled == true -> "Lendo os trechos encontrados"
                toolCalled == false -> "Respondendo com o contexto da conversa"
                stage == PipelineStage.RESPONDING -> "Gerando resposta"
                else -> "Decidindo se precisa buscar nas normas"
            }
            StatusBubble(label = label)
        } else {
            AssistantLabel()
        }
        if (chunks.isNotEmpty()) {
            Box(modifier = Modifier.fillMaxWidth(0.94f)) { SourcesSection(chunks = chunks) }
        }
        if (streamingAnswer.isNotEmpty()) {
            AnswerBubble(text = streamingAnswer)
        }
    }
}

@Composable
private fun StatusBubble(label: String) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        AssistantLabel(text = "Pensando", thinking = true)
        Row(
            modifier = Modifier
                .clip(RoundedCornerShape(16.dp))
                .background(CemigThinkingBg)
                .border(1.dp, CemigThinkingBorder, RoundedCornerShape(16.dp))
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            CircularProgressIndicator(modifier = Modifier.size(16.dp), color = CemigThinking, strokeWidth = 2.dp)
            Spacer(modifier = Modifier.width(10.dp))
            Text(
                text = label,
                color = CemigThinkingText,
                fontFamily = InterFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp
            )
        }
    }
}

@Composable
private fun ErrorBanner(message: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(CemigErrorBg)
            .border(1.dp, CemigErrorBorder, RoundedCornerShape(16.dp))
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(Icons.Filled.WarningAmber, contentDescription = null, tint = CemigErrorText, modifier = Modifier.size(16.dp))
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = message,
            color = CemigErrorText,
            fontFamily = InterFontFamily,
            fontWeight = FontWeight.SemiBold,
            fontSize = 12.sp,
            lineHeight = 18.sp
        )
    }
}

// -------------------------------------------------------------------------
// Composer: cartão arredondado com campo multilinha + botão circular contextual
// (microfone segurar-para-falar quando vazio; enviar quando há texto).
// -------------------------------------------------------------------------
@Composable
private fun Composer(
    input: String,
    isListening: Boolean,
    isTranscribing: Boolean,
    awaitingConfirmation: Boolean,
    pipelineBusy: Boolean,
    onInputChange: (String) -> Unit,
    onStartPress: () -> Unit,
    onRelease: () -> Unit,
    onSend: () -> Unit
) {
    Column(modifier = Modifier.fillMaxWidth().background(CemigCard)) {
        Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(CemigBorder))
        Column(modifier = Modifier.padding(start = 12.dp, end = 12.dp, top = 10.dp, bottom = 12.dp)) {
            // Dica contextual de voz: gravando / transcrevendo / revisar antes de enviar.
            val hint = when {
                isListening -> Triple(CemigDestructive, Icons.Filled.GraphicEq, "Gravando… solte o microfone para transcrever")
                isTranscribing -> Triple(CemigThinking, Icons.Filled.GraphicEq, "Transcrevendo sua fala…")
                awaitingConfirmation -> Triple(CemigPrimary, Icons.Outlined.CheckCircle, "Revise a transcrição e toque em enviar")
                else -> null
            }
            if (hint != null) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(hint.first.copy(alpha = 0.10f))
                        .padding(horizontal = 12.dp, vertical = 7.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(hint.second, contentDescription = null, tint = hint.first, modifier = Modifier.size(15.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = hint.third,
                        color = hint.first,
                        fontFamily = InterFontFamily,
                        fontWeight = FontWeight.Bold,
                        fontSize = 12.sp
                    )
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .shadow(1.dp, RoundedCornerShape(24.dp))
                    .clip(RoundedCornerShape(24.dp))
                    .background(CemigCard)
                    .border(1.dp, if (isListening) CemigDestructive else CemigBorder, RoundedCornerShape(24.dp))
                    .padding(start = 16.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
                verticalAlignment = Alignment.Bottom
            ) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 56.dp, max = 168.dp)
                        .padding(vertical = 8.dp),
                    contentAlignment = Alignment.CenterStart
                ) {
                    // Campo multilinha legível (16sp, cresce até ~7 linhas e rola) — fix do capitão.
                    BasicTextField(
                        value = input,
                        onValueChange = onInputChange,
                        textStyle = TextStyle(
                            color = CemigForeground,
                            fontSize = 16.sp,
                            lineHeight = 22.sp,
                            fontFamily = InterFontFamily
                        ),
                        cursorBrush = SolidColor(CemigPrimary),
                        keyboardOptions = KeyboardOptions.Default,
                        modifier = Modifier
                            .fillMaxWidth()
                            .verticalScroll(rememberScrollState())
                            .semantics { contentDescription = "Mensagem para o assistente" },
                        decorationBox = { inner ->
                            if (input.isEmpty()) {
                                Text(
                                    text = if (isListening) "Ouvindo…" else "Fale ou digite sua dúvida sobre as normas…",
                                    color = CemigPlaceholder,
                                    fontSize = 16.sp,
                                    fontFamily = InterFontFamily
                                )
                            }
                            inner()
                        }
                    )
                }
                if (input.isNotEmpty() && !isListening) {
                    // Limpar: volta o botão para microfone (permite regravar).
                    Box(
                        modifier = Modifier
                            .padding(bottom = 10.dp)
                            .size(36.dp)
                            .clip(CircleShape)
                            .clickable { onInputChange("") }
                            .semantics { contentDescription = "Limpar" },
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.Filled.Close, contentDescription = null, tint = CemigMutedFg, modifier = Modifier.size(18.dp))
                    }
                }
                Spacer(modifier = Modifier.width(4.dp))
                PushToTalkButton(
                    isListening = isListening,
                    hasText = input.isNotBlank(),
                    sendEnabled = input.isNotBlank() && !pipelineBusy,
                    busy = pipelineBusy,
                    onStartPress = onStartPress,
                    onRelease = onRelease,
                    onSend = onSend
                )
            }
            Text(
                text = if (input.isBlank() && !isListening) "Segure o microfone para falar · 100% offline" else " ",
                color = CemigMutedFg,
                fontFamily = InterFontFamily,
                fontSize = 11.sp,
                modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center
            )
        }
    }
}

// -------------------------------------------------------------------------
// Splash animado curto (ideia do AnimatedSplashOverlay do app de referência, recriada):
// fundo verde com o selo + nome, some com fade. Só visual; motores carregam por baixo.
// -------------------------------------------------------------------------
@Composable
private fun SplashOverlay() {
    var visible by rememberSaveable { mutableStateOf(true) }
    LaunchedEffect(Unit) {
        delay(900)
        visible = false
    }
    AnimatedVisibility(visible = visible, exit = fadeOut(animationSpec = tween(500))) {
        Box(
            modifier = Modifier.fillMaxSize().background(CemigPrimary),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Box(
                    modifier = Modifier
                        .size(112.dp)
                        .clip(RoundedCornerShape(36.dp))
                        .background(Color.White.copy(alpha = 0.16f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Filled.ElectricBolt, contentDescription = null, tint = Color.White, modifier = Modifier.size(64.dp))
                }
                Spacer(modifier = Modifier.height(18.dp))
                Text(
                    text = "Assistente CEMIG",
                    color = Color.White,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 24.sp
                )
                Text(
                    text = "CEIA · normas no bolso, 100% offline",
                    color = Color.White.copy(alpha = 0.85f),
                    fontFamily = InterFontFamily,
                    fontSize = 13.sp
                )
            }
        }
    }
}
