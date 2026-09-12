package br.org.ceia.cemigpoc.ui

import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AirplanemodeActive
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.org.ceia.cemigpoc.R
import br.org.ceia.cemigpoc.domain.model.ConversationTurn
import br.org.ceia.cemigpoc.domain.model.PipelineStage
import br.org.ceia.cemigpoc.ui.components.DebugTurnCard
import br.org.ceia.cemigpoc.ui.components.PushToTalkButton
import br.org.ceia.cemigpoc.ui.components.SourcesSection
import br.org.ceia.cemigpoc.ui.theme.ActayWideFontFamily
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue100
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue500
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue600
import br.org.ceia.cemigpoc.ui.theme.CeiaCream
import br.org.ceia.cemigpoc.ui.theme.CeiaDanger
import br.org.ceia.cemigpoc.ui.theme.CeiaGray100
import br.org.ceia.cemigpoc.ui.theme.CeiaGray200
import br.org.ceia.cemigpoc.ui.theme.CeiaGray500
import br.org.ceia.cemigpoc.ui.theme.CeiaGray700
import br.org.ceia.cemigpoc.ui.theme.CeiaGray900
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy800
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy950
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusFull
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusLg
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusMd
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusSm
import br.org.ceia.cemigpoc.ui.theme.CeiaSuccess
import br.org.ceia.cemigpoc.ui.theme.CeiaTeal600
import br.org.ceia.cemigpoc.ui.theme.CeiaWhite
import br.org.ceia.cemigpoc.ui.theme.InterFontFamily

/**
 * Interface conversacional multiturno de campo para operários do setor elétrico.
 */
@Composable
fun MainScreen(
    viewModel: MainViewModel,
    modifier: Modifier = Modifier
) {
    val uiState by viewModel.uiState.collectAsState()
    val listState = rememberLazyListState()

    // Rola automaticamente para o fim da conversa ao receber novas mensagens
    LaunchedEffect(uiState.history.size, uiState.currentStreamingAnswer) {
        val totalItems = uiState.history.size * 2 + if (uiState.currentStreamingAnswer.isNotEmpty()) 2 else 0
        if (totalItems > 0) {
            listState.animateScrollToItem(listState.layoutInfo.totalItemsCount.coerceAtLeast(1) - 1)
        }
    }

    val quickQueries = listOf(
        "Quais as etapas obrigatórias para desenergização na NR-10?",
        "Qual o EPI para trabalho em altura acima de 2 metros na NR-35?",
        "Posso trabalhar com aliança no barramento energizado?",
        "Qual a distância segura para média tensão 13,8 kV?"
    )

    Scaffold(
        modifier = modifier.fillMaxSize(),
        containerColor = CeiaCream,
        topBar = {
            // -----------------------------------------------------------------
            // Cabeçalho institucional CEIA + Ações Rápidas (Debug e Nova Conversa)
            // -----------------------------------------------------------------
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(CeiaWhite)
                    .border(1.dp, CeiaGray200)
                    .padding(horizontal = 16.dp, vertical = 10.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Image(
                        painter = painterResource(id = R.drawable.ceia_logo_navy),
                        contentDescription = "CEIA CEMIG",
                        modifier = Modifier.height(28.dp),
                        contentScale = ContentScale.Fit
                    )

                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        // Badge Offline
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .clip(RoundedCornerShape(CeiaRadiusSm))
                                .background(CeiaBlue100)
                                .padding(horizontal = 6.dp, vertical = 3.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Filled.AirplanemodeActive,
                                contentDescription = null,
                                tint = CeiaNavy800,
                                modifier = Modifier.size(12.dp)
                            )
                            Spacer(modifier = Modifier.width(3.dp))
                            Text(
                                text = "100% OFFLINE",
                                color = CeiaNavy800,
                                fontFamily = InterFontFamily,
                                fontWeight = FontWeight.Bold,
                                fontSize = 10.sp
                            )
                        }

                        // Botão Nova Conversa
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(CeiaRadiusSm))
                                .background(CeiaGray100)
                                .clickable { viewModel.newConversation() }
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = Icons.Filled.Refresh,
                                    contentDescription = "Nova Conversa",
                                    tint = CeiaNavy950,
                                    modifier = Modifier.size(14.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(
                                    text = "NOVA CONVERSA",
                                    color = CeiaNavy950,
                                    fontFamily = InterFontFamily,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 10.sp
                                )
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(6.dp))

                // Linha de controle: Modo Engenharia / Debug solicitado pelo Capitão
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Filled.BugReport,
                            contentDescription = null,
                            tint = if (uiState.isDebugMode) CeiaTeal600 else CeiaGray500,
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "MODO ENGENHARIA / DEBUG",
                            color = if (uiState.isDebugMode) CeiaNavy950 else CeiaGray500,
                            fontFamily = InterFontFamily,
                            fontWeight = if (uiState.isDebugMode) FontWeight.Bold else FontWeight.Medium,
                            fontSize = 11.sp
                        )
                    }

                    Switch(
                        checked = uiState.isDebugMode,
                        onCheckedChange = { viewModel.toggleDebugMode() },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = CeiaWhite,
                            checkedTrackColor = CeiaTeal600,
                            uncheckedThumbColor = CeiaWhite,
                            uncheckedTrackColor = CeiaGray200
                        ),
                        modifier = Modifier.height(24.dp)
                    )
                }
            }
        },
        bottomBar = {
            // -----------------------------------------------------------------
            // Barra inferior fixa: Indicador de Etapa + Entrada de Texto + PTT
            // -----------------------------------------------------------------
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(CeiaWhite)
                    .border(1.dp, CeiaGray200)
                    .padding(horizontal = 14.dp, vertical = 10.dp)
            ) {
                // Indicador dinâmico de etapa (o operário vê que o app está vivo)
                StageIndicatorBanner(stage = uiState.stage, errorMessage = uiState.errorMessage)

                // Banner de confirmação: após a transcrição, o operário REVISA e só
                // então envia. Nunca disparamos a resposta sem exibir a fala transcrita.
                if (uiState.awaitingConfirmation) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(CeiaRadiusSm))
                            .background(CeiaSuccess.copy(alpha = 0.14f))
                            .padding(horizontal = 10.dp, vertical = 5.dp),
                        contentAlignment = Alignment.CenterStart
                    ) {
                        Text(
                            text = "REVISE A TRANSCRIÇÃO E TOQUE EM ENVIAR",
                            color = CeiaSuccess,
                            fontFamily = InterFontFamily,
                            fontWeight = FontWeight.Bold,
                            fontSize = 11.sp
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Campo editável de transcrição / digitação
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = uiState.currentQuestionInput,
                        onValueChange = { viewModel.onQuestionInputChanged(it) },
                        placeholder = {
                            Text(
                                text = if (uiState.isListening) "Ouvindo microfone..." else "Fale pelo PTT ou digite sua dúvida...",
                                color = CeiaGray500,
                                fontSize = 13.sp,
                                fontFamily = InterFontFamily
                            )
                        },
                        modifier = Modifier.weight(1f),
                        singleLine = true,
                        shape = RoundedCornerShape(CeiaRadiusMd),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = CeiaBlue500,
                            unfocusedBorderColor = CeiaGray200,
                            focusedContainerColor = CeiaWhite,
                            unfocusedContainerColor = CeiaWhite
                        ),
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                        keyboardActions = KeyboardActions(onSend = { viewModel.submitQuestion() }),
                        trailingIcon = {
                            if (uiState.currentQuestionInput.isNotEmpty()) {
                                IconButton(onClick = { viewModel.onQuestionInputChanged("") }) {
                                    Icon(
                                        imageVector = Icons.Filled.Clear,
                                        contentDescription = "Limpar",
                                        tint = CeiaGray500,
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                            }
                        }
                    )

                    Spacer(modifier = Modifier.width(8.dp))

                    // Habilitado sempre que houver texto e o pipeline NÃO estiver processando
                    // ativamente (classificando/respondendo). Assim uma transcrição (IDLE +
                    // awaitingConfirmation), um texto digitado ou uma reescrita após erro de
                    // voz podem ser enviados; só bloqueia durante a geração em andamento.
                    val pipelineBusy = uiState.stage == PipelineStage.CLASSIFYING ||
                        uiState.stage == PipelineStage.RESPONDING
                    IconButton(
                        onClick = { viewModel.submitQuestion() },
                        enabled = uiState.currentQuestionInput.isNotBlank() && !pipelineBusy,
                        modifier = Modifier
                            .size(48.dp)
                            .clip(CircleShape)
                            .background(if (uiState.currentQuestionInput.isNotBlank()) CeiaBlue500 else CeiaGray200)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Send,
                            contentDescription = "Enviar Pergunta",
                            tint = CeiaWhite,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Botão gigante Push-To-Talk para uso com luvas de eletricista
                PushToTalkButton(
                    isListening = uiState.isListening,
                    onStartPress = { viewModel.startPushToTalk() },
                    onRelease = { viewModel.stopPushToTalk() }
                )
            }
        }
    ) { innerPadding ->
        LazyColumn(
            state = listState,
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            contentPadding = PaddingValues(14.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // Se os modelos ainda estiverem copiando/carregando
            if (!uiState.isModelReady) {
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(CeiaRadiusMd))
                            .background(CeiaBlue100)
                            .padding(14.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(18.dp),
                                color = CeiaNavy800,
                                strokeWidth = 2.dp
                            )
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(
                                text = uiState.modelStatusMessage,
                                color = CeiaNavy800,
                                fontFamily = InterFontFamily,
                                fontWeight = FontWeight.SemiBold,
                                fontSize = 12.sp
                            )
                        }
                    }
                }
            }

            // Histórico vazio: exibe apresentação e perguntas rápidas
            if (uiState.history.isEmpty() && uiState.currentStreamingAnswer.isEmpty()) {
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(CeiaRadiusLg))
                            .background(CeiaWhite)
                            .border(1.dp, CeiaGray200, RoundedCornerShape(CeiaRadiusLg))
                            .padding(16.dp)
                    ) {
                        Text(
                            text = "Assistente Operacional de Normas",
                            color = CeiaNavy950,
                            fontFamily = ActayWideFontFamily,
                            fontWeight = FontWeight.Bold,
                            fontSize = 18.sp
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Motor LFM2.5 1.2B Instruct · Whisper Base Q5_1 · 36 NRs",
                            color = CeiaGray500,
                            fontFamily = InterFontFamily,
                            fontSize = 12.sp
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = "Segure o botão Push-to-Talk para perguntar por voz, ou toque em um dos exemplos rápidos abaixo:",
                            color = CeiaGray700,
                            fontFamily = InterFontFamily,
                            fontSize = 13.sp,
                            lineHeight = 18.sp
                        )
                        Spacer(modifier = Modifier.height(12.dp))

                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            quickQueries.forEach { query ->
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clip(RoundedCornerShape(CeiaRadiusSm))
                                        .background(CeiaGray100)
                                        .border(1.dp, CeiaBlue500.copy(alpha = 0.3f), RoundedCornerShape(CeiaRadiusSm))
                                        .clickable { viewModel.processQuestion(query) }
                                        .padding(horizontal = 12.dp, vertical = 8.dp)
                                ) {
                                    Text(
                                        text = query,
                                        color = CeiaNavy950,
                                        fontFamily = InterFontFamily,
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.Medium
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // Lista de trocas da conversa multiturno
            items(uiState.history) { turn ->
                ConversationTurnItem(
                    turn = turn,
                    isDebugMode = uiState.isDebugMode
                )
            }

            // Turno ativo atualmente em geração/streaming
            if (uiState.currentStreamingAnswer.isNotEmpty() || (uiState.stage != PipelineStage.IDLE && uiState.stage != PipelineStage.ERROR)) {
                item {
                    ActiveStreamingItem(
                        streamingAnswer = uiState.currentStreamingAnswer,
                        chunks = uiState.currentChunks
                    )
                }
            }

            // Espaço final para rolagem
            item {
                Spacer(modifier = Modifier.height(20.dp))
            }
        }
    }
}

/**
 * Item de uma troca completa na lista de conversa.
 */
@Composable
private fun ConversationTurnItem(
    turn: ConversationTurn,
    isDebugMode: Boolean
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Balão do Usuário (à direita)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(0.88f)
                    .clip(RoundedCornerShape(topStart = 16.dp, topEnd = 4.dp, bottomStart = 16.dp, bottomEnd = 16.dp))
                    .background(CeiaBlue600)
                    .padding(14.dp)
            ) {
                Text(
                    text = turn.question,
                    color = CeiaWhite,
                    fontFamily = InterFontFamily,
                    fontWeight = FontWeight.Medium,
                    fontSize = 14.sp,
                    lineHeight = 20.sp
                )
            }
        }

        // Balão do Assistente (à esquerda)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Start
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth(0.96f)
                    .clip(RoundedCornerShape(topStart = 4.dp, topEnd = 16.dp, bottomStart = 16.dp, bottomEnd = 16.dp))
                    .background(CeiaWhite)
                    .border(1.dp, CeiaGray200, RoundedCornerShape(topStart = 4.dp, topEnd = 16.dp, bottomStart = 16.dp, bottomEnd = 16.dp))
                    .padding(14.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "ASSISTENTE CEMIG",
                        color = CeiaNavy800,
                        fontFamily = ActayWideFontFamily,
                        fontWeight = FontWeight.Bold,
                        fontSize = 12.sp
                    )
                    Text(
                        text = "%.2fs".format(turn.metrics.totalMs / 1000.0),
                        color = CeiaGray500,
                        fontSize = 11.sp,
                        fontFamily = InterFontFamily
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = turn.answer,
                    color = CeiaGray900,
                    fontFamily = InterFontFamily,
                    fontSize = 14.sp,
                    lineHeight = 21.sp
                )

                Spacer(modifier = Modifier.height(10.dp))

                // Fontes normativas consultadas expansíveis
                SourcesSection(chunks = turn.chunks)

                // Card de diagnóstico e telemetria inline se Modo Engenharia estiver ativo
                if (isDebugMode) {
                    Spacer(modifier = Modifier.height(10.dp))
                    DebugTurnCard(
                        metrics = turn.metrics,
                        chunks = turn.chunks,
                        transcription = turn.question
                    )
                }
            }
        }
    }
}

/**
 * Exibição do turno ativo enquanto o streaming está acontecendo.
 */
@Composable
private fun ActiveStreamingItem(
    streamingAnswer: String,
    chunks: List<br.org.ceia.cemigpoc.domain.model.Chunk>
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth(0.96f)
                .clip(RoundedCornerShape(16.dp))
                .background(CeiaWhite)
                .border(1.dp, CeiaBlue500.copy(alpha = 0.5f), RoundedCornerShape(16.dp))
                .padding(14.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = "ASSISTENTE CEMIG",
                    color = CeiaBlue600,
                    fontFamily = ActayWideFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp
                )
                CircularProgressIndicator(
                    modifier = Modifier.size(14.dp),
                    strokeWidth = 2.dp,
                    color = CeiaBlue500
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = if (streamingAnswer.isNotEmpty()) streamingAnswer else "Processando raciocínio e consultando normas...",
                color = if (streamingAnswer.isNotEmpty()) CeiaGray900 else CeiaGray500,
                fontFamily = InterFontFamily,
                fontSize = 14.sp,
                lineHeight = 21.sp
            )

            if (chunks.isNotEmpty()) {
                Spacer(modifier = Modifier.height(10.dp))
                SourcesSection(chunks = chunks)
            }
        }
    }
}

/**
 * Banner superior com status vivo da etapa corrente do pipeline.
 */
@Composable
private fun StageIndicatorBanner(
    stage: PipelineStage,
    errorMessage: String?
) {
    val (bg, textColor, text) = when (stage) {
        PipelineStage.IDLE -> Triple(CeiaGray100, CeiaGray700, "100% OFFLINE · PRONTO")
        PipelineStage.LISTENING -> Triple(CeiaDanger.copy(alpha = 0.15f), CeiaDanger, "GRAVANDO ÁUDIO DO OPERADOR...")
        PipelineStage.TRANSCRIBING -> Triple(CeiaBlue100, CeiaNavy800, "TRANSCREVENDO (WHISPER BASE Q5_1)...")
        PipelineStage.CLASSIFYING -> Triple(CeiaBlue100, CeiaNavy800, "CLASSIFICANDO NR + BUSCANDO BM25 (TOP-2)...")
        PipelineStage.RESPONDING -> Triple(CeiaBlue100, CeiaNavy800, "SINTETIZANDO RESPOSTA (LFM2.5)...")
        PipelineStage.DONE -> Triple(CeiaSuccess.copy(alpha = 0.15f), CeiaSuccess, "RESPOSTA CONCLUÍDA")
        PipelineStage.ERROR -> Triple(CeiaDanger.copy(alpha = 0.15f), CeiaDanger, errorMessage ?: "ERRO OPERACIONAL")
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(CeiaRadiusSm))
            .background(bg)
            .padding(horizontal = 10.dp, vertical = 5.dp),
        contentAlignment = Alignment.CenterStart
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(textColor)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = text,
                color = textColor,
                fontFamily = InterFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 11.sp
            )
        }
    }
}
