package br.org.ceia.cemigpoc.ui

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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AirplanemodeActive
import androidx.compose.material3.Icon
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.org.ceia.cemigpoc.R
import br.org.ceia.cemigpoc.ui.components.PushToTalkButton
import br.org.ceia.cemigpoc.ui.components.ResponseCard
import br.org.ceia.cemigpoc.ui.components.SourcesSection
import br.org.ceia.cemigpoc.ui.components.StatusBadge
import br.org.ceia.cemigpoc.ui.components.TranscriptionCard
import br.org.ceia.cemigpoc.ui.theme.ActayWideFontFamily
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue100
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue500
import br.org.ceia.cemigpoc.ui.theme.CeiaCream
import br.org.ceia.cemigpoc.ui.theme.CeiaGray200
import br.org.ceia.cemigpoc.ui.theme.CeiaGray500
import br.org.ceia.cemigpoc.ui.theme.CeiaGray700
import br.org.ceia.cemigpoc.ui.theme.CeiaGray900
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy800
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy950
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusFull
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusMd
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusSm
import br.org.ceia.cemigpoc.ui.theme.CeiaWhite
import br.org.ceia.cemigpoc.ui.theme.InterFontFamily

/**
 * Tela única do Assistente Offline para operários de campo.
 */
@Composable
fun MainScreen(
    viewModel: MainViewModel,
    modifier: Modifier = Modifier
) {
    val uiState by viewModel.uiState.collectAsState()

    val quickQueries = listOf(
        "Quais as etapas para desenergização na NR-10?",
        "Qual o EPI obrigatório para trabalho em altura na NR-35?",
        "Vestimentas e adornos são permitidos pela NR-10?",
        "Olá, boa tarde assistente!"
    )

    Scaffold(
        modifier = modifier.fillMaxSize(),
        containerColor = CeiaCream,
        bottomBar = {
            // Área fixa inferior com o botão gigante Push-to-Talk
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(CeiaWhite)
                    .border(1.dp, CeiaGray200)
                    .padding(horizontal = 16.dp, vertical = 14.dp)
            ) {
                PushToTalkButton(
                    isListening = uiState.isListening,
                    onStartPress = { viewModel.startPushToTalk() },
                    onRelease = { viewModel.stopPushToTalk() }
                )
            }
        }
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // -----------------------------------------------------------------
            // Cabeçalho da Marca CEIA e Status de Conexão Offline
            // -----------------------------------------------------------------
            item {
                Column(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        // Logotipo oficial CEIA sem recolorir
                        Image(
                            painter = painterResource(id = R.drawable.ceia_logo_navy),
                            contentDescription = "Logotipo CEIA",
                            modifier = Modifier.height(34.dp),
                            contentScale = ContentScale.Fit
                        )

                        // Indicador de modo avião / 100% offline
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .clip(RoundedCornerShape(CeiaRadiusSm))
                                .background(CeiaBlue100)
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Filled.AirplanemodeActive,
                                contentDescription = null,
                                tint = CeiaNavy800,
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "100% OFFLINE",
                                color = CeiaNavy800,
                                fontFamily = InterFontFamily,
                                fontWeight = FontWeight.Bold,
                                fontSize = 11.sp
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Text(
                        text = "Assistente Operacional de Normas",
                        color = CeiaNavy950,
                        fontFamily = ActayWideFontFamily,
                        fontWeight = FontWeight.Bold,
                        fontSize = 20.sp,
                        lineHeight = 26.sp
                    )

                    Spacer(modifier = Modifier.height(4.dp))

                    Text(
                        text = "Setor Elétrico · Galaxy S21 / S24+ · Resposta ≤ 10s",
                        color = CeiaGray500,
                        fontFamily = InterFontFamily,
                        fontSize = 12.sp
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    // Badge de estado obrigatório do Design System (nunca só por cor)
                    StatusBadge(status = uiState.status)
                }
            }

            // -----------------------------------------------------------------
            // Chips de teste rápido (perguntas técnicas e saudação)
            // -----------------------------------------------------------------
            item {
                Column {
                    Text(
                        text = "Exemplos rápidos de perguntas:",
                        color = CeiaGray700,
                        fontFamily = InterFontFamily,
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 12.sp
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    LazyRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(quickQueries) { query ->
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(CeiaRadiusFull))
                                    .background(CeiaWhite)
                                    .border(1.dp, CeiaBlue500.copy(alpha = 0.4f), RoundedCornerShape(CeiaRadiusFull))
                                    .clickable { viewModel.processQuestion(query) }
                                    .padding(horizontal = 12.dp, vertical = 6.dp)
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

            // -----------------------------------------------------------------
            // Card de Transcrição para Conferência
            // -----------------------------------------------------------------
            item {
                TranscriptionCard(
                    transcription = uiState.transcription,
                    isListening = uiState.isListening
                )
            }

            // -----------------------------------------------------------------
            // Card de Resposta em Streaming
            // -----------------------------------------------------------------
            item {
                ResponseCard(
                    responseText = uiState.streamingResponse,
                    toolCall = uiState.activeToolCall,
                    totalDurationMs = uiState.totalDurationMs
                )
            }

            // -----------------------------------------------------------------
            // Seção de Fontes Obrigatórias Expansíveis
            // -----------------------------------------------------------------
            item {
                SourcesSection(chunks = uiState.sources)
            }

            // Espaço final para respiro de rolagem
            item {
                Spacer(modifier = Modifier.height(24.dp))
            }
        }
    }
}
