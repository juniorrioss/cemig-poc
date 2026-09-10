package br.org.ceia.cemigpoc.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import br.org.ceia.cemigpoc.domain.model.PipelineStage
import br.org.ceia.cemigpoc.ui.theme.CeiaBlue500
import br.org.ceia.cemigpoc.ui.theme.CeiaDanger
import br.org.ceia.cemigpoc.ui.theme.CeiaDangerBg
import br.org.ceia.cemigpoc.ui.theme.CeiaInfoBg
import br.org.ceia.cemigpoc.ui.theme.CeiaNavy800
import br.org.ceia.cemigpoc.ui.theme.CeiaPurple
import br.org.ceia.cemigpoc.ui.theme.CeiaPurpleBg
import br.org.ceia.cemigpoc.ui.theme.CeiaRadiusSm
import br.org.ceia.cemigpoc.ui.theme.CeiaSuccess
import br.org.ceia.cemigpoc.ui.theme.CeiaSuccessBg
import br.org.ceia.cemigpoc.ui.theme.CeiaWarning
import br.org.ceia.cemigpoc.ui.theme.CeiaWarningBg
import br.org.ceia.cemigpoc.ui.theme.InterFontFamily

/**
 * Badge de status que obedece à regra do Design System CEIA:
 * "O estado NUNCA é transmitido apenas por cor — deve sempre incluir um rótulo textual legível".
 */
@Composable
fun StatusBadge(
    stage: PipelineStage,
    modifier: Modifier = Modifier
) {
    val (bgColor, textColor, dotColor) = when (stage) {
        PipelineStage.IDLE -> Triple(CeiaSuccessBg, CeiaSuccess, CeiaSuccess)
        PipelineStage.LISTENING -> Triple(CeiaWarningBg, CeiaWarning, CeiaWarning)
        PipelineStage.TRANSCRIBING -> Triple(CeiaInfoBg, CeiaBlue500, CeiaBlue500)
        PipelineStage.REWRITING -> Triple(CeiaPurpleBg, CeiaPurple, CeiaPurple)
        PipelineStage.SEARCHING -> Triple(CeiaInfoBg, CeiaBlue500, CeiaBlue500)
        PipelineStage.RESPONDING -> Triple(CeiaPurpleBg, CeiaPurple, CeiaPurple)
        PipelineStage.DONE -> Triple(CeiaSuccessBg, CeiaNavy800, CeiaSuccess)
        PipelineStage.ERROR -> Triple(CeiaDangerBg, CeiaDanger, CeiaDanger)
    }

    Row(
        modifier = modifier
            .clip(RoundedCornerShape(CeiaRadiusSm))
            .background(bgColor)
            .border(1.dp, dotColor.copy(alpha = 0.35f), RoundedCornerShape(CeiaRadiusSm))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(dotColor)
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            text = stage.label,
            color = textColor,
            fontFamily = InterFontFamily,
            fontWeight = FontWeight.SemiBold,
            fontSize = 12.sp,
            letterSpacing = 0.5.sp
        )
    }
}
