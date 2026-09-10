package br.org.ceia.cemigpoc.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.ui.unit.dp

// ==========================================================================
// CEIA Design System — Formas e Elevação
// Cantos generosos: sm=6dp, md=10dp, lg=16dp, full=999dp
// ==========================================================================

val CeiaRadiusSm = 6.dp
val CeiaRadiusMd = 10.dp
val CeiaRadiusLg = 16.dp
val CeiaRadiusFull = 999.dp

val CeiaShapes = Shapes(
    extraSmall = RoundedCornerShape(CeiaRadiusSm),
    small = RoundedCornerShape(CeiaRadiusMd),
    medium = RoundedCornerShape(CeiaRadiusLg),
    large = RoundedCornerShape(20.dp),
    extraLarge = RoundedCornerShape(CeiaRadiusFull)
)
