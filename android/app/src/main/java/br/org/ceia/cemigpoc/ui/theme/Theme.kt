package br.org.ceia.cemigpoc.ui.theme

import android.app.Activity
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// ==========================================================================
// CEIA Theme — Esquema de cores e tokens do Design System CEIA
// Paleta oficial: Navy 950 (#001a5b), Blue 500 (#2866e5), Teal 500 (#0cd4aa),
// Cream (#f2f1ee), superfícies brancas e neutros derivados do marinho.
// ==========================================================================

// F8: esquema de cores aproximado do app de referência (verde CEMIG sobre fundo claro).
private val CeiaLightColorScheme = lightColorScheme(
    primary = CemigPrimary,
    onPrimary = CeiaWhite,
    primaryContainer = CemigSecondary,
    onPrimaryContainer = CemigSecondaryFg,
    secondary = CemigSecondaryFg,
    onSecondary = CeiaWhite,
    secondaryContainer = CemigSecondary,
    onSecondaryContainer = CemigSecondaryFg,
    tertiary = CemigRing,
    onTertiary = CeiaWhite,
    tertiaryContainer = CemigAccent,
    onTertiaryContainer = CemigSecondaryFg,
    background = CemigBackground,
    onBackground = CemigForeground,
    surface = CemigCard,
    onSurface = CemigForeground,
    surfaceVariant = CemigMuted,
    onSurfaceVariant = CemigMutedFg,
    outline = CemigBorder,
    outlineVariant = CemigBorder,
    error = CemigDestructive,
    onError = CeiaWhite,
    errorContainer = CemigErrorBg,
    onErrorContainer = CemigErrorText
)

@Composable
fun CeiaTheme(
    content: @Composable () -> Unit
) {
    val colorScheme = CeiaLightColorScheme
    val view = LocalView.current

    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as? Activity)?.window
            if (window != null) {
                // F8: barra de status clara com ícones escuros (StatusBar style="dark" no app de referência)
                window.statusBarColor = CemigCard.toArgb()
                window.navigationBarColor = CemigCard.toArgb()
                val controller = WindowCompat.getInsetsController(window, view)
                controller.isAppearanceLightStatusBars = true
                controller.isAppearanceLightNavigationBars = true
            }
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = CeiaTypography,
        shapes = CeiaShapes,
        content = content
    )
}
