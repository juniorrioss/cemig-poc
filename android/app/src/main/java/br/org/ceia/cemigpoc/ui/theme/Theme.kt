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

private val CeiaLightColorScheme = lightColorScheme(
    primary = CeiaBlue500,
    onPrimary = CeiaWhite,
    primaryContainer = CeiaBlue50,
    onPrimaryContainer = CeiaNavy950,
    secondary = CeiaNavy950,
    onSecondary = CeiaWhite,
    secondaryContainer = CeiaBlue100,
    onSecondaryContainer = CeiaNavy800,
    tertiary = CeiaTeal600,
    onTertiary = CeiaWhite,
    tertiaryContainer = CeiaTeal100,
    onTertiaryContainer = CeiaTeal700,
    background = CeiaCream,
    onBackground = CeiaGray900,
    surface = CeiaWhite,
    onSurface = CeiaGray900,
    surfaceVariant = CeiaGray100,
    onSurfaceVariant = CeiaGray700,
    outline = CeiaGray300,
    outlineVariant = CeiaGray200,
    error = CeiaDanger,
    onError = CeiaWhite,
    errorContainer = CeiaDangerBg,
    onErrorContainer = CeiaDanger
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
                window.statusBarColor = CeiaNavy950.toArgb()
                window.navigationBarColor = CeiaCream.toArgb()
                val controller = WindowCompat.getInsetsController(window, view)
                controller.isAppearanceLightStatusBars = false
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
