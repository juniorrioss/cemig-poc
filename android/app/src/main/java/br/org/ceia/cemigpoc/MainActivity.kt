package br.org.ceia.cemigpoc

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import br.org.ceia.cemigpoc.ui.MainScreen
import br.org.ceia.cemigpoc.ui.MainViewModel
import br.org.ceia.cemigpoc.ui.theme.CeiaTheme

/**
 * Ponto de entrada da aplicação Android (Single-Activity Jetpack Compose).
 */
class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CeiaTheme {
                MainScreen(viewModel = viewModel)
            }
        }
    }
}
