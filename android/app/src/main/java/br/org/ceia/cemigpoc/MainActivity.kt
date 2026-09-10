package br.org.ceia.cemigpoc

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import br.org.ceia.cemigpoc.ui.MainScreen
import br.org.ceia.cemigpoc.ui.MainViewModel
import br.org.ceia.cemigpoc.ui.theme.CeiaTheme

/**
 * Ponto de entrada da aplicação Android (Single-Activity Jetpack Compose).
 */
class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        // Permissão de microfone tratada
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Exibe sobre a tela de bloqueio e mantém acordado para testes de campo e telemetria
        setShowWhenLocked(true)
        setTurnScreenOn(true)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Solicita permissão de gravação de áudio se ainda não concedida
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        setContent {
            CeiaTheme {
                MainScreen(viewModel = viewModel)
            }
        }

        val runAcceptance = intent?.getBooleanExtra("run_acceptance", false) ?: false
        if (runAcceptance) {
            viewModel.runAcceptanceTest()
        }
    }
}
