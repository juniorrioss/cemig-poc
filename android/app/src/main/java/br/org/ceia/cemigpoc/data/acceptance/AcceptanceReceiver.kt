package br.org.ceia.cemigpoc.data.acceptance

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * BroadcastReceiver para disparo do roteiro de aceitação via ADB:
 * `adb shell am broadcast -a br.org.ceia.cemigpoc.RUN_ACCEPTANCE`
 */
class AcceptanceReceiver : BroadcastReceiver() {

    companion object {
        const val ACTION_RUN_ACCEPTANCE = "br.org.ceia.cemigpoc.RUN_ACCEPTANCE"
        const val ACTION_RUN_TOOL_E2E = "br.org.ceia.cemigpoc.RUN_TOOL_E2E"
        private const val TAG = "AcceptanceReceiver"
    }

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            ACTION_RUN_ACCEPTANCE -> {
                Log.i(TAG, "Recebido comando para executar roteiro de aceitação CEMIG POC M2")
                val pendingResult = goAsync()
                CoroutineScope(Dispatchers.IO).launch {
                    try {
                        AcceptanceRunner.run(context.applicationContext)
                    } catch (e: Throwable) {
                        Log.e(TAG, "Falha na execução do roteiro de aceitação", e)
                    } finally {
                        pendingResult.finish()
                    }
                }
            }
            ACTION_RUN_TOOL_E2E -> {
                Log.i(TAG, "Recebido comando para executar E2E do pipeline híbrido de tool-calling")
                // O E2E gera por vários minutos (5 casos × 2 turnos). NÃO seguramos o broadcast
                // com goAsync (estoura o timeout -> ANR/kill); finalizamos já e rodamos detached
                // num escopo de processo. O disparo ADB deve manter o app em foreground para o
                // processo não ser reciclado (o runner segura um PARTIAL_WAKE_LOCK).
                val app = context.applicationContext
                CoroutineScope(Dispatchers.IO).launch {
                    try {
                        ToolE2ERunner.run(app)
                    } catch (e: Throwable) {
                        Log.e(TAG, "Falha na execução do E2E de tool-calling", e)
                    }
                }
            }
        }
    }
}
