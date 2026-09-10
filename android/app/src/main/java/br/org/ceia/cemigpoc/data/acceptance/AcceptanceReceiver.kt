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
        private const val TAG = "AcceptanceReceiver"
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == ACTION_RUN_ACCEPTANCE) {
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
    }
}
