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
        const val ACTION_RUN_F7_E2E = "br.org.ceia.cemigpoc.RUN_F7_E2E"

        /** Gancho registrado pelo MainViewModel (motores já carregados) p/ o E2E da F7. */
        @Volatile
        var f7Hook: ((Set<String>?, Boolean?, Boolean) -> Unit)? = null
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
            ACTION_RUN_F7_E2E -> {
                // Usa o pipeline JÁ carregado do app (abra o app antes: am start). Opcional:
                // --es ids "id1,id2" para rodar só alguns casos.
                val ids = intent.getStringExtra("ids")?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() }?.toSet()
                val hook = f7Hook
                if (hook == null) Log.e(TAG, "RUN_F7_E2E: app não inicializado (abra o app antes)")
                // --ez keep_prefix false: desliga o reuso de prefixo decisão->síntese (A/B de TTFT).
                // --ez voice true: modo VOZ (f7_voice/<id>.wav -> ASR ativo -> pipeline).
                else hook(ids, if (intent.hasExtra("keep_prefix")) intent.getBooleanExtra("keep_prefix", true) else null,
                    intent.getBooleanExtra("voice", false))
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
