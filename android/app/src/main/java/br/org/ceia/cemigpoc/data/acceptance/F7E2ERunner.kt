package br.org.ceia.cemigpoc.data.acceptance

import android.content.Context
import android.os.Debug
import android.os.PowerManager
import android.util.Log
import br.org.ceia.cemigpoc.BuildConfig
import br.org.ceia.cemigpoc.data.model.ModelFileManager
import br.org.ceia.cemigpoc.domain.model.ConversationTurn
import br.org.ceia.cemigpoc.domain.model.TurnEvent
import br.org.ceia.cemigpoc.domain.pipeline.AskPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * E2E da F7 NO APARELHO: roda as perguntas-exemplo (benchmark operacional) pelo MESMO
 * AskPipeline/retriever/engine que a UI usa (ponto de operação da F6: gate 0,7/2x + manuais
 * margem + top-5 -> compressão 50% + LFM3 f4-r32-ep2 Q4_0), mais um followup (reuso) e uma
 * saudação. Grava o LOG BRUTO por caso (decisão, chamada, chunks, contexto comprimido,
 * resposta, tempos, RAM) em getExternalFilesDir(null)/f7_e2e_results.json + logcat CEMIG_F7_E2E.
 *
 * Dispara: adb shell am broadcast -a br.org.ceia.cemigpoc.RUN_F7_E2E   (app em foreground)
 */
object F7E2ERunner {
    private const val TAG = "CEMIG_F7_E2E"

    suspend fun run(context: Context, pipeline: AskPipeline, onlyIds: Set<String>? = null) = withContext(Dispatchers.IO) {
        val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
        val wake = pm?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "cemigpoc:F7E2E")?.apply { acquire(30 * 60 * 1000L) }
        try {
            val arr = JSONArray(context.assets.open("example_questions.json").bufferedReader().use { it.readText() })
            data class Case(val id: String, val kind: String, val q: String, val followOf: String? = null)
            val cases = ArrayList<Case>()
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                cases.add(Case(o.getString("id"), o.getString("kind"), o.getString("question")))
            }
            // multiturno: followup do 1º exemplo (o modelo deve REUSAR) + saudação (zero busca)
            cases.add(Case("followup_reuso", "followup", "E se a equipe não tiver feito essa avaliação, o serviço pode começar?", followOf = cases[0].id))
            cases.add(Case("saudacao", "smalltalk", "Valeu, obrigado pela ajuda!"))
            val todo = if (onlyIds != null) cases.filter { it.id in onlyIds } else cases

            Log.i(TAG, "==================== E2E F7 (${todo.size} casos) modelo=${ModelFileManager.LLM_MODEL_NAME} asr=${BuildConfig.ASR_ENGINE}")
            val history = HashMap<String, ConversationTurn>()
            val out = JSONArray()
            var peakPss = 0.0
            for (c in todo) {
                val hist = c.followOf?.let { history[it] }?.let { listOf(it) } ?: emptyList()
                val rec = JSONObject().put("id", c.id).put("kind", c.kind).put("question", c.q)
                    .put("history_turns", hist.size)
                val answer = StringBuilder()
                var done: TurnEvent.Done? = null
                val t0 = System.currentTimeMillis()
                var firstDeltaMs = -1L
                pipeline.execute(c.q, hist).collect { ev ->
                    when (ev) {
                        is TurnEvent.ToolDecided -> rec.put("called_tool", ev.called).put("tool_consulta", ev.consulta)
                            .put("tool_nr", ev.nr).put("decision_ms", ev.reasonMs)
                        is TurnEvent.TextDelta -> {
                            if (firstDeltaMs < 0) firstDeltaMs = System.currentTimeMillis() - t0
                            answer.append(ev.text)
                        }
                        is TurnEvent.Done -> done = ev
                        is TurnEvent.Error -> rec.put("error", ev.message)
                        else -> {}
                    }
                }
                val d = done
                if (d != null) {
                    val m = d.metrics
                    rec.put("final_answer", d.finalAnswer)
                        .put("chunks", JSONArray(d.chunksUsed.map { "${it.id} ${it.doc} ${it.section}" }))
                        .put("n_manual_chunks", d.chunksUsed.count { it.id >= 1_000_000L })
                        .put("tool_context", d.toolContext ?: "")
                        .put("gate", m.gateMode).put("nr_top1", m.nrTop1).put("nr_top1_prob", m.nrTop1Prob.toDouble())
                        .put("retrieval", m.retrievalMode).put("manual_on", m.manualOn)
                        .put("budget", m.compressBudget).put("prompt_tokens", m.promptTokens)
                        .put("search_ms", m.searchMs).put("dense_encode_ms", m.denseEncodeMs).put("compress_ms", m.compressMs)
                        .put("ttft_ms", m.ttftMs).put("decode_ms", m.decodeMs).put("completion_tokens", m.completionTokens)
                        .put("tok_per_s", m.tokPerSec).put("total_ms", m.totalMs)
                        .put("first_visible_token_ms", firstDeltaMs)
                    history[c.id] = ConversationTurn(
                        question = c.q, answer = d.finalAnswer, chunks = d.chunksUsed, metrics = m,
                        calledTool = m.calledTool, toolConsulta = m.toolConsulta.ifBlank { null },
                        toolNr = m.toolNr.ifBlank { null }, toolContext = d.toolContext
                    )
                }
                val mi = Debug.MemoryInfo(); Debug.getMemoryInfo(mi)
                val pss = mi.totalPss / 1024.0
                peakPss = maxOf(peakPss, pss)
                rec.put("pss_mb_after", pss)
                out.put(rec)
                Log.i(TAG, "[${c.id}] ${rec.toString().take(3500)}")
            }
            val res = JSONObject().put("model", ModelFileManager.LLM_MODEL_NAME).put("asr", BuildConfig.ASR_ENGINE)
                .put("peak_pss_mb", peakPss).put("cases", out)
            val f = context.getExternalFilesDir(null)?.resolve("f7_e2e_results.json") ?: File(context.filesDir, "f7_e2e_results.json")
            f.writeText(res.toString(1))
            Log.i(TAG, "==================== E2E F7 CONCLUÍDO -> ${f.absolutePath} (pico PSS ${"%.0f".format(peakPss)} MB)")
        } finally {
            try { if (wake?.isHeld == true) wake.release() } catch (_: Exception) {}
        }
    }
}
