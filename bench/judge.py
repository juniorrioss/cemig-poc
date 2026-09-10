"""
judge.py — Avaliação automatizada de respostas de SLMs em 4 eixos via LLM Juiz (vLLM Qwen 3.8 27B / Claude CLI) (v2).

Eixos de Avaliação (escala 1 a 5):
1. Acerto Factual (Accuracy)
2. Fidelidade ao Contexto / Disse não-sei quando devia (Faithfulness & Refusal)
3. Qualidade do Português Brasileiro (Fluência técnica)
4. Citação de Fonte Técnica (Norma e item/seção explícitos)

Métricas de Sistema:
- Recall de Retrieval (Hit@3 da norma/seção ouro na busca BM25)
- Latência média total e por turno (Turno 1 / Turno 2 no Modo C)
- Gate Pass Rate (%)

Backend padrão: vLLM dedicado em http://10.100.0.111:8005/v1 (modelo Qwen/Qwen3.8-27B-FP8).
Amostra de calibração: 50 itens comparados entre vLLM e claude -p com medição de concordância (Spearman + %+-1).

AVISO OBRIGATÓRIO: O LLM Juiz é estritamente auxiliar. A validação humana
das 10 primeiras questões do modelo vencedor é mandatória para decisões de homologação em campo.
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("bench.judge")

DEFAULT_VLLM_URL = "http://10.100.0.111:8005/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3.8-27B-FP8"

HUMAN_REVIEW_DISCLAIMER = """
********************************************************************************
AVISO OBRIGATÓRIO DE GOVERNANÇA:
A avaliação automatizada realizada pelo LLM Juiz é uma ferramenta de triagem AUXILIAR.
A leitura humana individual das 10 primeiras questões do vencedor é
MANDATÓRIA antes de qualquer homologação ou decisão de embarque no app de campo.
********************************************************************************
"""


def parse_llm_json_response(raw_text: str) -> Dict[str, Any]:
    """
    Extrai e valida o JSON estruturado gerado pelo LLM Juiz.
    Varre de trás para frente para capturar o ÚLTIMO bloco JSON válido
    (evita capturar rascunhos da cadeia de raciocínio/thinking).
    """
    text = raw_text.strip()

    # 1. Tenta blocos ```json ... ``` de trás para frente
    md_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for block in reversed(md_blocks):
        try:
            d = json.loads(block)
            if "acerto_factual" in d:
                return _normalize_judge_dict(d)
        except Exception:
            pass

    # 2. Tenta todos os blocos {...} de trás para frente
    matches = re.findall(r"(\{(?:[^{}]|\{[^{}]*\})*\})", text, re.DOTALL)
    for m in reversed(matches):
        try:
            d = json.loads(m)
            if "acerto_factual" in d:
                return _normalize_judge_dict(d)
        except Exception:
            pass

    # 3. Fallback: substring entre a primeira { e a última }
    first_idx = text.find("{")
    last_idx = text.rfind("}")
    if first_idx != -1 and last_idx > first_idx:
        try:
            d = json.loads(text[first_idx:last_idx + 1])
            if "acerto_factual" in d:
                return _normalize_judge_dict(d)
        except Exception:
            pass

    logger.warning("Falha ao decodificar JSON do juiz. Texto recebido: %s", text[:120])
    return fallback_rule_judge("", "", "")


def _normalize_judge_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Converte e limita as notas entre 1 e 5."""
    return {
        "acerto_factual": max(1, min(5, int(data.get("acerto_factual", 3)))),
        "fidelidade_contexto": max(1, min(5, int(data.get("fidelidade_contexto", 3)))),
        "qualidade_pt": max(1, min(5, int(data.get("qualidade_pt", 3)))),
        "citacao_fonte": max(1, min(5, int(data.get("citacao_fonte", 1)))),
        "justificativa": str(data.get("justificativa", "")).strip()
    }


def fallback_rule_judge(
    golden: str,
    response: str,
    doc_expected: str
) -> Dict[str, Any]:
    """Avaliação heurística de contingência caso o LLM Juiz falhe ou esteja indisponível."""
    resp_low = response.lower()
    gold_low = golden.lower()

    # Citação de fonte
    cit_score = 1
    if re.search(r"nr-?\d{1,2}", resp_low):
        cit_score = 3
        if re.search(r"(?:item|seção|subitem|§)\s*\d+\.\d+", resp_low):
            cit_score = 5

    # Não sei quando devia
    refusal_in_resp = any(p in resp_low for p in ["não sei", "não consta", "não foi possível identificar"])
    refusal_in_gold = any(p in gold_low for p in ["não sei", "não consta", "exclusivamente"])

    fid_score = 3
    if refusal_in_gold:
        fid_score = 5 if refusal_in_resp else 1
    elif refusal_in_resp:
        fid_score = 3
    else:
        fid_score = 4

    # Qualidade PT
    pt_score = 4
    if any(en in resp_low for en in ["the ", "is ", "and ", "electrical", "safety"]):
        pt_score = 2
    if len(response.strip()) < 10:
        pt_score = 1

    # Acerto factual aproximado por sobreposição
    words_gold = set(re.findall(r"\w+", gold_low)) - {"de", "a", "o", "que", "e", "em", "do", "da"}
    words_resp = set(re.findall(r"\w+", resp_low))
    overlap = len(words_gold & words_resp) / max(1, len(words_gold))

    if refusal_in_gold and refusal_in_resp:
        fac_score = 5
    elif overlap > 0.4:
        fac_score = 4
    elif overlap > 0.2:
        fac_score = 3
    else:
        fac_score = 2

    return {
        "acerto_factual": fac_score,
        "fidelidade_contexto": fid_score,
        "qualidade_pt": pt_score,
        "citacao_fonte": cit_score,
        "justificativa": f"Avaliação heurística (overlap de termos={overlap:.2f}, citação={cit_score})"
    }


def call_llm_judge(
    question: str,
    golden_answer: str,
    model_response: str,
    doc_expected: str,
    section_expected: str,
    cli_tool: str = "vllm",
    vllm_url: str = DEFAULT_VLLM_URL,
    vllm_model: str = DEFAULT_VLLM_MODEL
) -> Dict[str, Any]:
    """Invoca o LLM Juiz via vLLM (padrão de volume) ou CLI local (claude, pi, rule)."""
    if not model_response or not model_response.strip():
        return {
            "acerto_factual": 1,
            "fidelidade_contexto": 1,
            "qualidade_pt": 1,
            "citacao_fonte": 1,
            "justificativa": "Resposta vazia ou truncada."
        }

    prompt = f"""Você é um auditor técnico especialista em segurança do trabalho e Normas Regulamentadoras brasileiras (NR-10, NR-06, NR-35, NR-12, NR-18).
Avalie a resposta gerada por um modelo de linguagem local contra a resposta-ouro oficial.

[PERGUNTA DO OPERÁRIO]:
{question}

[RESPOSTA-OURO DE REFERÊNCIA]:
{golden_answer}
(Norma esperada: {doc_expected.upper()}, Seção: {section_expected})

[RESPOSTA GERADA PELO MODELO LOCAL]:
{model_response}

Critérios de pontuação (atribua uma nota inteira de 1 a 5 para cada eixo):
1. 'acerto_factual' (1-5):
   - 5: Totalmente correta, reflete todos os pontos críticos da resposta-ouro.
   - 4: Quase correta, omite detalhe secundário sem comprometer a segurança.
   - 3: Parcialmente correta, acerta o sentido geral mas com imprecisão.
   - 2: Incorreta ou incompleta demais.
   - 1: Gravemente incorreta ou recomendação perigosa contra a vida.
2. 'fidelidade_contexto' (1-5):
   - 5: Fiel às normas. Se for tema fora do escopo ou pergunta negativa, disse 'não sei' ou negou com clareza.
   - 3: Mistura fatos normativos com suposições não citadas.
   - 1: Alucinação sem base regulamentar.
3. 'qualidade_pt' (1-5):
   - 5: Português brasileiro fluente, técnico, direto e gramaticalmente correto.
   - 3: Compreensível, mas com construções truncadas ou estilo estranho.
   - 1: Mistura com inglês, código solto ou texto ininteligível.
4. 'citacao_fonte' (1-5):
   - 5: Citou explicitamente a norma e item/seção corretos (ex: 'NR-10 item 10.5.1').
   - 3: Citou apenas a norma de forma genérica (ex: 'segundo a NR-10') sem o item.
   - 1: Não citou nenhuma norma ou inventou número inexistente.

Retorne EXCLUSIVAMENTE um objeto JSON estrito com esta estrutura:
{{
  "acerto_factual": 5,
  "fidelidade_contexto": 5,
  "qualidade_pt": 5,
  "citacao_fonte": 5,
  "justificativa": "Texto explicativo sucinto das notas"
}}"""

    # Backend 1: vLLM (volume alto, 8 workers)
    if cli_tool == "vllm":
        try:
            payload = {
                "model": vllm_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}
            }
            res = requests.post(f"{vllm_url}/chat/completions", json=payload, timeout=30)
            if res.status_code == 200:
                raw_out = res.json()["choices"][0]["message"]["content"]
                return parse_llm_json_response(raw_out)
            else:
                logger.warning("vLLM retornou HTTP %d: %s. Tentando fallback.", res.status_code, res.text[:100])
        except Exception as e:
            logger.warning("Falha na chamada vLLM (%s): %s. Usando fallback.", vllm_url, e)

    elif cli_tool == "claude":
        try:
            cmd = ["claude", "-p", prompt, "--bare", "--tools", ""]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if proc.returncode == 0 and proc.stdout.strip():
                return parse_llm_json_response(proc.stdout)
        except Exception as e:
            logger.warning("Falha ao chamar claude CLI: %s. Usando fallback.", e)

    elif cli_tool == "pi":
        try:
            cmd = ["pi", "-p", "--no-tools", prompt]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode == 0 and proc.stdout.strip():
                return parse_llm_json_response(proc.stdout)
        except Exception as e:
            logger.warning("Falha ao chamar pi CLI: %s. Usando fallback.", e)

    # Fallback heurístico
    return fallback_rule_judge(golden_answer, model_response, doc_expected)


def compute_calibration_agreement(
    calib_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calcula concordância estatística entre vLLM e Claude CLI na amostra de calibração."""
    from scipy.stats import spearmanr  # se disponível, senão Pearson básico
    axes = ["acerto_factual", "fidelidade_contexto", "qualidade_pt", "citacao_fonte"]
    report = {}

    for ax in axes:
        vllm_vals = [it["vllm_metrics"][ax] for it in calib_items]
        claude_vals = [it["claude_metrics"][ax] for it in calib_items]
        n = len(vllm_vals)
        if n == 0:
            continue

        # % dentro de +-1 ponto
        within_1 = sum(1 for v, c in zip(vllm_vals, claude_vals) if abs(v - c) <= 1)
        pct_within_1 = round((within_1 / n) * 100, 1)

        # % concordância exata
        exact = sum(1 for v, c in zip(vllm_vals, claude_vals) if v == c)
        pct_exact = round((exact / n) * 100, 1)

        # Spearman rank correlation aproximada
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(vllm_vals, claude_vals)
        except Exception:
            # Fallback Pearson se scipy não estiver instalado
            mean_v = sum(vllm_vals) / n
            mean_c = sum(claude_vals) / n
            num = sum((v - mean_v) * (c - mean_c) for v, c in zip(vllm_vals, claude_vals))
            den = (sum((v - mean_v)**2 for v in vllm_vals) * sum((c - mean_c)**2 for c in claude_vals)) ** 0.5
            corr = num / den if den > 0 else 1.0

        report[ax] = {
            "spearman_corr": round(float(corr), 3),
            "pct_within_1": pct_within_1,
            "pct_exact": pct_exact
        }

    return report


def evaluate_benchmark_results(
    results_file: Path,
    output_eval_file: Path,
    output_csv_file: Path,
    cli_tool: str = "vllm",
    vllm_url: str = DEFAULT_VLLM_URL,
    vllm_model: str = DEFAULT_VLLM_MODEL,
    run_calibration: bool = True
) -> Dict[str, Any]:
    """Avalia todas as respostas do benchmark e gera relatório e CSV consolidado."""
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Carrega cache prévio de avaliações se existir
    cached_evals: Dict[str, Dict[str, Any]] = {}
    if output_eval_file.exists():
        try:
            with open(output_eval_file, "r", encoding="utf-8") as f_prev:
                prev_data = json.load(f_prev)
                for m_k, m_v in prev_data.get("models", {}).items():
                    for mode_k, mode_v in m_v.get("modes", {}).items():
                        for item_rec in mode_v.get("items", []):
                            cache_key = f"{m_k}:{mode_k}:{item_rec.get('item_id')}:{hash(item_rec.get('response', ''))}"
                            cached_evals[cache_key] = item_rec
            logger.info("Carregadas %d avaliações do cache prévio.", len(cached_evals))
        except Exception as e:
            logger.warning("Erro ao carregar cache prévio: %s", e)

    models_data = data.get("models", {})
    evaluated_report: Dict[str, Any] = {
        "metadata": {
            **data.get("metadata", {}),
            "judge_tool": cli_tool,
            "vllm_url": vllm_url,
            "vllm_model": vllm_model,
            "disclaimer": HUMAN_REVIEW_DISCLAIMER.strip(),
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "models": {}
    }

    summary_rows = []
    all_evaluated_items_flat = []

    for model_key, m_info in models_data.items():
        logger.info("Avaliando modelo: %s...", model_key)
        results = m_info.get("results", {})
        model_eval: Dict[str, Any] = {
            "description": m_info.get("description", ""),
            "modes": {}
        }

        # Avalia todos os modos presentes nos resultados do modelo
        for mode_name in ("classic_rag", "query_rewrite_inject", "tool_calling", "classic_gold", "query_rewrite_inject_topk2", "classic_rag_topk2"):
            items = results.get(mode_name, [])
            if not items:
                continue

            evaluated_items = []
            fac_scores, fid_scores, pt_scores, cit_scores, latencies, tok_speeds = [], [], [], [], [], []
            lat_r1_list, lat_r2_list = [], []
            tool_calls_count = 0
            tool_expected_count = 0
            query_reformulated_count = 0
            query_english_count = 0
            retrieval_hits_count = 0

            def _eval_single_item(it):
                resp_text = it.get("response", "")
                question = it.get("question", "")
                golden = it.get("golden_answer", "")
                doc_exp = it.get("doc", "")
                sec_exp = it.get("section", "")
                it_id = it.get("item_id")
                retrieval_hit = it.get("retrieval_hit", False)

                # Verifica cache
                cache_key = f"{model_key}:{mode_name}:{it_id}:{hash(resp_text)}"
                if cache_key in cached_evals:
                    ev_record = cached_evals[cache_key]
                    ev = ev_record.get("metrics", {})
                    return it_id, ev_record, ev

                ev = call_llm_judge(
                    question=question,
                    golden_answer=golden,
                    model_response=resp_text,
                    doc_expected=doc_exp,
                    section_expected=sec_exp,
                    cli_tool=cli_tool,
                    vllm_url=vllm_url,
                    vllm_model=vllm_model
                )

                score_global = round(
                    ev["acerto_factual"] * 0.35 +
                    ev["fidelidade_contexto"] * 0.30 +
                    ev["citacao_fonte"] * 0.20 +
                    ev["qualidade_pt"] * 0.15,
                    2
                )
                pass_gate = (
                    ev["acerto_factual"] >= 3 and
                    ev["fidelidade_contexto"] >= 3 and
                    ev["citacao_fonte"] >= 3 and
                    score_global >= 3.5
                )

                ev_record = {
                    "item_id": it_id,
                    "question": question,
                    "response": resp_text,
                    "golden_answer": golden,
                    "doc": doc_exp,
                    "section": sec_exp,
                    "retrieval_hit": retrieval_hit,
                    "metrics": ev,
                    "score_global": score_global,
                    "pass_gate": pass_gate,
                    "latency_s": it.get("latency_s") or it.get("latency_total_s", 0),
                    "predicted_tok_per_s": it.get("predicted_tok_per_s", 0)
                }

                if mode_name.startswith("query_rewrite_inject"):
                    ev_record["query_rewritten"] = it.get("query_rewritten", "")
                    ev_record["raw_rewrite"] = it.get("raw_rewrite", "")
                    ev_record["latency_round1_s"] = it.get("latency_round1_s", 0)
                    ev_record["latency_round2_s"] = it.get("latency_round2_s", 0)
                elif mode_name.startswith("tool_calling"):
                    ev_record["called_tool"] = it.get("called_tool", False)
                    ev_record["tool_call_format"] = it.get("tool_call_format", "none")
                    ev_record["tool_query"] = it.get("tool_query", "")
                    ev_record["tool_query_reformulated"] = it.get("tool_query_reformulated", False)
                    ev_record["tool_query_is_english"] = it.get("tool_query_is_english", False)
                    ev_record["latency_round1_s"] = it.get("latency_round1_s", 0)
                    ev_record["latency_round2_s"] = it.get("latency_round2_s", 0)

                return it_id, ev_record, ev

            item_order = {it.get("item_id"): i for i, it in enumerate(items)}
            evaluated_items_dict = {}
            ev_list = []

            # 8 workers paralelos para vLLM
            max_workers = 8 if cli_tool == "vllm" else (5 if cli_tool == "claude" else 10)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_eval_single_item, it) for it in items]
                for fut in as_completed(futures):
                    item_id, ev_record, ev = fut.result()
                    evaluated_items_dict[item_id] = ev_record
                    ev_list.append((item_id, ev, ev_record))

            sorted_item_ids = sorted(evaluated_items_dict.keys(), key=lambda x: item_order.get(x, 0))
            evaluated_items = [evaluated_items_dict[x] for x in sorted_item_ids]
            all_evaluated_items_flat.extend(evaluated_items)

            for it_id, ev, ev_record in ev_list:
                fac_scores.append(ev["acerto_factual"])
                fid_scores.append(ev["fidelidade_contexto"])
                pt_scores.append(ev["qualidade_pt"])
                cit_scores.append(ev["citacao_fonte"])

                lat = ev_record.get("latency_s", 0)
                if lat:
                    latencies.append(lat)
                spd = ev_record.get("predicted_tok_per_s", 0)
                if spd:
                    tok_speeds.append(spd)

                if ev_record.get("retrieval_hit"):
                    retrieval_hits_count += 1

                if mode_name.startswith("query_rewrite_inject"):
                    lat_r1_list.append(ev_record.get("latency_round1_s", 0))
                    lat_r2_list.append(ev_record.get("latency_round2_s", 0))
                elif mode_name.startswith("tool_calling"):
                    if ev_record.get("called_tool"):
                        tool_calls_count += 1
                    orig_item = next((x for x in items if x.get("item_id") == it_id), {})
                    if ev_record.get("called_tool") == orig_item.get("requires_tool", True):
                        tool_expected_count += 1
                    if ev_record.get("tool_query_reformulated"):
                        query_reformulated_count += 1
                    if ev_record.get("tool_query_is_english"):
                        query_english_count += 1
                    lat_r1_list.append(ev_record.get("latency_round1_s", 0))
                    lat_r2_list.append(ev_record.get("latency_round2_s", 0))

            n_items = len(items)
            avg_fac = round(sum(fac_scores) / n_items, 2) if n_items else 0
            avg_fid = round(sum(fid_scores) / n_items, 2) if n_items else 0
            avg_pt = round(sum(pt_scores) / n_items, 2) if n_items else 0
            avg_cit = round(sum(cit_scores) / n_items, 2) if n_items else 0
            avg_global = round(
                avg_fac * 0.35 + avg_fid * 0.30 + avg_cit * 0.20 + avg_pt * 0.15,
                2
            )
            pass_count = sum(1 for e in evaluated_items if e["pass_gate"])
            pass_rate = round((pass_count / n_items) * 100, 1) if n_items else 0
            avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0
            avg_spd = round(sum(tok_speeds) / len(tok_speeds), 1) if tok_speeds else 0
            recall_rate = round((retrieval_hits_count / n_items) * 100, 1) if n_items else 0

            mode_summary = {
                "total_items": n_items,
                "retrieval_recall_pct": recall_rate,
                "retrieval_hits": retrieval_hits_count,
                "acerto_factual_avg": avg_fac,
                "fidelidade_contexto_avg": avg_fid,
                "qualidade_pt_avg": avg_pt,
                "citacao_fonte_avg": avg_cit,
                "score_global_avg": avg_global,
                "gate_pass_rate_pct": pass_rate,
                "latency_avg_s": avg_lat,
                "speed_avg_tok_s": avg_spd,
            }

            if mode_name.startswith(("query_rewrite_inject", "tool_calling")):
                mode_summary["latency_round1_avg_s"] = round(sum(lat_r1_list) / len(lat_r1_list), 2) if lat_r1_list else 0
                mode_summary["latency_round2_avg_s"] = round(sum(lat_r2_list) / len(lat_r2_list), 2) if lat_r2_list else 0

            if mode_name.startswith("tool_calling"):
                mode_summary["tool_call_rate_pct"] = round((tool_calls_count / n_items) * 100, 1)
                mode_summary["tool_expected_rate_pct"] = round((tool_expected_count / n_items) * 100, 1)
                mode_summary["query_reformulation_rate_pct"] = round((query_reformulated_count / max(1, tool_calls_count)) * 100, 1)
                mode_summary["query_english_rate_pct"] = round((query_english_count / max(1, tool_calls_count)) * 100, 1)

            model_eval["modes"][mode_name] = {
                "summary": mode_summary,
                "items": evaluated_items
            }

            summary_rows.append({
                "model": model_key,
                "description": m_info.get("description", ""),
                "mode": mode_name,
                "total_items": n_items,
                "retrieval_recall_pct": recall_rate,
                "acerto_factual": avg_fac,
                "fidelidade_contexto": avg_fid,
                "qualidade_pt": avg_pt,
                "citacao_fonte": avg_cit,
                "score_global": avg_global,
                "pass_gate_pct": pass_rate,
                "latency_avg_s": avg_lat,
                "latency_r1_s": mode_summary.get("latency_round1_avg_s", "N/A"),
                "latency_r2_s": mode_summary.get("latency_round2_avg_s", "N/A"),
                "speed_tok_s": avg_spd,
                "tool_call_rate_pct": mode_summary.get("tool_call_rate_pct", "N/A"),
                "query_reformulation_rate_pct": mode_summary.get("query_reformulation_rate_pct", "N/A"),
            })

        evaluated_report["models"][model_key] = model_eval

    # Amostra de calibração com Claude CLI (50 itens aleatórios)
    if run_calibration and cli_tool == "vllm" and len(all_evaluated_items_flat) >= 50:
        logger.info("Executando amostra de calibração em 50 itens via claude -p...")
        random.seed(42)
        sample_subset = random.sample(all_evaluated_items_flat, 50)
        calib_records = []

        for item_record in sample_subset:
            claude_ev = call_llm_judge(
                question=item_record["question"],
                golden_answer=item_record["golden_answer"],
                model_response=item_record["response"],
                doc_expected=item_record.get("doc", ""),
                section_expected=item_record.get("section", ""),
                cli_tool="claude"
            )
            calib_records.append({
                "item_id": item_record.get("item_id"),
                "question": item_record["question"][:60],
                "vllm_metrics": item_record["metrics"],
                "claude_metrics": claude_ev
            })

        agreement_report = compute_calibration_agreement(calib_records)
        evaluated_report["calibration_agreement"] = agreement_report
        calib_file = output_eval_file.parent / "judge_calibration_50.json"
        with open(calib_file, "w", encoding="utf-8") as f:
            json.dump({"agreement": agreement_report, "sample_records": calib_records}, f, ensure_ascii=False, indent=2)
        logger.info("Calibração gravada em: %s", calib_file)
        print("=== CONCORDÂNCIA DE CALIBRAÇÃO (vLLM Qwen 27B vs Claude CLI) ===")
        for ax, vals in agreement_report.items():
            print(f"  {ax:<22}: Spearman={vals['spearman_corr']:.3f} | %+-1={vals['pct_within_1']}% | %Exato={vals['pct_exact']}%")

    # Salva relatório JSON completo
    output_eval_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_eval_file, "w", encoding="utf-8") as f:
        json.dump(evaluated_report, f, ensure_ascii=False, indent=2)
    logger.info("Relatório detalhado de avaliação gravado em: %s", output_eval_file)

    # Salva CSV consolidado
    output_csv_file.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(output_csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("Planilha resumo gravada em: %s", output_csv_file)

    print(HUMAN_REVIEW_DISCLAIMER)
    print("=== RESUMO DO BENCHMARK CONSOLIDADO (v2) ===")
    fmt_hdr = "{:<22} | {:<22} | {:<8} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6} | {:<7} | {:<7}"
    print(fmt_hdr.format("Modelo", "Modo", "Recall@3", "Fact", "Fidel", "PT", "Fonte", "Global", "Pass%", "Latência"))
    print("-" * 115)
    for r in summary_rows:
        print(fmt_hdr.format(
            r["model"][:22],
            r["mode"][:22],
            f"{r['retrieval_recall_pct']:.1f}%",
            f"{r['acerto_factual']:.2f}",
            f"{r['fidelidade_contexto']:.2f}",
            f"{r['qualidade_pt']:.2f}",
            f"{r['citacao_fonte']:.2f}",
            f"{r['score_global']:.2f}",
            f"{r['pass_gate_pct']:.1f}%",
            f"{r['latency_avg_s']:.2f}s"
        ))
    print("=" * 115)

    return evaluated_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliador de benchmark de SLMs via vLLM / LLM Juiz (CEMIG POC v2)")
    parser.add_argument(
        "--input",
        type=str,
        default="bench/results/v2/harness_results_v2.json",
        help="Arquivo JSON de resultados gerado pelo harness.py"
    )
    parser.add_argument(
        "--output-eval",
        type=str,
        default="bench/results/v2/judge_evaluations_v2.json",
        help="Arquivo JSON de saída com avaliações detalhadas por item"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="bench/results/v2/bench_summary_v2.csv",
        help="Arquivo CSV consolidado de métricas e gate pass rates"
    )
    parser.add_argument(
        "--cli-tool",
        type=str,
        default="vllm",
        choices=["vllm", "claude", "pi", "rule"],
        help="Backend do LLM juiz (vllm, claude, pi ou rule)"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=DEFAULT_VLLM_URL,
        help="URL base do endpoint vLLM OpenAI-compat"
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        default=DEFAULT_VLLM_MODEL,
        help="Nome do modelo no vLLM"
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Desativa a calibração com Claude CLI"
    )

    args = parser.parse_args()
    evaluate_benchmark_results(
        results_file=Path(args.input),
        output_eval_file=Path(args.output_eval),
        output_csv_file=Path(args.output_csv),
        cli_tool=args.cli_tool,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        run_calibration=not args.no_calibration
    )


if __name__ == "__main__":
    main()
