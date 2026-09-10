"""
judge.py — Avaliação automatizada de respostas de SLMs em 4 eixos via LLM Juiz CLI.

Eixos de Avaliação (escala 1 a 5):
1. Acerto Factual (Accuracy)
2. Fidelidade ao Contexto / Disse não-sei quando devia (Faithfulness & Refusal)
3. Qualidade do Português Brasileiro (Fluência técnica)
4. Citação de Fonte Técnica (Norma e item/seção explícitos)

AVISO OBRIGATÓRIO: O LLM Juiz é estritamente auxiliar. A validação humana
das primeiras 20 questões é mandatória para decisões de homologação em campo.
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("bench.judge")

HUMAN_REVIEW_DISCLAIMER = """
********************************************************************************
AVISO OBRIGATÓRIO DE GOVERNANÇA:
A avaliação automatizada realizada pelo LLM Juiz é uma ferramenta de triagem AUXILIAR.
A leitura humana individual das 20 primeiras questões manuais (smoke test) é
MANDATÓRIA antes de qualquer homologação ou decisão de embarque no app de campo.
********************************************************************************
"""


def parse_llm_json_response(raw_text: str) -> Dict[str, Any]:
    """Extrai e valida o JSON estruturado gerado pelo LLM Juiz."""
    text = raw_text.strip()
    # Tenta extrair bloco ```json ... ``` se presente
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m2 = re.search(r"(\{.*\})", text, re.DOTALL)
        if m2:
            text = m2.group(1)

    try:
        data = json.loads(text)
        return {
            "acerto_factual": max(1, min(5, int(data.get("acerto_factual", 3)))),
            "fidelidade_contexto": max(1, min(5, int(data.get("fidelidade_contexto", 3)))),
            "qualidade_pt": max(1, min(5, int(data.get("qualidade_pt", 3)))),
            "citacao_fonte": max(1, min(5, int(data.get("citacao_fonte", 1)))),
            "justificativa": str(data.get("justificativa", "")).strip()
        }
    except Exception as e:
        logger.warning("Falha ao decodificar JSON do juiz: %s | Texto: %s", e, text[:120])
        return fallback_rule_judge("", "", "")


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
    cli_tool: str = "claude"
) -> Dict[str, Any]:
    """Invoca o LLM Juiz via CLI do sistema (padrão: claude -p)."""
    if not model_response or not model_response.strip():
        return {
            "acerto_factual": 1,
            "fidelidade_contexto": 1,
            "qualidade_pt": 1,
            "citacao_fonte": 1,
            "justificativa": "Resposta vazia ou truncada."
        }

    prompt = f"""Você é um auditor técnico especialista em segurança do trabalho e Normas Regulamentadoras brasileiras (NR-10, NR-06).
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

    if cli_tool == "claude":
        try:
            cmd = ["claude", "-p", prompt, "--bare", "--tools", ""]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if proc.returncode == 0 and proc.stdout.strip():
                return parse_llm_json_response(proc.stdout)
        except Exception as e:
            logger.warning("Falha ao chamar claude CLI: %s. Usando fallback.", e)

    elif cli_tool == "pi":
        try:
            cmd = ["pi", "-p", "--no-tools", prompt]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if proc.returncode == 0 and proc.stdout.strip():
                return parse_llm_json_response(proc.stdout)
        except Exception as e:
            logger.warning("Falha ao chamar pi CLI: %s. Usando fallback.", e)

    # Fallback heurístico
    return fallback_rule_judge(golden_answer, model_response, doc_expected)


def evaluate_benchmark_results(
    results_file: Path,
    output_eval_file: Path,
    output_csv_file: Path,
    cli_tool: str = "claude"
) -> Dict[str, Any]:
    """Avalia todas as respostas do benchmark e gera relatório e CSV consolidado."""
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_data = data.get("models", {})
    evaluated_report: Dict[str, Any] = {
        "metadata": {
            **data.get("metadata", {}),
            "judge_tool": cli_tool,
            "disclaimer": HUMAN_REVIEW_DISCLAIMER.strip(),
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "models": {}
    }

    summary_rows = []

    for model_key, m_info in models_data.items():
        logger.info("Avaliando modelo: %s...", model_key)
        results = m_info.get("results", {})
        model_eval: Dict[str, Any] = {
            "description": m_info.get("description", ""),
            "modes": {}
        }

        for mode_name in ("classic_rag", "tool_calling"):
            items = results.get(mode_name, [])
            if not items:
                continue

            evaluated_items = []
            fac_scores, fid_scores, pt_scores, cit_scores, latencies, tok_speeds = [], [], [], [], [], []
            tool_calls_count = 0
            tool_expected_count = 0
            query_reformulated_count = 0
            query_english_count = 0

            def _eval_single_item(it):
                resp_text = it.get("response", "")
                question = it.get("question", "")
                golden = it.get("golden_answer", "")
                doc_exp = it.get("doc", "")
                sec_exp = it.get("section", "")

                ev = call_llm_judge(
                    question=question,
                    golden_answer=golden,
                    model_response=resp_text,
                    doc_expected=doc_exp,
                    section_expected=sec_exp,
                    cli_tool=cli_tool
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
                    "item_id": it.get("item_id"),
                    "question": question,
                    "response": resp_text,
                    "golden_answer": golden,
                    "metrics": ev,
                    "score_global": score_global,
                    "pass_gate": pass_gate,
                    "latency_s": it.get("latency_s") or it.get("latency_total_s", 0),
                    "predicted_tok_per_s": it.get("predicted_tok_per_s", 0)
                }

                if mode_name == "tool_calling":
                    ev_record["called_tool"] = it.get("called_tool", False)
                    ev_record["tool_query"] = it.get("tool_query", "")
                    ev_record["tool_query_reformulated"] = it.get("tool_query_reformulated", False)
                    ev_record["tool_query_is_english"] = it.get("tool_query_is_english", False)

                return it.get("item_id"), ev_record, ev

            # Avaliação concorrente para acelerar o processo mantendo a precisão
            item_order = {it.get("item_id"): i for i, it in enumerate(items)}
            evaluated_items_dict = {}
            ev_list = []

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(_eval_single_item, it) for it in items]
                for fut in as_completed(futures):
                    item_id, ev_record, ev = fut.result()
                    evaluated_items_dict[item_id] = ev_record
                    ev_list.append((item_id, ev, ev_record))

            # Ordena os itens para manter a ordem original
            sorted_item_ids = sorted(evaluated_items_dict.keys(), key=lambda x: item_order.get(x, 0))
            evaluated_items = [evaluated_items_dict[x] for x in sorted_item_ids]

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
                if mode_name == "tool_calling":
                    if ev_record.get("called_tool"):
                        tool_calls_count += 1
                    # Compara com requires_tool original
                    orig_item = next((x for x in items if x.get("item_id") == it_id), {})
                    if ev_record.get("called_tool") == orig_item.get("requires_tool", True):
                        tool_expected_count += 1
                    if ev_record.get("tool_query_reformulated"):
                        query_reformulated_count += 1
                    if ev_record.get("tool_query_is_english"):
                        query_english_count += 1

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

            mode_summary = {
                "total_items": n_items,
                "acerto_factual_avg": avg_fac,
                "fidelidade_contexto_avg": avg_fid,
                "qualidade_pt_avg": avg_pt,
                "citacao_fonte_avg": avg_cit,
                "score_global_avg": avg_global,
                "gate_pass_rate_pct": pass_rate,
                "latency_avg_s": avg_lat,
                "speed_avg_tok_s": avg_spd,
            }

            if mode_name == "tool_calling":
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
                "acerto_factual": avg_fac,
                "fidelidade_contexto": avg_fid,
                "qualidade_pt": avg_pt,
                "citacao_fonte": avg_cit,
                "score_global": avg_global,
                "pass_gate_pct": pass_rate,
                "latency_avg_s": avg_lat,
                "speed_tok_s": avg_spd,
                "tool_call_rate_pct": mode_summary.get("tool_call_rate_pct", "N/A"),
                "tool_expected_rate_pct": mode_summary.get("tool_expected_rate_pct", "N/A"),
                "query_reformulation_rate_pct": mode_summary.get("query_reformulation_rate_pct", "N/A"),
                "query_english_rate_pct": mode_summary.get("query_english_rate_pct", "N/A"),
            })

        evaluated_report["models"][model_key] = model_eval

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
    print("=== RESUMO DO BENCHMARK CONSOLIDADO ===")
    fmt_hdr = "{:<14} | {:<12} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6} | {:<8} | {:<8}"
    print(fmt_hdr.format("Modelo", "Modo", "Factual", "Fidel", "PT", "Fonte", "Global", "Pass%", "Latência"))
    print("-" * 88)
    for r in summary_rows:
        print(fmt_hdr.format(
            r["model"],
            r["mode"],
            f"{r['acerto_factual']:.2f}",
            f"{r['fidelidade_contexto']:.2f}",
            f"{r['qualidade_pt']:.2f}",
            f"{r['citacao_fonte']:.2f}",
            f"{r['score_global']:.2f}",
            f"{r['pass_gate_pct']:.1f}%",
            f"{r['latency_avg_s']:.2f}s"
        ))
    print("=" * 88)

    return evaluated_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliador de benchmark de SLMs via LLM Juiz CLI (CEMIG POC)")
    parser.add_argument(
        "--input",
        type=str,
        default="bench/results/harness_results.json",
        help="Arquivo JSON de resultados gerado pelo harness.py"
    )
    parser.add_argument(
        "--output-eval",
        type=str,
        default="bench/results/judge_evaluations.json",
        help="Arquivo JSON de saída com avaliações detalhadas por item"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="bench/results/bench_summary.csv",
        help="Arquivo CSV consolidado de métricas e gate pass rates"
    )
    parser.add_argument(
        "--cli-tool",
        type=str,
        default="claude",
        choices=["claude", "pi", "rule"],
        help="Ferramenta CLI do LLM juiz (claude, pi ou rule para heurística offline)"
    )

    args = parser.parse_args()
    evaluate_benchmark_results(
        results_file=Path(args.input),
        output_eval_file=Path(args.output_eval),
        output_csv_file=Path(args.output_csv),
        cli_tool=args.cli_tool
    )


if __name__ == "__main__":
    main()
