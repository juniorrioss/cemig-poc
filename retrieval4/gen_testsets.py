#!/usr/bin/env python3
"""
gen_testsets.py — TRAVA 3 (paráfrase adversarial): dois conjuntos de teste do classificador.

Gera falas de campo rotuladas por NR em DOIS regimes:
  - samegen : MESMO gerador e MESMO estilo de prompt do treino (Qwen3.8-27B-FP8 via vLLM).
              É o teste OTIMISTA — mede o desempenho quando o teste "fala como o treino".
  - diffgen : gerador de FAMÍLIA DIFERENTE (Gemma-3-1B via llama-server local) + prompt
              radicalmente diferente + temperatura alta + persona distinta. É o teste
              HONESTO — mede generalização à FALA REAL, não ao gerador.

A DISTÂNCIA samegen−diffgen é o "tamanho da mascarada" (o quanto o classificador
aprendeu os tiques do gerador em vez do domínio). O capitão pediu exatamente isso.

Anti-contaminação: as falas de teste são geradas por NR de forma genérica (nunca veem
o holdout). NÃO entram em treino — são test-only. Procedência por linha (generator,
prompt_kind). Higiene: não-interativo, timeout, retry, checkpoint incremental, paralelo.

Uso (classifier/.venv):
  ../classifier/.venv/bin/python gen_testsets.py --which samegen --per-nr 12 --workers 8
  ../classifier/.venv/bin/python gen_testsets.py --which diffgen --per-nr 12 --workers 4

Comentários PT-BR, código em inglês. Procedência: task poc-retrieval-v4.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import requests
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "classifier"))

from nr_taxonomy import CLASSES, ELECTRICIAN_NRS, NR_ONELINE  # noqa: E402

# NRs cobertas no teste: as 9 do eletricista + as reservadas + algumas gerais p/ ruído.
TEST_NRS = sorted(set(ELECTRICIAN_NRS) | {"nr-33", "nr-16", "nr-26",
                                          "nr-15", "nr-31", "nr-23", "nr-17"})

# ------------------- SAMEGEN: espelha o estilo do treino (gen_labels) -------------------
SAME_SYSTEM = (
    "Você é um gerador de dados de treinamento. Produz falas curtas, coloquiais e "
    "realistas de trabalhadores brasileiros de campo (eletricistas, operários, "
    "instaladores), como se fossem transcrições de voz feitas no celular durante o "
    "serviço. As falas têm gíria regional, erros leves, hesitação ('ô', 'tipo', 'aí'), "
    "e NÃO usam jargão técnico de norma nem citam o número da NR."
)


def same_user(nr: str, n: int) -> str:
    return (
        f"Gere {n} falas DIFERENTES de trabalhador de campo cuja dúvida seja resolvida "
        f"pela norma sobre: {NR_ONELINE[nr]}.\n"
        "Regras:\n"
        "1. Cada fala é curta (1 a 3 frases), coloquial, como fala espontânea de voz.\n"
        "2. NUNCA cite o número da norma nem palavras como 'norma', 'NR'.\n"
        "3. NÃO copie os termos técnicos da descrição; use vocabulário de operário.\n"
        'Responda SOMENTE um array JSON de strings.'
    )


# ------------------- DIFFGEN: persona e formato RADICALMENTE diferentes ------------------
# Objetivo: forçar um "sotaque" de gerador distinto. Persona de 1ª pessoa, monólogo,
# sem instrução de formato coloquial "ô/tipo/aí" (esses são tiques do treino), pedindo
# variedade de REGISTRO: pergunta indireta, frase truncada, reclamação, desabafo.
DIFF_SYSTEM = (
    "Você improvisa a voz de UM trabalhador brasileiro específico falando sozinho, sem "
    "roteiro. Escreva como a pessoa realmente fala: às vezes uma frase pela metade, às "
    "vezes um desabafo, às vezes uma pergunta enrolada. Cada pessoa tem um jeito próprio."
)


def diff_user(nr: str, n: int, persona: str, mode: str) -> str:
    modes = {
        "indireta": "faça a dúvida de forma INDIRETA (rodeia o assunto, não pergunta direto)",
        "truncada": "deixe frases pela METADE, cortadas, como fala apressada no rádio",
        "desabafo": "seja um DESABAFO/reclamação sobre a situação, sem pedir ajuda claramente",
        "narrativa": "conte o que ESTÁ acontecendo agora, em tempo real, sem perguntar nada",
    }
    return (
        f"A pessoa é: {persona}. A situação envolve o tema: {NR_ONELINE[nr]}.\n"
        f"Gere {n} falas curtas e MUITO diferentes entre si; em cada uma, {modes[mode]}.\n"
        "Nunca escreva 'norma', 'NR', nem número de norma. Nunca use termos técnicos de "
        "engenharia; use as palavras do dia a dia dessa pessoa. Evite começar com 'ô' ou 'ei'.\n"
        'Responda SOMENTE um array JSON de strings (aspas duplas).'
    )


PERSONAS = [
    "um eletricista de 55 anos do interior de Minas, fala mansa e enrolada",
    "uma operária jovem de obra, direta e impaciente",
    "um encarregado nervoso falando no rádio comunicador",
    "um ajudante novato inseguro que gagueja",
    "um técnico experiente meio cínico, reclamão",
    "um trabalhador nordestino com gíria regional forte",
]
MODES = ["indireta", "truncada", "desabafo", "narrativa"]


@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1.5, min=2, max=20),
       retry=retry_if_exception_type((requests.exceptions.RequestException,)),
       reraise=True)
def call_llm(base_url: str, model: str, system: str, user: str,
             temperature: float, timeout: float = 90.0) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature, "top_p": 0.98, "max_tokens": 900,
    }
    # thinking-off só p/ o vLLM Qwen (Gemma ignora o campo)
    if "8005" in base_url:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content", "") or ""


def extract_array(raw: str) -> List[str]:
    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if m:
            text = m.group(1)
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e != -1 and e > s:
        text = text[s:e + 1]
    try:
        parsed = json.loads(text)
    except Exception:
        # fallback: linhas com hífen/aspas
        lines = [ln.strip(' -*"\t') for ln in raw.splitlines() if len(ln.strip()) > 8]
        return [ln for ln in lines if not ln.startswith(("{", "[", "```"))][:12]
    out = []
    if isinstance(parsed, list):
        for it in parsed:
            if isinstance(it, str) and it.strip():
                out.append(it.strip())
    return out


def clean(utt: str) -> str:
    return re.sub(r"\s+", " ", utt).strip()


def valid(utt: str) -> bool:
    # descarta vazamento de número de norma e falas curtas demais/longas demais
    if re.search(r"\bnr[\s-]?\d", utt.lower()):
        return False
    if "norma" in utt.lower():
        return False
    return 8 <= len(utt) <= 400


def build_batches(which: str, per_nr: int) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    per_call = 6
    for nr in TEST_NRS:
        left = per_nr
        while left > 0:
            k = min(per_call, left)
            if which == "samegen":
                batches.append({"nr": nr, "n": k})
            else:
                batches.append({"nr": nr, "n": k,
                                "persona": random.choice(PERSONAS),
                                "mode": random.choice(MODES)})
            left -= k
    random.shuffle(batches)
    return batches


def load_done(path: Path) -> set:
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                seen.add(json.loads(line)["text"].strip().lower())
            except Exception:
                pass
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["samegen", "diffgen"], required=True)
    ap.add_argument("--per-nr", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--samegen-url", default="http://10.100.0.111:8005/v1")
    ap.add_argument("--samegen-model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--diffgen-url", default="http://127.0.0.1:8091/v1")
    ap.add_argument("--diffgen-model", default="google_gemma-3-1b-it-Q4_K_M.gguf")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out) if args.out else _HERE / "data" / f"test_{args.which}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.which == "samegen":
        base_url, model, gen_name = args.samegen_url, args.samegen_model, args.samegen_model
        temperature = 0.9
    else:
        base_url, model, gen_name = args.diffgen_url, args.diffgen_model, args.diffgen_model
        temperature = 1.15  # alta p/ maximizar variação de registro

    batches = build_batches(args.which, args.per_nr)
    seen = load_done(out_path)
    print(f"[{args.which}] gerador={gen_name} lotes={len(batches)} já={len(seen)}")

    if not seen and not out_path.exists():
        out_path.write_text("# " + json.dumps({
            "artifact": f"classifier_testset_{args.which}",
            "task": "poc-retrieval-v4", "generator": gen_name, "endpoint": base_url,
            "temperature": temperature, "purpose": "test_only_never_train",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, ensure_ascii=False) + "\n", encoding="utf-8")

    def work(b: Dict[str, Any]) -> List[Dict[str, Any]]:
        if args.which == "samegen":
            sys_p, usr_p, pk = SAME_SYSTEM, same_user(b["nr"], b["n"]), f"same:{b['nr']}"
        else:
            sys_p = DIFF_SYSTEM
            usr_p = diff_user(b["nr"], b["n"], b["persona"], b["mode"])
            pk = f"diff:{b['nr']}:{b['mode']}"
        try:
            raw = call_llm(base_url, model, sys_p, usr_p, temperature)
        except Exception:
            return []
        recs = []
        for utt in extract_array(raw):
            utt = clean(utt)
            if valid(utt):
                recs.append({"text": utt, "primary": b["nr"], "gold": b["nr"],
                             "kind": args.which, "prompt_kind": pk, "generator": gen_name})
        return recs

    fh = out_path.open("a", encoding="utf-8")
    t0 = time.time()
    kept = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, b): b for b in batches}
        for i, fut in enumerate(as_completed(futs), 1):
            for r in fut.result():
                key = r["text"].strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
            if i % 10 == 0:
                fh.flush()
                print(f"  {i}/{len(batches)} lotes | {kept} falas | {time.time()-t0:.0f}s", flush=True)
    fh.flush()
    fh.close()
    print(f"[{args.which}] concluído: {kept} falas em {out_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
