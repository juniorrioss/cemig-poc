"""
rewrite_refusalish.py — passo de higiene dos alvos GOLD: a régua honesta do
F0 marca como "recusa" frases tipo "não está/não é possível afirmar"; um alvo
gold com essa fraseologia ensinaria o modelo a responder de um jeito que a
régua pontua como recusa-indevida. Este passo re-pede ao 27B que REFRASEIE
(mesmos fatos, sem os gatilhos) só os alvos gold que disparam
looks_like_refusal. Alvos partial NÃO passam por aqui (o hedge é intencional).

Uso: classifier/.venv/bin/python bench/lfm3_f3/synth/rewrite_refusalish.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
REPO = F3.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "bench" / "lfm3_f0"))
from common_f3 import f2_synth_modules  # noqa: E402
from common_f0 import looks_like_refusal  # noqa: E402

cs = f2_synth_modules()
TRAIN = F3 / "data" / "train"

SYSTEM = (
    "Você revisa respostas de treino de um assistente de segurança do "
    "trabalho. Reescreva mantendo TODOS os fatos, números e citações. "
    "Responda sempre com um bloco JSON ao final.")

PROMPT = """RESPOSTA ORIGINAL:
{target}

Reescreva a resposta preservando TODOS os fatos, números, condições e a
citação da norma/item, mas SEM usar expressões que soem como recusa ou
indisponibilidade de informação, tais como: "não está", "não consta",
"não é possível afirmar", "não há", "não encontrei", "seria preciso
consultar", "o contexto não traz". Afirme o conteúdo diretamente (pode manter
um "Não," inicial se a resposta à pergunta for negativa, mas sem as expressões
acima). 2 a 5 frases.

Pense primeiro e emita APENAS o JSON final no formato
{{"resposta": "<texto completo reescrito>"}}"""


def main() -> None:
    targets = cs.read_jsonl(TRAIN / "targets.jsonl")
    todo = [t for t in targets
            if t["case"] == "gold" and looks_like_refusal(t["target"])]
    print(f"[rewrite] {len(todo)} alvos gold com fraseologia de recusa")
    ckpt = TRAIN / "rewrite.ckpt.json"

    def worker(t: dict) -> dict:
        last = ""
        for _ in range(3):
            v = cs.gen_json(PROMPT.format(target=t["target"]), system=SYSTEM,
                            temperature=0.3, max_tokens=2048)
            if isinstance(v, list):
                v = v[0] if v else {}
            resp = str(v.get("resposta", "")).strip() if isinstance(v, dict) else ""
            if len(resp) > 30 and "<texto" not in resp and not looks_like_refusal(resp):
                return {"target": resp}
            last = resp
        raise ValueError(f"reescrita ainda dispara recusa 3x: {last[:120]!r}")

    results = cs.parallel_map(worker, todo, max_workers=12,
                              checkpoint_path=ckpt, key_fn=lambda t: t["qa_id"])
    n_ok, n_err = 0, 0
    new = {t["qa_id"]: t for t in targets}
    for t in todo:
        v = results.get(t["qa_id"])
        if not v or "__error__" in v:
            n_err += 1
            continue
        new[t["qa_id"]] = {**t, "target": v["target"]}
        n_ok += 1
    cs.write_jsonl(TRAIN / "targets.jsonl", list(new.values()))
    print(f"[ok] {n_ok} reescritos, {n_err} erros (mantidos originais)")


if __name__ == "__main__":
    main()
