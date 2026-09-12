"""
test_tautology.py — Valida o detector de tautologia e o verificador de cobertura
nos exemplos citados pelo capitão. Roda offline (sem juiz). Não-interativo.

Executar: python3 bench/regua/test_tautology.py
"""

from __future__ import annotations

import sys

from ruler import deterministic_pass


def _check(name: str, cond: bool) -> bool:
    status = "OK " if cond else "FALHOU"
    print(f"  [{status}] {name}")
    return cond


def main() -> int:
    all_ok = True

    # ---- Caso 1: tautologia flagrante citada pelo capitão ----
    # Pergunta: EPI para trabalho em altura acima de 2 metros?
    # Resposta ruim: "O EPI para trabalho em altura acima de 2 metros é obrigatório."
    q1 = "Qual o EPI para trabalho em altura acima de 2 metros?"
    a1 = "O EPI para trabalho em altura acima de 2 metros é obrigatório."
    facts1 = ["cinturão tipo paraquedista", "talabarte", "trava-quedas", "ponto de ancoragem"]
    r1 = deterministic_pass(q1, a1, facts1)
    print("Caso 1 — tautologia do capitão (deve REPROVAR):")
    all_ok &= _check("detecta tautologia", r1.tautology)
    all_ok &= _check("não aprovada", not r1.approved)
    all_ok &= _check("cobertura zero de fatos", r1.coverage == 0.0)
    print(f"    detalhe: cov={r1.coverage} q_overlap={r1.detail.get('q_overlap')} "
          f"present={r1.facts_present}")

    # ---- Caso 2: resposta boa (contém os fatos) ----
    q2 = q1
    a2 = ("Para trabalho em altura acima de 2 metros use cinturão de segurança tipo "
          "paraquedista com talabarte e trava-quedas, sempre conectado a um ponto de "
          "ancoragem resistente, conforme a NR-35.")
    r2 = deterministic_pass(q2, a2, facts1)
    print("Caso 2 — resposta correta (deve APROVAR):")
    all_ok &= _check("não é tautologia", not r2.tautology)
    all_ok &= _check("cobertura alta", r2.coverage >= 0.75)
    all_ok &= _check("aprovada", r2.approved)
    print(f"    detalhe: cov={r2.coverage} present={r2.facts_present}")

    # ---- Caso 3: reordenação pura da pergunta, tema proteção coletiva ----
    q3 = "Posso trabalhar só com a luva de borracha ou tem outro jeito antes do EPI?"
    a3 = "Você pode trabalhar com a luva de borracha antes de partir para o EPI."
    facts3 = ["medidas de proteção coletiva", "desenergização elétrica", "tensão de segurança"]
    r3 = deterministic_pass(q3, a3, facts3)
    print("Caso 3 — reordenação sem fato novo (deve REPROVAR):")
    all_ok &= _check("detecta tautologia", r3.tautology)
    all_ok &= _check("não aprovada", not r3.approved)
    print(f"    detalhe: cov={r3.coverage} q_overlap={r3.detail.get('q_overlap')}")

    # ---- Caso 4: resposta vazia ----
    r4 = deterministic_pass(q1, "", facts1)
    print("Caso 4 — resposta vazia (deve REPROVAR):")
    all_ok &= _check("marcada vazia", r4.empty)
    all_ok &= _check("não aprovada", not r4.approved)

    # ---- Caso 5: resposta parcial legítima (1 de 4 fatos) NÃO é tautologia ----
    q5 = q1
    a5 = "Use cinturão tipo paraquedista para trabalho em altura."
    r5 = deterministic_pass(q5, a5, facts1)
    print("Caso 5 — parcial legítima (1 fato novo, não é tautologia):")
    all_ok &= _check("não é tautologia (tem fato novo)", not r5.tautology)
    print(f"    detalhe: cov={r5.coverage} present={r5.facts_present}")

    print()
    if all_ok:
        print("TODOS OS TESTES PASSARAM.")
        return 0
    print("HÁ FALHAS NO DETECTOR.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
