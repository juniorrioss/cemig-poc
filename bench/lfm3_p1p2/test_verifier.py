"""
test_verifier.py — testes unitários do CONTRATO do reward (verifier.py).
Publicados ANTES do treino P2. Rodar: python3 bench/lfm3_p1p2/test_verifier.py
(stdlib unittest; só depende de bench/regua e bench/lfm3_f0).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verifier as V  # noqa: E402

CTX = ("[1] (nr-18 - 18.8 - Escadas, Rampas e Passarelas):\nNR-18 · 18.8 — 18.8.6.1 "
       "As escadas de uso coletivo devem: c) ter largura mínima de 0,8 m (oitenta "
       "centímetros); d) ter altura uniforme entre os degraus de, no máximo, 0,2 m "
       "(vinte centímetros); e) ter patamar intermediário, no máximo, a cada 2,9 m "
       "(dois metros e noventa centímetros) de altura.")
ITEM = {"expect": "answer", "question": "Qual a largura mínima da escada coletiva?",
        "context": CTX, "gold_items": ["18.8.6.1"],
        "facts": ["largura mínima de 0,8 m", "escadas de uso coletivo"]}


class TestNumbers(unittest.TestCase):
    def test_decimal_comma_and_dot(self):
        self.assertEqual(V.unsupported_numbers("largura de 0.8 m", CTX), [])
        self.assertEqual(V.unsupported_numbers("largura de 0,80 m", CTX), [])

    def test_word_number_in_context(self):
        # "oitenta centímetros" no contexto autoriza "80 cm"
        self.assertEqual(V.unsupported_numbers("80 cm", CTX), [])
        self.assertEqual(V.unsupported_numbers("a cada 2,9 m (90)", CTX), [])

    def test_invented_number(self):
        self.assertEqual(V.unsupported_numbers("largura de 1,2 m", CTX), ["1,2"])

    def test_item_parent_allowed(self):
        self.assertEqual(V.unsupported_numbers("ver item 18.8", CTX), [])

    def test_question_numbers_allowed(self):
        self.assertEqual(V.unsupported_numbers("para 13,8 kV", "nada", "rede de 13,8 kV?"), [])

    def test_thousands(self):
        self.assertEqual(V.unsupported_numbers("1000 trabalhadores", "até 1.000 empregados"), [])

    def test_word_compound(self):
        self.assertIn("120", V.word_numbers("cento e vinte dias"))
        self.assertIn("2000", V.word_numbers("dois mil"))


class TestCitation(unittest.TestCase):
    def test_exact_and_child(self):
        self.assertTrue(V.cites_gold_item("conforme o item 18.8.6.1", ["18.8.6.1"]))
        self.assertTrue(V.cites_gold_item("item 18.8.6.1.2", ["18.8.6.1"]))

    def test_parent_is_not_enough(self):
        self.assertFalse(V.cites_gold_item("item 18.8", ["18.8.6.1"]))

    def test_wrong_item(self):
        self.assertFalse(V.cites_gold_item("item 18.8.2", ["18.8.6.1"]))

    def test_no_code_dispensed(self):
        self.assertTrue(V.cites_gold_item("qualquer coisa", []))


class TestReward(unittest.TestCase):
    def test_good_answer(self):
        a = ("Pela NR-18, item 18.8.6.1, alínea c, as escadas de uso coletivo devem ter "
             "largura mínima de 0,8 m.")
        v = V.verify(a, ITEM)
        self.assertEqual(v.reward, 1, v)

    def test_thinking_block_stripped(self):
        a = ("<think></think>Pela NR-18, item 18.8.6.1, as escadas de uso coletivo devem "
             "ter largura mínima de 0,8 m.<|im_end|>")
        self.assertEqual(V.verify(a, ITEM).reward, 1)

    def test_no_citation(self):
        a = "As escadas de uso coletivo devem ter largura mínima de 0,8 m."
        self.assertEqual(V.verify(a, ITEM).reason, "no_gold_item")

    def test_invented_number_fails(self):
        a = ("Pela NR-18, item 18.8.6.1, escadas de uso coletivo: largura mínima de 0,8 m "
             "e corrimão a 1,1 m.")
        self.assertEqual(V.verify(a, ITEM).reason, "number_outside_context")

    def test_low_coverage(self):
        a = "Segundo a NR-18, item 18.8.6.1, siga a norma."
        self.assertEqual(V.verify(a, ITEM).reason, "low_coverage")

    def test_undue_refusal(self):
        a = "Não encontrei essa informação nas normas recuperadas (item 18.8.6.1)."
        self.assertEqual(V.verify(a, ITEM).reward, 0)

    def test_truncated(self):
        a = "Pela NR-18, item 18.8.6.1, escadas de uso coletivo largura mínima de 0,8 m"
        self.assertEqual(V.verify(a, ITEM, stopped=False).reason, "truncated")

    def test_refusal_rewarded_when_gold_absent(self):
        it = dict(ITEM, expect="refuse")
        a = ("Não encontrei essa informação nas normas recuperadas. Consulte a NR-18 "
             "ou o SESMT.")
        self.assertEqual(V.verify(a, it).reward, 1)

    def test_answer_rewarded_zero_when_gold_absent(self):
        it = dict(ITEM, expect="refuse")
        a = "Pela NR-18, item 18.8.6.1, a largura mínima é 0,8 m."
        self.assertEqual(V.verify(a, it).reward, 0)

    def test_v2_hedged_answer_is_not_refusal(self):
        # contrato v2: responder e fechar com ressalva NÃO é recusa (reward 0)
        it = dict(ITEM, expect="refuse")
        a = ("Sim, a largura mínima é 0,8 m. A norma não traz outra exigência além dessa.")
        self.assertEqual(V.verify(a, it).reason, "hedged_refusal")

    def test_v2_refusal_must_be_upfront(self):
        it = dict(ITEM, expect="refuse")
        a = ("Pela NR-18 a escada deve ser dimensionada pelo fluxo de trabalhadores e ter "
             "proteção contra quedas, além de patamares e degraus uniformes conforme o "
             "projeto e as cargas previstas para a obra inteira. Os trechos recuperados não "
             "cobrem o resto.")
        self.assertGreater(len(a), V.REFUSAL_HEAD_CHARS)
        self.assertEqual(V.verify(a, it).reward, 0)

    def test_v2_refusal_templates(self):
        it = dict(ITEM, expect="refuse")
        for a in ("Os trechos de norma recuperados não cobrem esse ponto. Verifique a NR-32.",
                  "Isso não está nos trechos das normas que recuperei. Consulte o SESMT.",
                  "Não encontrei essa informação nas normas recuperadas.",
                  "O contexto fornecido não traz essa informação."):
            self.assertEqual(V.verify(a, it).reward, 1, a)

    def test_spurious_tool_call(self):
        a = '<tool_call>\n<function=buscar_norma>\n</function>\n</tool_call>'
        self.assertEqual(V.verify(a, ITEM).reason, "spurious_tool_call")
        self.assertEqual(V.verify(a, dict(ITEM, expect="no_tool")).reward, 0)

    def test_no_tool_family(self):
        self.assertEqual(V.verify("De nada!", dict(ITEM, expect="no_tool")).reward, 1)

    def test_partial_skips_citation(self):
        it = dict(ITEM, expect="partial")
        a = "As escadas de uso coletivo devem ter largura mínima de 0,8 m; o resto não consta."
        self.assertEqual(V.verify(a, it).reward, 1)

    def test_threshold_x(self):
        a = "Pela NR-18, item 18.8.6.1, a largura mínima de 0,8 m."
        self.assertEqual(V.verify(a, ITEM, x=0.5).reward, 1)
        self.assertEqual(V.verify(a, ITEM, x=1.0).reward, 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
