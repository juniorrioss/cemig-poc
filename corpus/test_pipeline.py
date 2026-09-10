"""
test_pipeline.py — Testes unitários e de integração para o pipeline de corpus do CEMIG POC.
Utiliza unittest (stdlib) sem dependências externas adicionais.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from corpus.build_index import (
    DEFAULT_DB_PATH,
    build_index,
    format_fts_query,
    init_database,
    populate_chunks,
    search_reference,
)
from corpus.chunk import build_chunk_text, chunk_items, count_tokens, format_title
from corpus.extract import DEFAULT_PDF_DIR, clean_line_text, extract_document


class TestCorpusPipeline(unittest.TestCase):
    """Bateria de testes cobrindo extração, chunking, indexação FTS5 e dataset de avaliação."""

    def test_text_cleaning_and_formatting(self):
        """Valida a limpeza de espaçamentos e títulos em português."""
        raw = "observando  -se   as   norma s  técnicas"
        cleaned = clean_line_text(raw)
        self.assertIn("observando-se", cleaned)

        title = format_title("10.2 - MEDIDAS DE PROTEÇÃO COLETIVA COM EPC E EPI")
        self.assertIn("Medidas de Proteção Coletiva com EPC e EPI", title)
        # Verifica preservação de siglas técnicas
        self.assertIn("EPC", title)
        self.assertIn("EPI", title)

    def test_chunk_header_and_token_count(self):
        """Valida que o header contextual é prefixado e a contagem de tokens funciona."""
        header = build_chunk_text(
            doc_id="nr-10",
            section="10.2.8",
            title="Medidas de Proteção Coletiva",
            body="As medidas de proteção coletiva compreendem a desenergização elétrica.",
        )
        self.assertTrue(header.startswith("NR-10 · 10.2.8 Medidas de Proteção Coletiva — "))
        tokens = count_tokens(header)
        self.assertGreater(tokens, 10)

    def test_extract_nr10_sample(self):
        """Valida a extração real a partir do PDF da NR-10."""
        pdf_path = DEFAULT_PDF_DIR / "nr-10.pdf"
        if not pdf_path.exists():
            self.skipTest(f"Arquivo {pdf_path} não encontrado no ambiente")

        doc = extract_document(pdf_path)
        self.assertEqual(doc.doc, "nr-10")
        self.assertIn("SEGURANÇA", doc.doc_title.upper())
        self.assertGreater(len(doc.items), 50)

        # Procura item 10.5.1 de desenergização
        sec_ids = [it.section for it in doc.items]
        has_10_5 = any("10.5" in s for s in sec_ids)
        self.assertTrue(has_10_5, "Deveria conter seções de 10.5")

    def test_fts5_unicode_diacritics_and_bm25(self):
        """Valida o funcionamento do FTS5 com unicode61 remove_diacritics 2."""
        con = sqlite3.connect(":memory:")
        init_database(con)

        sample_chunks = [
            {
                "doc": "nr-10",
                "section": "10.5.1",
                "title": "Segurança em Instalações Elétricas Desenergizadas",
                "page": 5,
                "text": "NR-10 · 10.5.1 Segurança em Instalações Elétricas Desenergizadas — A desenergização requer seccionamento e aterramento temporário.",
            },
            {
                "doc": "nr-35",
                "section": "35.2.1",
                "title": "Campo de Aplicação",
                "page": 1,
                "text": "NR-35 · 35.2.1 Campo de Aplicação — Trabalho em altura executado acima de dois metros com risco de queda.",
            }
        ]
        populate_chunks(con, sample_chunks)

        cur = con.cursor()
        # Busca sem acento para testar remoção de diacríticos: 'desenergizacao' -> deve achar 'desenergização'
        q = format_fts_query("desenergizacao temporario", mode="AND")
        cur.execute("""
            SELECT c.doc, c.section, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC;
        """, (q,))
        rows = cur.fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "nr-10")
        self.assertEqual(rows[0][1], "10.5.1")
        con.close()

    def test_qa_dataset_integrity(self):
        """Valida a consistência e o esquema do dataset de perguntas ouro."""
        qa_file = Path(__file__).parent / "qa_pairs.jsonl"
        self.assertTrue(qa_file.exists(), f"Arquivo {qa_file} deve existir")

        with open(qa_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Requisito do brief: 60 a 100 pares
        self.assertGreaterEqual(len(lines), 60, "Dataset deve conter pelo menos 60 pares")
        self.assertLessEqual(len(lines), 110, "Dataset deve conter até ~100 pares")

        required_keys = {"id", "doc", "section", "chunk_id", "question", "golden_answer", "query_terms"}
        valid_docs = {"nr-10", "nr-06", "nr-35", "nr-12", "nr-18"}

        for line in lines:
            item = json.loads(line)
            self.assertTrue(required_keys.issubset(set(item.keys())), f"Chaves faltando no item {item}")
            self.assertIn(item["doc"], valid_docs, f"Doc inválido: {item['doc']}")
            self.assertGreater(len(item["question"]), 10)
            self.assertGreater(len(item["golden_answer"]), 15)
            self.assertIsInstance(item["query_terms"], str)


if __name__ == "__main__":
    unittest.main()
