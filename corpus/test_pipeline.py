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

    def test_ingest_hf_parser_norma_and_manual(self):
        """Valida a separação entre norma vinculante e manual comentado no ingest_hf."""
        from corpus.ingest_hf import parse_manual_content, parse_norma_content

        sample_norma = """# NR-10 — SEGURANÇA EM ELETRICIDADE
## Norma
*Fonte: nr-10.pdf*
10.1 OBJETIVO
10.1.1 Esta norma estabelece condições mínimas de segurança em eletricidade.
10.2 MEDIDAS DE CONTROLE
10.2.8 MEDIDAS DE PROTEÇÃO COLETIVA
10.2.8.1 Em todos os serviços devem ser previstas medidas de proteção coletiva.
"""
        doc_norma = parse_norma_content("NR-10", 10, "Segurança Elétrica", sample_norma)
        self.assertEqual(doc_norma.doc, "nr-10")
        self.assertGreaterEqual(len(doc_norma.items), 2)
        # Verifica seção extraída
        sec_codes = [it.section for it in doc_norma.items]
        self.assertTrue(any("10.2" in s for s in sec_codes))

        sample_manual = """# Manual de Auxílio na Interpretação da NR-10 (2010)
10.2.8 - MEDIDAS DE PROTEÇÃO COLETIVA
Comentário
As medidas de proteção coletiva são de suma importância para evitar choques.
10.2.8.2 As medidas de proteção compreendem a desenergização.
"""
        doc_manual = parse_manual_content("NR-10", 10, "Segurança Elétrica", sample_manual)
        self.assertIsNotNone(doc_manual)
        self.assertEqual(doc_manual.doc, "nr-10-manual")
        self.assertIn("Manual", doc_manual.doc_title)
        self.assertGreaterEqual(len(doc_manual.items), 1)

    def test_qa_v2_dataset_integrity(self):
        """Valida o esquema e contagem do dataset consolidado v2."""
        qa_v2_file = Path(__file__).parent / "qa_pairs_v2.jsonl"
        self.assertTrue(qa_v2_file.exists(), f"Arquivo {qa_v2_file} deve existir")

        with open(qa_v2_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Requisito do brief: 101 v1 + ~40 novas (>= 140)
        self.assertGreaterEqual(len(lines), 140, "Dataset v2 deve conter pelo menos 140 perguntas")

        required_keys = {"id", "doc", "section", "chunk_id", "question", "golden_answer", "query_terms"}
        valid_docs = {"nr-10", "nr-06", "nr-35", "nr-12", "nr-18", "nr-01", "nr-33", "nr-16", "nr-26"}

        for line in lines:
            item = json.loads(line)
            self.assertTrue(required_keys.issubset(set(item.keys())), f"Chaves faltando no item {item}")
            self.assertIn(item["doc"], valid_docs, f"Doc inválido: {item['doc']}")
            self.assertGreater(len(item["question"]), 10)
            self.assertGreater(len(item["golden_answer"]), 15)

    def test_v2_indices_size_and_schema(self):
        """Valida se os três índices do v2 existem, possuem esquema correto e estão abaixo de 50 MB."""
        indices = [
            Path(__file__).parent / "index_hf_5nr.db",
            Path(__file__).parent / "index_hf_5nr_manual.db",
            Path(__file__).parent / "index_hf_36nr.db",
        ]
        for db_file in indices:
            self.assertTrue(db_file.exists(), f"Banco {db_file.name} deve existir")
            size_mb = db_file.stat().st_size / (1024 * 1024)
            self.assertLessEqual(size_mb, 50.0, f"{db_file.name} excede o limite de 50 MB (tamanho: {size_mb:.2f} MB)")

            con = sqlite3.connect(str(db_file))
            cur = con.cursor()
            cur.execute("SELECT count(*) FROM chunks;")
            count = cur.fetchone()[0]
            self.assertGreater(count, 300, f"{db_file.name} deve ter pelo menos 300 chunks")
            cur.execute("SELECT count(*) FROM chunks_fts;")
            fts_count = cur.fetchone()[0]
            self.assertEqual(count, fts_count, "chunks e chunks_fts devem estar sincronizados")
            con.close()


if __name__ == "__main__":
    unittest.main()
