"""
chunk.py — Segmentação de Normas Regulamentadoras em chunks de ~200-400 tokens.

Alinha os blocos às seções naturais da norma, preservando metadados de origem
(doc, seção, título, página) e incluindo cabeçalho contextual no próprio texto:
"NR-10 · 10.2.8 Medidas de proteção coletiva — ..."
"""

import argparse
import glob
import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from corpus.extract import ExtractedDocument, ExtractedItem, extract_all_documents


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Representa um chunk pronto para indexação FTS5 e consumo pelo retriever/SLM."""
    id: int                # Identificador único sequencial (chave primária)
    doc: str               # Documento de origem, ex.: 'nr-10'
    section: str           # Código da seção/item, ex.: '10.2.8' ou '10.2.8.2'
    title: str             # Título formatado da seção
    page: int              # Página aproximada no PDF de origem
    text: str              # Texto integral com cabeçalho contextual
    token_count: int       # Quantidade estimada de tokens


def count_tokens(text: str) -> int:
    """Estima o número de tokens no padrão de LLMs para texto em português."""
    return len(re.findall(r'\w+|[^\w\s]', text))


def format_title(raw_title: str) -> str:
    """Formata títulos em Title Case em português, mantendo siglas técnicas oficiais em maiúsculas."""
    raw_title = raw_title.strip("- ")
    raw_title = re.sub(r'^\d+(\.\d+)*\s*[-–.]?\s*', '', raw_title)

    words = raw_title.split()
    if not words:
        return "Disposições Gerais"

    acronyms = {
        "EPI", "EPC", "SEP", "NR", "CA", "AT", "BT", "EBT", "DR",
        "SPQ", "AR", "PT", "PGR", "DOU", "MTE", "CIPA", "LOTO", "ABNT", "NBR"
    }
    lower_words = {
        "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
        "e", "com", "por", "para", "ao", "à", "às", "aos"
    }

    clean_words = []
    for i, w in enumerate(words):
        w_clean = re.sub(r'[^\w]', '', w)
        if w_clean.upper() in acronyms:
            clean_words.append(w.upper())
        elif i > 0 and w_clean.lower() in lower_words:
            clean_words.append(w.lower())
        else:
            clean_words.append(w.capitalize())

    return " ".join(clean_words)


def build_chunk_text(doc_id: str, section: str, title: str, body: str) -> str:
    """Monta o cabeçalho contextual padronizado na cabeça do texto do chunk."""
    doc_label = doc_id.upper()
    return f"{doc_label} · {section} {title} — {body.strip()}"


def chunk_items(
    items: List[Dict[str, Any] | ExtractedItem],
    target_min: int = 180,
    target_max: int = 380,
    hard_max: int = 450,
) -> List[Dict[str, Any]]:
    """Agrupa itens estruturados em blocos coerentes de ~200-400 tokens."""
    # Converte ExtractedItem para dict se necessário
    normalized_items: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, ExtractedItem):
            normalized_items.append(asdict(it))
        else:
            normalized_items.append(it)

    raw_chunks: List[Dict[str, Any]] = []
    buffer: List[Dict[str, Any]] = []
    buffer_tok: int = 0

    def flush(buf: List[Dict[str, Any]]) -> None:
        if not buf:
            return
        first = buf[0]
        doc = first["doc"]

        # Identifica a seção mais representativa (prioriza parent_section se compartilhado)
        parents = [it.get("parent_section") for it in buf if it.get("parent_section")]
        nr_root = f"{doc.replace('nr-', '').lstrip('0')}.0"
        if parents and parents[0] and parents[0] != nr_root:
            sec = parents[0]
        else:
            sec = first["section"]

        title = format_title(first["title"])
        page = first["page"]
        body = " ".join(it["text"].strip() for it in buf if it["text"].strip())
        full_text = build_chunk_text(doc, sec, title, body)

        raw_chunks.append({
            "doc": doc,
            "section": sec,
            "title": title,
            "page": page,
            "text": full_text,
            "token_count": count_tokens(full_text),
        })

    for item in normalized_items:
        it_text = item["text"]
        it_tok = count_tokens(it_text)

        # Trata itens excessivamente longos (> hard_max) particionando por sentenças
        if it_tok > hard_max:
            if buffer:
                flush(buffer)
                buffer = []
                buffer_tok = 0

            sentences = re.split(r'(?<=[.;])\s+', it_text)
            sub_buf: List[str] = []
            sub_tok: int = 0
            part: int = 1

            for s in sentences:
                s_tok = count_tokens(s)
                if sub_buf and (sub_tok + s_tok > target_max):
                    title_formatted = format_title(item["title"])
                    part_title = f"{title_formatted} (parte {part})"
                    body = " ".join(sub_buf)
                    chunk_str = build_chunk_text(item["doc"], item["section"], part_title, body)
                    raw_chunks.append({
                        "doc": item["doc"],
                        "section": item["section"],
                        "title": part_title,
                        "page": item["page"],
                        "text": chunk_str,
                        "token_count": count_tokens(chunk_str),
                    })
                    part += 1
                    sub_buf = [s]
                    sub_tok = s_tok
                else:
                    sub_buf.append(s)
                    sub_tok += s_tok

            if sub_buf:
                title_formatted = format_title(item["title"])
                part_title = f"{title_formatted} (parte {part})" if part > 1 else title_formatted
                body = " ".join(sub_buf)
                chunk_str = build_chunk_text(item["doc"], item["section"], part_title, body)
                raw_chunks.append({
                    "doc": item["doc"],
                    "section": item["section"],
                    "title": part_title,
                    "page": item["page"],
                    "text": chunk_str,
                    "token_count": count_tokens(chunk_str),
                })
            continue

        # Verifica se o item pertence ao mesmo grupo de seção
        same_major = False
        if buffer:
            b_sec = buffer[0].get("parent_section") or buffer[0]["section"]
            i_sec = item.get("parent_section") or item["section"]
            b_major = b_sec.split(".")[0]
            i_major = i_sec.split(".")[0]
            same_major = (b_major == i_major)

        if buffer and (buffer_tok + it_tok > target_max or (not same_major and buffer_tok >= target_min)):
            flush(buffer)
            buffer = [item]
            buffer_tok = it_tok
        else:
            buffer.append(item)
            buffer_tok += it_tok

    if buffer:
        flush(buffer)

    # Segunda passagem: mescla chunks resíduo pequenos (<160 tokens) com chunk anterior se couber
    refined_chunks: List[Dict[str, Any]] = []
    for c in raw_chunks:
        if refined_chunks and refined_chunks[-1]["doc"] == c["doc"] and (refined_chunks[-1]["token_count"] + c["token_count"] <= 430):
            if refined_chunks[-1]["token_count"] < 160 or c["token_count"] < 160:
                prev = refined_chunks[-1]
                prev_body = prev["text"].split(" — ", 1)[-1]
                curr_body = c["text"].split(" — ", 1)[-1]
                merged_body = prev_body + " " + curr_body
                new_text = build_chunk_text(prev["doc"], prev["section"], prev["title"], merged_body)
                prev["text"] = new_text
                prev["token_count"] = count_tokens(new_text)
                continue
        refined_chunks.append(c)

    return refined_chunks


def create_chunks_from_extracted_dir(
    extracted_dir: str | Path,
    target_nrs: Optional[List[str]] = None,
) -> List[Chunk]:
    """Lê os arquivos JSON extraídos e produz a lista indexada de objetos Chunk."""
    extracted_dir = Path(extracted_dir)
    json_files = sorted(extracted_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"Nenhum arquivo JSON extraído encontrado em {extracted_dir}")

    all_chunks: List[Chunk] = []
    global_id = 1

    for jf in json_files:
        doc_id = jf.stem.lower()
        if target_nrs:
            normalized_targets = [t.lower().replace("_", "-") for t in target_nrs]
            if doc_id not in normalized_targets:
                continue

        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("items", [])
        if not items:
            continue

        raw_doc_chunks = chunk_items(items)
        for rdc in raw_doc_chunks:
            chunk_obj = Chunk(
                id=global_id,
                doc=rdc["doc"],
                section=rdc["section"],
                title=rdc["title"],
                page=rdc["page"],
                text=rdc["text"],
                token_count=rdc["token_count"],
            )
            all_chunks.append(chunk_obj)
            global_id += 1

    logger.info("Total de chunks gerados: %d", len(all_chunks))
    if all_chunks:
        toks = [c.token_count for c in all_chunks]
        logger.info(
            "Distribuição de tokens: mín=%d, máx=%d, média=%.1f, mediana=%.1f",
            min(toks), max(toks), statistics.mean(toks), statistics.median(toks)
        )

    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera chunks de ~200-400 tokens a partir dos dados extraídos de NRs.")
    parser.add_argument("--extracted-dir", type=str, default="corpus/data/extracted", help="Diretório com JSONs de extração.")
    parser.add_argument("--output-file", type=str, default="corpus/data/chunks.json", help="Arquivo JSON de saída com os chunks.")
    parser.add_argument("--nrs", type=str, default=None, help="Lista de NRs separadas por vírgula (ex.: nr-10,nr-06,nr-35).")
    args = parser.parse_args()

    target_nrs = [nr.strip() for nr in args.nrs.split(",")] if args.nrs else None
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = create_chunks_from_extracted_dir(args.extracted_dir, target_nrs=target_nrs)
    chunks_dict_list = [asdict(c) for c in chunks]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks_dict_list, f, ensure_ascii=False, indent=2)

    logger.info("Salvo arquivo de chunks: %s (%d chunks)", out_path, len(chunks))


if __name__ == "__main__":
    main()
