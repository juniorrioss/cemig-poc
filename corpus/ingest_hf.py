"""
ingest_hf.py — Ingestão de Normas Regulamentadoras e Manuais Comentados a partir de Parquet limpo.

Lê o dataset HuggingFace / Parquet contendo as 36 NRs em formato Markdown limpo e os
manuais comentados oficiais do MTE. Realiza o parsing da numeração hierárquica e seções,
gerando objetos ExtractedDocument e ExtractedItem compatíveis com corpus/chunk.py.

Trata normas regulamentadoras vinculantes (ex.: doc='nr-10') e manuais comentados
interpretativos (ex.: doc='nr-10-manual') como documentos distintos para rastreabilidade
e citação normativa precisa.
"""

import argparse
import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from corpus.extract import ExtractedDocument, ExtractedItem


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PARQUET_PATH = Path("/home/rios/projetos/cemig-mobile-llm/firstmate/data/nrs_hf/nrs.parquet")
DEFAULT_OUTPUT_DIR = Path("corpus/data/extracted_hf")

# Lista das 5 NRs prioritárias do v1
CORE_5_NRS = ["nr-06", "nr-10", "nr-12", "nr-18", "nr-35"]


def clean_markdown_line(line: str) -> str:
    """Higieniza linhas de Markdown, removendo quebras forçadas e normalizando espaços."""
    # Normaliza espaçamento em pronomes enclíticos (ex: 'aplicando -se' -> 'aplicando-se')
    line = re.sub(r'(\w+)\s+-\s*(se|nos|los|las|me|te)\b', r'\1-\2', line, flags=re.IGNORECASE)
    # Remove múltiplos espaços consecutivos
    line = re.sub(r'[ \t]+', ' ', line)
    return line.strip()


def find_norma_body_start(lines: List[str], nr_num: int) -> int:
    """Localiza a linha em que o corpo real da norma começa (ex.: 10.1 ou 10.1.1), pulando sumário."""
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        # Remove eventuais prefixos de portarias
        line = re.sub(r'^\((?:Redação|Texto|Alterad|Revogad|Inserid)[^)]+\)\s*', '', line)
        if re.match(rf'^{nr_num}\.1(?:\.1)?\b', line):
            # Assegura que não seja uma linha de sumário comprimido contendo múltiplas seções
            if not re.search(rf'\b{nr_num}\.2\b', line):
                return idx
    return 0


def parse_norma_content(
    norma_str: str,
    numero: int,
    titulo: str,
    conteudo: str,
) -> ExtractedDocument:
    """Converte o conteúdo Markdown limpo de uma NR em um ExtractedDocument estruturado."""
    doc_id = f"nr-{numero:02d}"
    lines = conteudo.splitlines()

    # Identificação do título oficial a partir do Markdown ou dos metadados da tabela
    first_line = lines[0] if lines else ""
    m_title = re.search(r'#\s*(NR-\d+\s*[-—]\s*(.*))', first_line)
    if m_title:
        doc_title = m_title.group(1).strip()
    else:
        doc_title = f"NR-{numero:02d} - {titulo.strip()}"

    start_idx = find_norma_body_start(lines, numero)
    body_lines = lines[start_idx:]

    current_major = f"{numero}.0"
    current_major_title = doc_title
    current_sub = current_major
    current_sub_title = current_major_title
    in_annex = False

    items: List[ExtractedItem] = []
    curr_sec_code = current_major
    curr_parent = current_major
    curr_title = current_major_title
    curr_text_lines: List[str] = []

    # Regex para itens numéricos hierárquicos: '10.2.8.2', '10.2', '21.1.', etc.
    item_pattern = re.compile(rf'^({numero}\.\d+(?:\.\d+)*)\.?\s*[-–.]?\s*(.*)$')
    annex_header_pattern = re.compile(r'^(?:#+\s*)?(ANEXO\s+[IVX\d]+[A-Z]?|GLOSSÁRIO)(?:\s*[-–:]?\s*(.*))?$', re.IGNORECASE)
    annex_item_pattern = re.compile(r'^([A-Z]\.\d+|\d+\.\d+|\d+\.)\s*[-–]?\s*(.*)$')
    annex_table_pattern = re.compile(r'^\|\s*(NR[- ]\d+)\s*\|', re.IGNORECASE)
    letter_section_pattern = re.compile(r'^([A-Z])\s*[-–]\s*(.*)$')

    def commit_current():
        nonlocal curr_text_lines, curr_sec_code, curr_parent, curr_title
        if not curr_text_lines:
            return
        combined_text = " ".join(curr_text_lines).strip()
        if combined_text:
            items.append(
                ExtractedItem(
                    doc=doc_id,
                    doc_title=doc_title,
                    section=curr_sec_code,
                    parent_section=curr_parent,
                    title=curr_title,
                    page=1,
                    text=combined_text,
                )
            )
        curr_text_lines = []

    for raw_line in body_lines:
        line = clean_markdown_line(raw_line)
        if not line or line == "---":
            continue
        if line.startswith("*Fonte:") or line.startswith("## Norma") or line.startswith("# NR-"):
            continue
        if "Este texto não substitui o publicado no DOU" in line:
            continue
        if re.match(r'^(?:Publicação|Alterações/Atualizações)\s+D\.O\.U\.', line, re.IGNORECASE):
            continue

        # Remove anotações de portarias no início de linhas normativas
        m_port = re.match(r'^\((?:Redação|Texto|Alterad|Revogad|Inserid)[^)]+\)\s*(.*)$', line, re.IGNORECASE)
        if m_port:
            rest_after = m_port.group(1).strip()
            if not rest_after:
                continue
            line = rest_after

        m_ann = annex_header_pattern.match(line)
        m_it = item_pattern.match(line)

        # Entrada em bloco de Anexo ou Glossário
        if m_ann:
            commit_current()
            in_annex = True
            ann_name = m_ann.group(1).title()
            ann_suffix = m_ann.group(2) or ""
            current_major = ann_name
            current_major_title = f"{ann_name} {ann_suffix}".strip()
            current_sub = current_major
            current_sub_title = current_major_title

            curr_sec_code = current_major
            curr_parent = current_major
            curr_title = current_major_title
            curr_text_lines = [line]
            continue

        if in_annex:
            m_tbl = annex_table_pattern.match(line)
            m_letter = letter_section_pattern.match(line)
            m_an_it = annex_item_pattern.match(line)

            if m_tbl:
                commit_current()
                nr_sub = m_tbl.group(1).upper().replace(" ", "-")
                current_sub = f"{current_major}.{nr_sub}"
                current_sub_title = f"Infrações {nr_sub}"
                curr_sec_code = current_sub
                curr_parent = current_major
                curr_title = current_sub_title
                curr_text_lines = [line]
                continue

            if m_letter and len(line) < 85:
                commit_current()
                current_sub = f"{current_major}.{m_letter.group(1)}"
                current_sub_title = m_letter.group(2).strip()
                curr_sec_code = current_sub
                curr_parent = current_major
                curr_title = current_sub_title
                curr_text_lines = [line]
            elif m_an_it:
                commit_current()
                code_suffix = m_an_it.group(1).rstrip(".")
                curr_sec_code = f"{current_sub}.{code_suffix}"
                curr_parent = current_sub
                curr_title = current_sub_title
                curr_text_lines = [line]
            else:
                curr_text_lines.append(line)
                # Fragmenta tabelas ou listas extensas em blocos equilibrados de ~200 palavras
                if sum(len(x.split()) for x in curr_text_lines) >= 200:
                    commit_current()
            continue

        if m_it:
            num = m_it.group(1)
            rest = m_it.group(2).strip()
            dots = num.count(".")

            # Seção nível 1 (ex.: 10.2 MEDIDAS DE CONTROLE)
            if dots == 1:
                commit_current()
                current_major = num
                current_major_title = rest.strip("- ") if rest else f"Seção {num}"
                current_sub = current_major
                current_sub_title = current_major_title

                curr_sec_code = num
                curr_parent = f"{numero}.0"
                curr_title = current_major_title

                if rest and not rest.isupper() and len(rest.split()) > 6:
                    curr_text_lines = [f"{num} {rest}"]
                else:
                    curr_text_lines = [line]

            # Subseção com título próprio (ex.: 10.2.8 MEDIDAS DE PROTEÇÃO COLETIVA)
            elif dots == 2 and (rest.isupper() or (len(rest.split()) <= 6 and not rest.endswith("."))):
                commit_current()
                current_sub = num
                current_sub_title = rest.strip("- ") if rest else current_major_title
                curr_sec_code = num
                curr_parent = current_major
                curr_title = current_sub_title
                curr_text_lines = [line]

            # Item individual (ex.: 10.2.8.1 ou 10.2.1)
            else:
                commit_current()
                curr_sec_code = num
                curr_parent = current_sub if current_sub != current_major else current_major
                curr_title = current_sub_title if current_sub_title != current_major_title else current_major_title
                curr_text_lines = [f"{num} {rest}" if rest else num]
        else:
            curr_text_lines.append(line)
            # Evita blocos gigantes contínuos sem subtítulos (>220 palavras)
            if sum(len(x.split()) for x in curr_text_lines) >= 220:
                commit_current()

    commit_current()

    logger.info("Ingerida norma %s (%s): %d itens estruturados", doc_id, doc_title, len(items))
    return ExtractedDocument(
        doc=doc_id,
        doc_title=doc_title,
        total_pages=1,
        items=items,
    )


def parse_manual_content(
    norma_str: str,
    numero: int,
    titulo: str,
    manual: str,
    manual_filename: Optional[str] = None,
) -> Optional[ExtractedDocument]:
    """Converte o manual comentado oficial em um ExtractedDocument distinto (doc='nr-XX-manual')."""
    if not manual or not manual.strip():
        return None

    doc_id = f"nr-{numero:02d}-manual"
    lines = manual.splitlines()

    # Identificação do título do manual comentado
    first_line = lines[0] if lines else ""
    m_h = re.search(r'#\s*(.*)', first_line)
    if m_h:
        doc_title = m_h.group(1).strip()
    else:
        doc_title = f"NR-{numero:02d} Manual Comentado - {titulo.strip()}"

    # Limpeza de artefatos gráficos do InDesign, cabeçalhos repetitivos e pontilhados de sumário
    clean_lines: List[str] = []
    for raw_line in lines:
        line = clean_markdown_line(raw_line)
        if not line:
            continue
        # Artefato de exportação InDesign de diagramação (ex: '35978001_miolo ok.indd 25 15/03/11 11:58')
        if re.search(r'\bmiolo ok\.indd\b', line, re.IGNORECASE):
            continue
        # Cabeçalhos correntes repetitivos de páginas
        if re.match(r'^MANUAL DE AUXÍLIO NA INTERPRETAÇÃO', line, re.IGNORECASE):
            continue
        if re.match(r'^Secretaria de Inspeção do Trabalho - SIT', line, re.IGNORECASE):
            continue
        if re.match(r'^Departamento de Segurança e Saúde no Trabalho', line, re.IGNORECASE):
            continue
        # Linhas de índice/sumário pontilhadas (ex: '35.1 OBJETIVO ...................... 11')
        if re.search(r'\.{4,}\s*\d+$', line):
            continue
        clean_lines.append(line)

    current_major = f"{numero}.0-manual"
    current_major_title = doc_title
    current_sub = current_major
    current_sub_title = current_major_title

    items: List[ExtractedItem] = []
    curr_sec_code = current_major
    curr_parent = current_major
    curr_title = current_major_title
    curr_text_lines: List[str] = []

    # Padrões para captura de itens da norma comentados ou seções numeradas do manual
    num_item_pat = re.compile(rf'^({numero}\.\d+(?:\.\d+)*)\.?\s*[-–.]?\s*(.*)$')
    general_num_pat = re.compile(r'^(\d+(?:\.\d+)*)\.?\s*[-–.]?\s+(.*)$')
    annex_pat = re.compile(r'^(?:#+\s*)?(ANEXO\s+[IVX\d]+[A-Z]?|GLOSSÁRIO)(?:\s*[-–:]?\s*(.*))?$', re.IGNORECASE)

    def commit_current():
        nonlocal curr_text_lines, curr_sec_code, curr_parent, curr_title
        if not curr_text_lines:
            return
        combined_text = " ".join(curr_text_lines).strip()
        if combined_text:
            items.append(
                ExtractedItem(
                    doc=doc_id,
                    doc_title=doc_title,
                    section=curr_sec_code,
                    parent_section=curr_parent,
                    title=curr_title,
                    page=1,
                    text=combined_text,
                )
            )
        curr_text_lines = []

    for line in clean_lines:
        m_ann = annex_pat.match(line)
        m_num = num_item_pat.match(line)
        m_gen = general_num_pat.match(line)

        # Anexo ou Glossário do manual
        if m_ann:
            commit_current()
            ann_name = m_ann.group(1).title()
            ann_suf = m_ann.group(2) or ""
            current_major = ann_name
            current_major_title = f"{ann_name} {ann_suf}".strip()
            current_sub = current_major
            current_sub_title = current_major_title
            curr_sec_code = ann_name
            curr_parent = ann_name
            curr_title = current_major_title
            curr_text_lines = [line]
            continue

        # Item específico da NR comentado (ex: 10.2.8.2, 35.1.1)
        if m_num:
            commit_current()
            num_code = m_num.group(1)
            rest = m_num.group(2).strip()
            dots = num_code.count(".")

            if dots == 1:
                current_major = num_code
                current_major_title = rest.strip("- ") if rest else f"Seção {num_code}"
                current_sub = current_major
                current_sub_title = current_major_title
                curr_sec_code = num_code
                curr_parent = f"{numero}.0-manual"
                curr_title = current_major_title
            elif dots == 2 and (rest.isupper() or (len(rest.split()) <= 6 and not rest.endswith("."))):
                current_sub = num_code
                current_sub_title = rest.strip("- ") if rest else current_major_title
                curr_sec_code = num_code
                curr_parent = current_major
                curr_title = current_sub_title
            else:
                curr_sec_code = num_code
                curr_parent = current_sub if current_sub != current_major else current_major
                curr_title = current_sub_title if current_sub_title != current_major_title else current_major_title

            curr_text_lines = [line]
            continue

        # Seção numerada autônoma do manual (ex.: 3.1 Hierarquia das medidas de proteção)
        if m_gen and len(line) < 120 and (m_gen.group(2) and (m_gen.group(2)[0].isupper() or m_gen.group(2).startswith('“'))):
            commit_current()
            num_code = m_gen.group(1)
            rest = m_gen.group(2).strip()
            curr_sec_code = num_code
            curr_parent = current_major
            curr_title = rest if rest else f"Seção {num_code}"
            curr_text_lines = [line]
            continue

        # Cabeçalhos Markdown explícitos (# Seção)
        if line.startswith("#"):
            commit_current()
            h_text = re.sub(r'^#+\s*', '', line).strip()
            curr_sec_code = h_text[:40]
            curr_parent = f"{numero}.0-manual"
            curr_title = h_text
            curr_text_lines = [line]
            continue

        curr_text_lines.append(line)
        if sum(len(x.split()) for x in curr_text_lines) >= 220:
            commit_current()

    commit_current()

    logger.info("Ingerido manual %s (%s): %d itens estruturados", doc_id, doc_title, len(items))
    return ExtractedDocument(
        doc=doc_id,
        doc_title=doc_title,
        total_pages=1,
        items=items,
    )


def ingest_all_hf(
    parquet_path: str | Path = DEFAULT_PARQUET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    target_nrs: Optional[List[str]] = None,
    include_manuals: bool = True,
) -> Dict[str, ExtractedDocument]:
    """Lê todas as NRs do arquivo Parquet e salva os arquivos JSON estruturados."""
    if pq is None:
        raise ImportError("A biblioteca 'pyarrow' é necessária para ler o Parquet. Instale via pip install pyarrow.")

    parquet_path = Path(parquet_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Arquivo Parquet não encontrado em {parquet_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(parquet_path))
    rows = table.to_pylist()

    results: Dict[str, ExtractedDocument] = {}
    normalized_targets = [t.lower().replace("_", "-") for t in target_nrs] if target_nrs else None

    for row in rows:
        num = row["numero"]
        doc_id = f"nr-{num:02d}"

        # Verifica filtro de NRs se especificado
        if normalized_targets and doc_id not in normalized_targets:
            continue

        # Ingestão da norma vinculante
        doc_norma = parse_norma_content(
            norma_str=row["norma"],
            numero=num,
            titulo=row["titulo"],
            conteudo=row["conteudo"],
        )
        results[doc_norma.doc] = doc_norma

        norma_file = out_path / f"{doc_norma.doc}.json"
        with open(norma_file, "w", encoding="utf-8") as f:
            json.dump(doc_norma.to_dict(), f, ensure_ascii=False, indent=2)

        # Ingestão do manual comentado (se solicitado e presente)
        if include_manuals and row.get("manual"):
            doc_manual = parse_manual_content(
                norma_str=row["norma"],
                numero=num,
                titulo=row["titulo"],
                manual=row["manual"],
                manual_filename=row.get("manual_filename"),
            )
            if doc_manual and doc_manual.items:
                results[doc_manual.doc] = doc_manual
                manual_file = out_path / f"{doc_manual.doc}.json"
                with open(manual_file, "w", encoding="utf-8") as f:
                    json.dump(doc_manual.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info("Ingestão concluída! Total de documentos gerados: %d em %s", len(results), out_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestão de Normas Regulamentadoras e Manuais a partir de Parquet limpo.")
    parser.add_argument("--parquet-path", type=str, default=str(DEFAULT_PARQUET_PATH), help="Caminho do arquivo nrs.parquet.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Diretório para salvar os JSONs extraídos.")
    parser.add_argument("--nrs", type=str, default=None, help="Lista de NRs separadas por vírgula (ex.: nr-10,nr-06,nr-35). Padrão: todas.")
    parser.add_argument("--no-manuals", action="store_true", help="Desabilita a extração de manuais comentados.")
    args = parser.parse_args()

    target_nrs = [nr.strip() for nr in args.nrs.split(",")] if args.nrs else None
    ingest_all_hf(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        target_nrs=target_nrs,
        include_manuals=not args.no_manuals,
    )


if __name__ == "__main__":
    main()
