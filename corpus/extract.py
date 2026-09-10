"""
extract.py — Extração de texto estruturado de Normas Regulamentadoras (PDFs).

Preserva a numeração hierárquica oficial (ex.: '10.2.8.2'), títulos de seção,
páginas aproximadas e anexos/glossários, filtrando cabeçalhos de DOU e sumários.
"""

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pypdf
except ImportError:
    pypdf = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_PDF_DIR = Path("/home/rios/projetos/cemig-mobile-llm/firstmate/data/NRs")


@dataclass
class ExtractedItem:
    """Representa uma unidade estruturada de texto de uma Norma Regulamentadora."""
    doc: str               # Identificador canônico do documento, ex: 'nr-10'
    doc_title: str         # Título oficial, ex: 'NR 10 - SEGURANÇA EM INSTALAÇÕES E SERVIÇOS EM ELETRICIDADE'
    section: str           # Número do item ou código da seção, ex: '10.2.8.2' ou 'ANEXO I'
    parent_section: str    # Seção pai direta, ex: '10.2.8' ou '10.2'
    title: str             # Título contextual da seção, ex: 'Medidas de proteção coletiva'
    page: int              # Página aproximada do PDF de origem
    text: str              # Conteúdo textual integral do item


@dataclass
class ExtractedDocument:
    """Representa uma NR completa processada com suas seções estruturadas."""
    doc: str
    doc_title: str
    total_pages: int
    items: List[ExtractedItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc": self.doc,
            "doc_title": self.doc_title,
            "total_pages": self.total_pages,
            "total_items": len(self.items),
            "items": [asdict(item) for item in self.items],
        }


def clean_line_text(line: str) -> str:
    """Normaliza espaçamentos redundantes e quebras de hífen em texto extraído."""
    # Corrige espaços em pronomes enclíticos comuns em textos legais: 'observando -se' -> 'observando-se'
    line = re.sub(r'(\w+)\s+-\s*(se|nos|los|las|me|te)\b', r'\1-\2', line, flags=re.IGNORECASE)
    # Reduz espaços múltiplos contíguos
    line = re.sub(r'[ \t]+', ' ', line)
    return line.strip()


def extract_nr_metadata(first_pages_lines: List[str], filename: str) -> tuple[str, str, str]:
    """Identifica o número canônico da NR, o ID do documento e seu título formal."""
    m_file = re.search(r'nr[-_]?(\d+)', filename.lower())
    if m_file:
        nr_num = str(int(m_file.group(1)))
        doc_id = f"nr-{int(nr_num):02d}"
    else:
        nr_num = "0"
        doc_id = Path(filename).stem.lower()

    doc_title = ""
    for line in first_pages_lines[:25]:
        line_clean = clean_line_text(line)
        if re.match(r'^NR\s*\d+', line_clean, re.IGNORECASE):
            doc_title = line_clean
            break

    if not doc_title:
        doc_title = f"NR-{nr_num.zfill(2)}"

    return doc_id, nr_num, doc_title


def extract_document(pdf_path: str | Path) -> ExtractedDocument:
    """Extrai uma NR de um arquivo PDF, gerando itens estruturados por seção."""
    if pypdf is None:
        raise ImportError("A biblioteca 'pypdf' é necessária para extração de PDFs. Instale via pip install pypdf.")

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

    reader = pypdf.PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    # Coleta linhas com rastreamento de página original
    raw_lines: List[tuple[int, str]] = []
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        for line in page_text.splitlines():
            raw_lines.append((page_num, line.strip()))

    first_lines = [l for _, l in raw_lines[:40] if l]
    doc_id, nr_num, doc_title = extract_nr_metadata(first_lines, pdf_path.name)

    # Filtragem de sumário, números de página avulsos e carimbos de publicação do DOU
    cleaned_lines: List[tuple[int, str]] = []
    in_sumario = False
    sumario_item_count = 0

    for page_num, line in raw_lines:
        if not line:
            continue
        # Número de página isolado
        if re.match(r'^\d+$', line):
            continue
        # Carimbo repetitivo de rodapé/cabeçalho oficial
        if "Este texto não substitui o publicado no DOU" in line:
            continue

        # Identificação e salto da linha com o título da NR (já embutido nos headers dos chunks)
        if re.match(r'^NR\s*\d+', line, re.IGNORECASE):
            if not doc_title:
                doc_title = line
            continue

        # Filtragem de notas de redação de portaria entre parênteses
        if re.match(r'^\((?:Redação|Texto)\s+dad[ao]\s+pela\s+Portaria', line, re.IGNORECASE):
            continue

        # Filtragem do bloco de portarias e publicações do DOU antes do primeiro item real
        if re.match(r'^(?:Publicação|Alterações/Atualizações)\s+D\.O\.U\.', line, re.IGNORECASE):
            continue
        if re.match(r'^Portaria\s+[A-Z]+\s+n[ºo\.]', line, re.IGNORECASE):
            continue
        if re.match(r'^\d{2}/\d{2}/\d{2,4}$', line):
            continue

        # Detecção de início de sumário (índice preliminar)
        if line.upper() == "SUMÁRIO":
            in_sumario = True
            sumario_item_count = 0
            continue

        if in_sumario:
            sumario_item_count += 1
            # Fim do sumário ao encontrar cabeçalho de portaria
            if re.match(r'^\((?:Redação|Texto) dada', line, re.IGNORECASE):
                in_sumario = False
                continue
            # Fim do sumário quando a numeração reinicia no item 1 do corpo da norma
            if sumario_item_count > 3 and (re.match(rf'^{nr_num}\.1(?:\s+|\.|$)', line) or re.match(rf'^{nr_num}\.1\.1', line)):
                in_sumario = False
                # Não dá continue aqui: processa a linha atual como primeiro item do corpo
            else:
                continue

        cleaned_lines.append((page_num, clean_line_text(line)))

    # Padrões regex para hierarquia de itens da norma
    # Ex: '10.2.8.2', '10.2.8', '10.2', '6.5.1', '35.4.1.1'
    item_pattern = re.compile(rf'^({nr_num}\.\d+(?:\.\d+)*)\s*[-–.]?\s*(.*)$')
    # Anexos e Glossário
    annex_header_pattern = re.compile(r'^(ANEXO\s+[IVX\d]+|GLOSSÁRIO)(?:\s*[-–]\s*(.*))?$', re.IGNORECASE)
    annex_item_pattern = re.compile(r'^([A-Z]\.\d+|\d+\.\d+|\d+\.)\s*[-–]?\s*(.*)$')
    letter_section_pattern = re.compile(r'^([A-Z])\s*[-–]\s*(.*)$')

    current_major = f"{nr_num}.0"
    current_major_title = doc_title
    current_sub = current_major
    current_sub_title = current_major_title
    in_annex = False

    extracted_items: List[ExtractedItem] = []
    curr_sec_code = current_major
    curr_parent = current_major
    curr_title = current_major_title
    curr_page = 1
    curr_text_lines: List[str] = []

    def commit_current():
        nonlocal curr_text_lines, curr_sec_code, curr_parent, curr_title, curr_page
        if not curr_text_lines:
            return
        combined_text = " ".join(curr_text_lines).strip()
        if combined_text:
            extracted_items.append(
                ExtractedItem(
                    doc=doc_id,
                    doc_title=doc_title,
                    section=curr_sec_code,
                    parent_section=curr_parent,
                    title=curr_title,
                    page=curr_page,
                    text=combined_text,
                )
            )
        curr_text_lines = []

    for page_num, line in cleaned_lines:
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
            curr_page = page_num
            curr_text_lines = [line]
            continue

        if in_annex:
            m_letter = letter_section_pattern.match(line)
            m_an_it = annex_item_pattern.match(line)

            if m_letter and len(line) < 85:
                commit_current()
                current_sub = f"{current_major}.{m_letter.group(1)}"
                current_sub_title = m_letter.group(2).strip()
                curr_sec_code = current_sub
                curr_parent = current_major
                curr_title = current_sub_title
                curr_page = page_num
                curr_text_lines = [line]
            elif m_an_it:
                commit_current()
                code_suffix = m_an_it.group(1).rstrip(".")
                curr_sec_code = f"{current_sub}.{code_suffix}"
                curr_parent = current_sub
                curr_title = current_sub_title
                curr_page = page_num
                curr_text_lines = [line]
            else:
                curr_text_lines.append(line)
            continue

        if m_it:
            num = m_it.group(1)
            rest = m_it.group(2).strip()
            dots = num.count(".")

            # Seção nível 1 (ex.: 10.2 MEDIDAS DE CONTROLE)
            if dots == 1:
                commit_current()
                current_major = num
                # Título limpo
                current_major_title = rest.strip("- ") if rest else f"Seção {num}"
                current_sub = current_major
                current_sub_title = current_major_title

                curr_sec_code = num
                curr_parent = f"{nr_num}.0"
                curr_title = current_major_title
                curr_page = page_num

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
                curr_page = page_num
                curr_text_lines = [line]

            # Item individual (ex.: 10.2.8.1 ou 10.2.1)
            else:
                commit_current()
                curr_sec_code = num
                curr_parent = current_sub if current_sub != current_major else current_major
                curr_title = current_sub_title if current_sub_title != current_major_title else current_major_title
                curr_page = page_num
                curr_text_lines = [f"{num} {rest}" if rest else num]
        else:
            curr_text_lines.append(line)

    commit_current()

    logger.info("Extraído %s (%s): %d páginas, %d itens estruturados", doc_id, doc_title, total_pages, len(extracted_items))
    return ExtractedDocument(
        doc=doc_id,
        doc_title=doc_title,
        total_pages=total_pages,
        items=extracted_items,
    )


def extract_all_documents(
    pdf_dir: str | Path = DEFAULT_PDF_DIR,
    target_nrs: Optional[List[str]] = None,
) -> Dict[str, ExtractedDocument]:
    """Extrai todas as NRs selecionadas ou todas as encontradas no diretório."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"Diretório de PDFs não encontrado: {pdf_dir}")

    all_pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not all_pdf_files:
        raise FileNotFoundError(f"Nenhum arquivo PDF encontrado em {pdf_dir}")

    results: Dict[str, ExtractedDocument] = {}
    for pdf_file in all_pdf_files:
        m = re.search(r'nr[-_]?(\d+)', pdf_file.name.lower())
        if not m:
            continue
        doc_id = f"nr-{int(m.group(1)):02d}"

        if target_nrs:
            normalized_targets = [t.lower().replace("_", "-") for t in target_nrs]
            if doc_id not in normalized_targets:
                continue

        try:
            doc = extract_document(pdf_file)
            results[doc.doc] = doc
        except Exception as e:
            logger.error("Erro ao extrair %s: %s", pdf_file.name, e)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai texto estruturado de Normas Regulamentadoras (PDFs).")
    parser.add_argument("--pdf-dir", type=str, default=str(DEFAULT_PDF_DIR), help="Diretório contendo os PDFs das NRs.")
    parser.add_argument("--output-dir", type=str, default="corpus/data/extracted", help="Diretório para salvar os JSONs extraídos.")
    parser.add_argument("--nrs", type=str, default=None, help="Lista de NRs separadas por vírgula (ex.: nr-10,nr-06,nr-35). Padrão: todas.")
    args = parser.parse_args()

    target_nrs = [nr.strip() for nr in args.nrs.split(",")] if args.nrs else None
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    extracted_docs = extract_all_documents(args.pdf_dir, target_nrs=target_nrs)
    for doc_id, doc in extracted_docs.items():
        doc_file = out_path / f"{doc_id}.json"
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Salvo arquivo estruturado: %s (%d itens)", doc_file, len(doc.items))

    logger.info("Processamento concluído. Total de documentos extraídos: %d", len(extracted_docs))


if __name__ == "__main__":
    main()
