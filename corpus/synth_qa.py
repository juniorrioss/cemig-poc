"""
synth_qa.py — Geração sintética de pares Pergunta / Resposta-Ouro / Chunk-Fonte.

Gera entre 60 e 100 pares sintéticos a partir das 5 NRs críticas para o setor elétrico:
- NR-10 (Segurança em Eletricidade - Mandatória)
- NR-06 (Equipamentos de Proteção Individual - EPI)
- NR-35 (Trabalho em Altura)
- NR-12 (Máquinas e Equipamentos / Bloqueio LOTO)
- NR-18 (Construção Civil / Instalações Provisórias e Distâncias de Redes)

Utiliza o LLM CLI Anthropic Claude Code ('claude -p') documentado no ambiente,
simulando a linguagem natural e coloquial de eletricistas de campo (CEMIG).
Salva em formato JSONL com chunk_id correspondente no index.db.
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DB_PATH = Path("corpus/index.db")
DEFAULT_OUTPUT_JSONL = Path("corpus/qa_pairs.jsonl")

# Justificativa técnica formal para as 5 NRs selecionadas
NR_JUSTIFICATIONS = {
    "nr-10": "Mandatória para eletricistas: desenergização em 6 etapas, alta tensão, SEP, zonas controlada e de risco, prontuário (PIE), direito de recusa e qualificação.",
    "nr-06": "Essencial para segurança individual do eletricista: luvas de borracha isolante, vestimentas com classificação ATPV contra arco elétrico, capacetes classe B e calçados dielétricos.",
    "nr-35": "Rotina operacional em postes, linhas de transmissão e subestações (>2m): sistemas de proteção contra quedas (SPQ), cinto tipo paraquedista com talabarte duplo e pontos de ancoragem.",
    "nr-12": "Manutenção em painéis e motores: bloqueio e consignação de energias perigosas (LOTO), aterramento de carcaças, circuitos de comando elétrico e botoeiras de emergência.",
    "nr-18": "Trabalhos em canteiros de obra e proximidade com redes elétricas aéreas: quadros provisórios com proteção DR, cabos protegidos e afastamentos mínimos de linhas energizadas.",
}


# Tópicos-chave com alvos normativos para guiar a síntese de perguntas de operários
TARGET_TOPICS = [
    # --- NR-10 (Segurança em Eletricidade) ---
    ("nr-10", "10.2.8", "Medidas de proteção coletiva prioritárias e emprego de tensão de segurança"),
    ("nr-10", "10.2.9", "EPIs adequados, inflamabilidade de roupas e proibição absoluta de adornos pessoais"),
    ("nr-10", "10.4.5", "Iluminação adequada e ergonomia para trabalho elétrico com membros superiores livres"),
    ("nr-10", "10.5.1", "Sequência obrigatória das 6 etapas para desenergização de circuito elétrico"),
    ("nr-10", "10.5.2", "Procedimentos e etapas ordenadas para reenergização da instalação"),
    ("nr-10", "10.6.1", "Trabalho em instalação energizada ou ingresso na zona controlada"),
    ("nr-10", "10.7.1", "Obrigatoriedade de trabalho em dupla para alta tensão e SEP"),
    ("nr-10", "10.7.4", "Ordem de serviço específica para alta tensão e autorização prévia"),
    ("nr-10", "10.7.5", "Inspeção e teste de ferramentas isolantes antes de mexer em circuito energizado"),
    ("nr-10", "10.8.1", "Diferença entre trabalhador qualificado, legalmente habilitado, capacitado e autorizado"),
    ("nr-10", "10.11.2", "Análise de Risco (APR) e procedimentos de trabalho precedentes ao serviço"),
    ("nr-10", "10.12.2", "Treinamento em primeiros socorros, massagem cardíaca e resgate de acidentados elétricos"),
    ("nr-10", "10.14.1", "Direito de recusa do trabalhador diante de risco grave e iminente"),
    ("nr-10", "Glossário", "Definição de alta tensão (acima de 1000V CA ou 1500V CC) e baixa tensão"),
    ("nr-10", "Glossário", "Definição de zona de risco, zona controlada e zona livre"),
    ("nr-10", "10.2.4", "Prontuário de Instalações Elétricas (PIE) para estabelecimentos com carga > 75 kW"),
    ("nr-10", "10.8.8", "Reciclagem bienal do treinamento de segurança em instalações elétricas"),
    ("nr-10", "10.9.1", "Instalações elétricas em áreas classificadas e atmosferas explosivas"),
    ("nr-10", "10.10.1", "Sinalização de advertência e travamento contra acionamento indevido"),
    ("nr-10", "10.3.1", "Dispositivos de bloqueio contra reenergização acidental em projetos elétricos"),
    ("nr-10", "10.2.3", "Obrigatoriedade de manter esquemas unifilares atualizados nas instalações"),
    ("nr-10", "10.4.4", "Manutenção preventiva em instalações elétricas e inspeção periódica"),
    ("nr-10", "10.5.4", "Serviços em instalações elétricas desligadas mas com possibilidade de indução"),
    ("nr-10", "10.7.8", "Ensaios elétricos periódicos em ferramentas e luvas de borracha isolantes"),
    ("nr-10", "10.11.6", "Comunicação e sinalização de equipe de trabalho com líder designado"),

    # --- NR-06 (Equipamentos de Proteção Individual) ---
    ("nr-06", "6.5.1", "Obrigação da empresa de fornecer EPI gratuito com CA válido e treinar sobre o uso"),
    ("nr-06", "6.6.1", "Responsabilidade do operário de usar, guardar, zelar e avisar se o EPI danificar"),
    ("nr-06", "6.4.1", "Proibição de uso de EPI sem Certificado de Aprovação (CA) ou com CA vencido na compra"),
    ("nr-06", "6.7.2", "Orientações e treinamentos obrigatórios na entrega de novo equipamento de proteção"),
    ("nr-06", "Anexo I", "Luvas de borracha isolante e sobreluvas de couro para proteção contra choque"),
    ("nr-06", "Anexo I", "Capacete de proteção classe B para eletricistas contra impacto e choque elétrico"),
    ("nr-06", "Anexo I", "Vestimenta de segurança anti-arco elétrico com proteção térmica ATPV"),
    ("nr-06", "Anexo I", "Calçado com solado isolante dielétrico sem partes metálicas aparentes"),
    ("nr-06", "6.9.1", "Validade do CA e condições de higienização do EPI pelo trabalhador ou empresa"),
    ("nr-06", "6.5.2", "Critérios para seleção de EPI considerando riscos ergonômicos e operacionais"),
    ("nr-06", "6.3.2", "Conceito de EPI conjugado para múltiplos riscos no mesmo posto de trabalho"),
    ("nr-06", "Anexo I", "Óculos de proteção contra radiação ultravioleta e arco elétrico"),
    ("nr-06", "6.8.1", "Responsabilidade de fabricantes de fornecer manual de instruções do EPI em português"),

    # --- NR-35 (Trabalho em Altura) ---
    ("nr-35", "35.2.1", "Altura mínima considerada trabalho em altura pela norma (a partir de 2,00 metros)"),
    ("nr-35", "35.4.1", "Exigência de capacitação formal e exame médico de aptidão para altura (ASO específico)"),
    ("nr-35", "35.4.2", "Carga horária mínima do treinamento de trabalho em altura (8 horas com teoria e prática)"),
    ("nr-35", "35.5.1", "Obrigatoriedade de Análise de Risco (AR) e Permissão de Trabalho (PT)"),
    ("nr-35", "35.6.1", "Composição do Sistema de Proteção Contra Quedas (ancoragem, elemento de ligação e cinto)"),
    ("nr-35", "35.6.4", "Cinturão de segurança tipo paraquedista e uso de talabarte duplo"),
    ("nr-35", "35.7.1", "Procedimentos de emergência e plano de resgate para trabalhador suspenso em altura"),
    ("nr-35", "35.5.4", "Condições climáticas impeditivas (vento forte, chuva, descargas atmosféricas) para trabalho em altura"),
    ("nr-35", "Anexo II", "Requisitos de resistência e inspeção de pontos de ancoragem para linhas de vida"),
    ("nr-35", "Anexo III", "Uso seguro de escadas portáteis e manutenção de 3 pontos de contato"),
    ("nr-35", "35.4.3", "Periodicidade bienal do treinamento periódico de trabalho em altura"),
    ("nr-35", "Anexo I", "Técnicas de progressão por corda e uso de dois sistemas independentes"),
    ("nr-35", "35.6.6", "Uso obrigatório de absorvedor de energia acoplado ao talabarte"),

    # --- NR-12 (Segurança em Máquinas e Equipamentos / LOTO) ---
    ("nr-12", "12.3", "Prevenção de choque elétrico em quadros e fiações de máquinas e blindagem de cabos"),
    ("nr-12", "12.3.4", "Aterramento obrigatório das carcaças metálicas e partes condutivas de máquinas"),
    ("nr-12", "12.6", "Dispositivos de parada de emergência e proibição de usá-los como liga/desliga comum"),
    ("nr-12", "12.11", "Procedimento de manutenção com bloqueio de energia (LOTO / Lockout Tagout)"),
    ("nr-12", "12.11.3", "Travamento com cadeado físico de segurança e etiqueta de advertência de manutenção"),
    ("nr-12", "12.4.1", "Dispositivos de partida e comando fora da zona de perigo das máquinas"),
    ("nr-12", "12.5.1", "Proteções fixas e móveis intertravadas impedindo acesso a peças em movimento"),
    ("nr-12", "12.16.1", "Capacitação para operação e manutenção de máquinas perigosas"),
    ("nr-12", "12.12.1", "Sinalização de segurança visível em áreas com risco de aprisionamento ou choque elétrico"),
    ("nr-12", "12.7", "Válvulas de alívio e segurança em componentes pneumáticos e hidráulicos de máquinas"),
    ("nr-12", "12.8", "Proteção de transportadores contínuos de materiais e esteiras"),

    # --- NR-18 (Construção Civil / Instalações Elétricas Temporárias) ---
    ("nr-18", "18.6.1", "Instalações elétricas provisórias em obras devem atender integralmente à NR-10"),
    ("nr-18", "18.6.4", "Passagem e fixação de cabos elétricos sem cruzar passagens de pessoas e materiais no chão"),
    ("nr-18", "18.6.10", "Aterramento obrigatório de estruturas metálicas e quadros de distribuição do canteiro"),
    ("nr-18", "18.6.11", "Uso obrigatório de dispositivo diferencial residual (DR) nos circuitos do canteiro"),
    ("nr-18", "18.6.7", "Quadros elétricos trancados com aviso de advertência e acesso apenas para autorizados"),
    ("nr-18", "18.9.1", "Proteção coletiva contra queda em bordas e vãos de obras próximas a redes elétricas"),
    ("nr-18", "18.10.2", "Inspeção diária de ferramentas elétricas portáteis e proibição de cabos emendados com fita comum"),
    ("nr-18", "18.6.2", "Distâncias mínimas seguras de guindastes e andaimes em relação a redes elétricas aéreas"),
    ("nr-18", "18.5.1", "Condições de segurança em áreas de vivência e instalações sanitárias no canteiro"),
    ("nr-18", "18.14.1", "Treinamento admissional com carga horária de 4 horas antes de iniciar na obra"),
]


def find_matching_chunk(cur: sqlite3.Cursor, doc: str, section: str, keywords: str) -> Optional[int]:
    """Encontra o chunk mais relevante no SQLite index.db para um par Q&A."""
    # Busca por doc e seção exata ou prefixo
    cur.execute("""
        SELECT id, section, title, text
        FROM chunks
        WHERE doc = ? AND (section LIKE ? OR text LIKE ?)
        LIMIT 5;
    """, (doc, f"%{section}%", f"%{section}%"))
    candidates = cur.fetchall()

    if candidates:
        return candidates[0][0]

    # Busca alternativa com FTS5 combinando doc e palavras-chave
    words = [w for w in re.findall(r'\w+', keywords) if len(w) > 3][:4]
    if words:
        fts_q = f'"{doc}" AND (' + " OR ".join(f'"{w}"' for w in words) + ")"
        try:
            cur.execute("""
                SELECT c.id
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts) ASC
                LIMIT 1;
            """, (fts_q,))
            row = cur.fetchone()
            if row:
                return row[0]
        except sqlite3.OperationalError:
            pass

    # Fallback: primeiro chunk do documento
    cur.execute("SELECT id FROM chunks WHERE doc = ? LIMIT 1;", (doc,))
    row = cur.fetchone()
    return row[0] if row else None


def call_llm_synth(prompt: str) -> str:
    """Invoca o LLM CLI ('claude -p') documentado no ambiente."""
    cmd = ["claude", "-p", prompt]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def generate_batch_qa(topic_slice: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    """Gera um lote de pares Q&A a partir de uma fatia de tópicos normativos."""
    topics_desc = "\n".join(
        f"- Documento: {doc.upper()} | Seção/Item: {sec} | Tema: {desc}"
        for doc, sec, desc in topic_slice
    )

    prompt = f"""Você é um especialista em segurança do trabalho elétrico e engenharia de campo da CEMIG.
Gere pares de Perguntas e Respostas-Ouro sintéticas para avaliação de um sistema RAG offline para eletricistas.

TÓPICOS ALVO:
{topics_desc}

DIRETRIZES FUNDAMENTAIS:
1. 'question': Deve soar exatamente como um eletricista de campo ou operário brasileiro falaria no dia a dia pelo rádio ou gravando áudio no aplicativo ("Tô no poste...", "O encarregado pediu pra...", "Minha botina furou, posso...", "Como é a ordem pra desligar a chave faca?", "Posso mexer sozinho no barramento?"). Use gírias e expressões comuns de campo, nunca fale como advogado ou com a linguagem formal da norma.
2. 'golden_answer': Deve ser técnica, precisa, direta e OBRIGATORIAMENTE citar a norma e o item específico (ex: "Conforme o item 10.5.1 da NR-10...", "De acordo com o item 35.2.1 da NR-35...").
3. 'query_terms': 3 a 6 palavras-chave essenciais (termos soltos) que um modelo SLM geraria em tool-calling para buscar no banco SQLite FTS5.
4. 'doc': o ID em minúsculas (ex: 'nr-10', 'nr-06', 'nr-35', 'nr-12', 'nr-18').
5. 'section': o código do item/seção correspondente.

Retorne APENAS um array JSON puro (sem markdown, sem ```json, sem texto antes ou depois) contendo uma lista de objetos com as chaves:
["doc", "section", "question", "golden_answer", "query_terms"]"""

    raw_output = call_llm_synth(prompt)

    # Extrai o bloco JSON caso haja resíduo de formatação
    m_json = re.search(r'\[.*\]', raw_output, re.DOTALL)
    if m_json:
        raw_output = m_json.group(0)

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError as e:
        logger.error("Erro ao decodificar JSON gerado pelo LLM: %s\nConteúdo: %s", e, raw_output[:300])

    return []


def generate_all_qa_pairs(
    db_path: Path = DEFAULT_DB_PATH,
    output_path: Path = DEFAULT_OUTPUT_JSONL,
    batch_size: int = 5,
) -> List[Dict[str, Any]]:
    """Gera o dataset completo de 60-100 pares Q&A cobrindo as 5 NRs."""
    if not db_path.exists():
        raise FileNotFoundError(f"Banco {db_path} não encontrado. Execute corpus/build_index.py primeiro.")

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    all_pairs: List[Dict[str, Any]] = []
    pair_counter = 1

    logger.info("Iniciando geração sintética com LLM CLI ('claude -p') para %d tópicos...", len(TARGET_TOPICS))

    for i in range(0, len(TARGET_TOPICS), batch_size):
        slice_topics = TARGET_TOPICS[i : i + batch_size]
        logger.info("Gerando lote %d/%d (tópicos %d a %d)...", (i // batch_size) + 1, (len(TARGET_TOPICS) + batch_size - 1) // batch_size, i + 1, min(i + batch_size, len(TARGET_TOPICS)))

        try:
            batch_result = generate_batch_qa(slice_topics)
            for item in batch_result:
                doc = item.get("doc", "").lower()
                sec = item.get("section", "")
                q = item.get("question", "")
                ans = item.get("golden_answer", "")
                q_terms = item.get("query_terms", "")
                if isinstance(q_terms, list):
                    q_terms = " ".join(str(x) for x in q_terms)

                if not q or not ans:
                    continue

                chunk_id = find_matching_chunk(cur, doc, sec, q_terms)

                qa_record = {
                    "id": f"qa-{pair_counter:03d}",
                    "doc": doc,
                    "section": sec,
                    "chunk_id": chunk_id,
                    "relevant_chunk_ids": [chunk_id] if chunk_id else [],
                    "question": q,
                    "golden_answer": ans,
                    "query_terms": q_terms,
                }
                all_pairs.append(qa_record)
                pair_counter += 1

        except Exception as e:
            logger.error("Falha ao gerar lote: %s", e)

    con.close()

    # Salva no arquivo JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("Geração concluída com sucesso! Total de pares gerados: %d em %s", len(all_pairs), output_path)
    return all_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera pares P&R sintéticos de campo com LLM CLI ('claude -p').")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="Caminho do banco SQLite FTS5 index.db.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_JSONL), help="Arquivo JSONL de saída.")
    parser.add_argument("--force", action="store_true", help="Sobrescreve o arquivo JSONL existente.")
    args = parser.parse_args()

    out_file = Path(args.output)
    if out_file.exists() and not args.force:
        logger.info("Arquivo %s já existe (%d linhas). Use --force para regenerar.", out_file, sum(1 for _ in open(out_file, encoding='utf-8')))
        return

    generate_all_qa_pairs(db_path=Path(args.db), output_path=out_file)


if __name__ == "__main__":
    main()
