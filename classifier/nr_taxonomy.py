#!/usr/bin/env python3
"""
nr_taxonomy.py — Taxonomia canônica das 36 Normas Regulamentadoras (NRs).

Fonte dos títulos oficiais: dataset limpo `nrs.parquet` (coluna `titulo`). Fornece:
  - NR_TITLES: título oficial por código (nr-XX).
  - NR_ONELINE: descrição de 1 linha (para prompt de classificação no LFM2.5).
  - ELECTRICIAN_NRS: as 9 NRs mais relevantes para o eletricista/operário CEMIG.
  - CLASSES: lista ordenada canônica das 36 classes + rótulo especial 'nenhuma'.
  - normalize_nr(): normaliza qualquer variação textual (NR 10, nr10, 10) -> 'nr-10'.

Este módulo é a ÚNICA fonte de verdade de classes; gen_labels.py, o shootout de
classificadores e o prompt-classify no LFM2.5 importam daqui para evitar divergência.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Títulos oficiais (nrs.parquet). NR-02 e NR-27 foram revogadas -> ausentes do corpus.
NR_TITLES = {
    "nr-01": "DISPOSIÇÕES GERAIS E GERENCIAMENTO DE RISCOS OCUPACIONAIS",
    "nr-03": "EMBARGO E INTERDIÇÃO",
    "nr-04": "SERVIÇOS ESPECIALIZADOS EM SEGURANÇA E EM MEDICINA DO TRABALHO",
    "nr-05": "COMISSÃO INTERNA DE PREVENÇÃO DE ACIDENTES",
    "nr-06": "EQUIPAMENTO DE PROTEÇÃO INDIVIDUAL - EPI",
    "nr-07": "PROGRAMA DE CONTROLE MÉDICO DE SAÚDE OCUPACIONAL",
    "nr-08": "EDIFICAÇÕES",
    "nr-09": "AVALIAÇÃO E CONTROLE DAS EXPOSIÇÕES OCUPACIONAIS A AGENTES FÍSICOS, QUÍMICOS E BIOLÓGICOS",
    "nr-10": "SEGURANÇA EM INSTALAÇÕES E SERVIÇOS EM ELETRICIDADE",
    "nr-11": "TRANSPORTE, MOVIMENTAÇÃO, ARMAZENAGEM E MANUSEIO DE MATERIAIS",
    "nr-12": "SEGURANÇA NO TRABALHO EM MÁQUINAS E EQUIPAMENTOS",
    "nr-13": "CALDEIRAS, VASOS DE PRESSÃO E TUBULAÇÕES E TANQUES METÁLICOS DE ARMAZENAMENTO",
    "nr-14": "FORNOS",
    "nr-15": "ATIVIDADES E OPERAÇÕES INSALUBRES",
    "nr-16": "ATIVIDADES E OPERAÇÕES PERIGOSAS",
    "nr-17": "ERGONOMIA",
    "nr-18": "SEGURANÇA E SAÚDE NO TRABALHO NA INDÚSTRIA DA CONSTRUÇÃO",
    "nr-19": "EXPLOSIVOS",
    "nr-20": "SEGURANÇA E SAÚDE NO TRABALHO COM INFLAMÁVEIS E COMBUSTÍVEIS",
    "nr-21": "TRABALHOS A CÉU ABERTO",
    "nr-22": "SEGURANÇA E SAÚDE OCUPACIONAL NA MINERAÇÃO",
    "nr-23": "PROTEÇÃO CONTRA INCÊNDIOS",
    "nr-24": "CONDIÇÕES SANITÁRIAS E DE CONFORTO NOS LOCAIS DE TRABALHO",
    "nr-25": "RESÍDUOS INDUSTRIAIS",
    "nr-26": "SINALIZAÇÃO DE SEGURANÇA",
    "nr-28": "FISCALIZAÇÃO E PENALIDADES",
    "nr-29": "NORMA REGULAMENTADORA DE SEGURANÇA E SAÚDE NO TRABALHO PORTUÁRIO",
    "nr-30": "SEGURANÇA E SAÚDE NO TRABALHO AQUAVIÁRIO",
    "nr-31": "SEGURANÇA E SAÚDE NO TRABALHO NA AGRICULTURA, PECUÁRIA, SILVICULTURA, EXPLORAÇÃO FLORESTAL E AQUICULTURA",
    "nr-32": "SEGURANÇA E SAÚDE NO TRABALHO EM SERVIÇOS DE SAÚDE",
    "nr-33": "SEGURANÇA E SAÚDE NOS TRABALHOS EM ESPAÇOS CONFINADOS",
    "nr-34": "CONDIÇÕES E MEIO AMBIENTE DE TRABALHO NA INDÚSTRIA DA CONSTRUÇÃO, REPARAÇÃO E DESMONTE NAVAL",
    "nr-35": "TRABALHO EM ALTURA",
    "nr-36": "SEGURANÇA E SAÚDE NO TRABALHO EM EMPRESAS DE ABATE E PROCESSAMENTO DE CARNES E DERIVADOS",
    "nr-37": "SEGURANÇA E SAÚDE EM PLATAFORMAS DE PETRÓLEO",
    "nr-38": "SEGURANÇA E SAÚDE NO TRABALHO NAS ATIVIDADES DE LIMPEZA URBANA E MANEJO DE RESÍDUOS SÓLIDOS",
}

# Descrição de 1 linha em linguagem de campo (para o prompt de classificação no LFM2.5).
# Escrita para desambiguar da vizinha, priorizando o vocabulário do operário.
NR_ONELINE = {
    "nr-01": "regras gerais, gerenciamento de riscos (GRO/PGR), direito de recusa a trabalho de risco grave, ordem de serviço, treinamento",
    "nr-03": "embargo de obra e interdição de máquina/setor por risco grave e iminente pela fiscalização",
    "nr-04": "SESMT: equipe de engenheiros, técnicos de segurança e médicos do trabalho na empresa",
    "nr-05": "CIPA: comissão interna de prevenção de acidentes, eleição, mapa de risco",
    "nr-06": "EPI: capacete, luva, bota, óculos, cinto — fornecimento gratuito, obrigação de uso, troca e conservação",
    "nr-07": "PCMSO: exames médicos ocupacionais admissional, periódico, demissional, ASO",
    "nr-08": "edificações: piso, pé-direito, escadas, proteção contra intempéries no local de trabalho",
    "nr-09": "exposição a ruído, calor, vibração, agentes químicos e biológicos; limites e controle",
    "nr-10": "eletricidade: choque, arco elétrico, desenergização, aterramento, bloqueio, alta tensão, subestação, rede, poste",
    "nr-11": "transporte e movimentação de materiais, empilhadeira, guindaste, içamento, armazenagem",
    "nr-12": "máquinas e equipamentos: proteção de partes móveis, dispositivos de parada, bloqueio, prensa",
    "nr-13": "caldeiras, vasos de pressão, tubulação e tanque metálico; inspeção e operação",
    "nr-14": "fornos industriais: construção, isolamento e segurança",
    "nr-15": "insalubridade: ruído, calor, agentes nocivos, adicional de insalubridade",
    "nr-16": "periculosidade: adicional para atividade perigosa (energia elétrica, inflamável, explosivo, radiação)",
    "nr-17": "ergonomia: postura, levantamento de peso, mobiliário, ritmo de trabalho, pausas",
    "nr-18": "construção civil: canteiro de obra, andaime, escavação, guarda-corpo, betoneira",
    "nr-19": "explosivos: fabricação, transporte, armazenamento e manuseio",
    "nr-20": "inflamáveis e combustíveis: líquidos e gases, tanque, posto, armazenamento",
    "nr-21": "trabalho a céu aberto: abrigo contra sol e chuva, insolação",
    "nr-22": "mineração: subterrâneo, desmonte, ventilação de mina",
    "nr-23": "proteção contra incêndio: extintor, saída de emergência, brigada",
    "nr-24": "condições sanitárias: banheiro, vestiário, refeitório, água potável no trabalho",
    "nr-25": "resíduos industriais: destinação e descarte de rejeitos",
    "nr-26": "sinalização de segurança: cores, placas, rótulo de produto químico (GHS/FISPQ)",
    "nr-28": "fiscalização e penalidades, multas por infração trabalhista",
    "nr-29": "trabalho portuário: cais, navio, movimentação de carga em porto",
    "nr-30": "trabalho aquaviário: embarcação, navegação",
    "nr-31": "agricultura, pecuária, silvicultura: agrotóxico, trator, trabalho rural",
    "nr-32": "serviços de saúde: hospital, risco biológico, perfurocortante, agulha",
    "nr-33": "espaços confinados: tanque, silo, poço, permissão de entrada, gás, atmosfera IPVS",
    "nr-34": "construção naval: estaleiro, reparo e desmonte de navio",
    "nr-35": "trabalho em altura: acima de 2 metros, cinto paraquedista, talabarte, ancoragem, andaime, escada, poste, torre",
    "nr-36": "frigorífico: abate e processamento de carne, frio, repetitividade",
    "nr-37": "plataforma de petróleo: offshore, exploração marítima",
    "nr-38": "limpeza urbana: coleta de lixo, varrição, manejo de resíduos sólidos",
}

# As 9 NRs mais relevantes para o eletricista de campo da CEMIG (subestação, rede, poste).
ELECTRICIAN_NRS = ["nr-10", "nr-35", "nr-06", "nr-12", "nr-18", "nr-01", "nr-16", "nr-26", "nr-33"]

# Rótulo especial para falas fora do escopo das 36 NRs (ex.: RH, pagamento, dúvida pessoal).
NONE_LABEL = "nenhuma"

# Lista canônica ordenada das classes (36 NRs); 'nenhuma' NÃO entra como classe de treino
# obrigatória do classificador, mas é rótulo válido para falas fora de escopo.
CLASSES: List[str] = sorted(NR_TITLES.keys(), key=lambda c: int(c.split("-")[1]))


def normalize_nr(raw: Optional[str]) -> Optional[str]:
    """Normaliza qualquer variação textual de norma em 'nr-XX' (com zero à esquerda).

    Aceita: 'NR-10', 'nr 10', 'nr10', 'NR10', '10', 'norma 10', 'NR-06'.
    Retorna None se não houver número plausível ou se a NR não existir no corpus.
    Reconhece também o rótulo especial 'nenhuma'.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in ("nenhuma", "none", "nenhum", "n/a", "na", "-"):
        return NONE_LABEL
    m = re.search(r"(\d{1,2})", text)
    if not m:
        return None
    code = f"nr-{int(m.group(1)):02d}"
    return code if code in NR_TITLES else None


def build_prompt_catalog(only: Optional[List[str]] = None) -> str:
    """Monta o catálogo 'nr-XX: descrição' de 1 linha para o prompt de classificação."""
    codes = only or CLASSES
    lines = [f"{c}: {NR_ONELINE[c]}" for c in codes]
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Total de classes NR: {len(CLASSES)}")
    print(f"NRs do eletricista ({len(ELECTRICIAN_NRS)}): {', '.join(ELECTRICIAN_NRS)}")
    # Sanidade da normalização
    for s in ["NR-10", "nr 6", "nr35", "10", "norma 33", "nenhuma", "xyz", "nr-02"]:
        print(f"  normalize_nr({s!r:14}) -> {normalize_nr(s)}")
