"""
harness.py — Protocolo único de benchmark para SLMs Tier 1 (CEMIG POC v2).

Executa:
1. Modo RAG Clássico-Bruto: Busca no SQLite FTS5 com a pergunta bruta do operário (sem query_terms curados) e injeção de top-3 chunks.
2. Modo C (Query-Rewrite + Injeção): Turno 1 pede apenas a query de busca técnica reformulada; executa BM25; Turno 2 sintetiza a resposta com os chunks recuperados.
3. Modo Tool-Calling: Exposição de retriever(query) via endpoint OpenAI-compat do llama-server.
4. Modo Referência (Teto Oracle): RAG clássico com query_terms curados do dataset para medição do teto de retrieval.

Mede:
- Latência por turno e total
- Fidelidade e acerto factual
- Recall de retrieval (Hit@3 do chunk-ouro na busca BM25)
- Taxa de disparo de tool, reformulação de query e citação de fontes
"""

import argparse
import json
import logging
import os
import re
import socket
import sqlite3
import subprocess
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("bench.harness")

DEFAULT_MODELS_DIR = Path.home() / "models-poc"
DEFAULT_LLAMA_DIR = Path.home() / "llama.cpp"
DEFAULT_SMOKE_QA = Path("bench/data/smoke_qa_20.jsonl")
DEFAULT_CORPUS_QA = Path("corpus/qa_pairs.jsonl")
DEFAULT_STRATIFIED_40 = Path("bench/data/qa_pairs_stratified_40.jsonl")
DEFAULT_MINIMAL_DB = Path("bench/data/minimal_index.db")
DEFAULT_CORPUS_DB = Path("corpus/index.db")

KEY_MODELS = ["qwen3.5-0.8b", "lfm2-1.2b", "gemma-3-1b", "lfm2.5-1.2b-instruct", "lfm2.5-1.2b-thinking"]
SECONDARY_MODELS = ["qwen3-0.6b", "lfm2.5-350m", "llama-3.2-1b"]

AVAILABLE_MODELS = {
    "qwen3.5-0.8b": {
        "file": "qwen3.5-0.8b-q4_k_m.gguf",
        "description": "Qwen 3.5 0.8B (Gated DeltaNet hybrid)",
        "server_args": ["-c", "4096", "-t", "4", "--jinja"]
    },
    "qwen3-0.6b": {
        "file": "qwen3-0.6b-q4_k_m.gguf",
        "description": "Qwen 3 0.6B Instruct (Baseline Transformer)",
        "server_args": ["-c", "4096", "-t", "4", "--jinja"]
    },
    "gemma-3-1b": {
        "file": "gemma-3-1b-it-q4_k_m.gguf",
        "description": "Gemma 3 1B IT",
        "server_args": ["-c", "4096", "-t", "4", "--jinja"]
    },
    "lfm2-1.2b": {
        "file": "lfm2-1.2b-rag-q4_k_m.gguf",
        "description": "Liquid LFM2 1.2B RAG",
        "server_args": ["-c", "4096", "-t", "4", "--jinja"],
        "sampling": {"temperature": 0.1}
    },
    "lfm2.5-1.2b-instruct": {
        "file": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        "description": "Liquid LFM2.5 1.2B Instruct",
        "server_args": ["-c", "4096", "-t", "4", "--jinja"],
        "sampling": {"temperature": 0.1}
    },
    "lfm2.5-1.2b-thinking": {
        "file": "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
        "description": "Liquid LFM2.5 1.2B Thinking",
        "server_args": ["-c", "4096", "-t", "4", "--jinja"],
        "sampling": {"temperature": 0.05},
        "is_thinking": True
    },
    "lfm2.5-350m": {
        "file": "lfm2.5-350m-q4_k_m.gguf",
        "description": "Liquid LFM2.5 350M",
        "server_args": ["-c", "4096", "-t", "4", "--jinja"],
        "sampling": {"temperature": 0.1}
    },
    "llama-3.2-1b": {
        "file": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "description": "Meta Llama 3.2 1B Instruct (Controle)",
        "server_args": ["-c", "4096", "-t", "4", "--jinja"]
    }
}

PORTUGUESE_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "ate", "até",
    "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "do", "dos",
    "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa", "essas",
    "esse", "esses", "esta", "estas", "este", "estes", "eu", "foi", "fomos", "foram",
    "ha", "há", "isso", "isto", "ja", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo",
    "meu", "meus", "minha", "minhas", "muito", "na", "nas", "nao", "não", "no", "nos",
    "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "por", "qual", "quais", "quando", "que", "quem",
    "se", "seja", "sem", "so", "só", "sua", "suas", "seu", "seus", "tambem", "também",
    "te", "tem", "têm", "temos", "ter", "teu", "teus", "tua", "tuas", "um", "uma", "voce", "você", "voces", "vocês",
    # Palavras genéricas frequentes em chamadas de busca
    "norma", "normas", "regulamentar", "regulamentares", "regulamentaria", "regulamentarias",
    "seguranca", "segurança", "trabalho", "item", "artigo"
}


def normalize_text(text: str) -> str:
    """Remove acentuação e converte para minúsculas."""
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8").lower()


def sanitize_fts_query(raw_query: str) -> str:
    """Higieniza e formata termos de busca para o SQLite FTS5 com casamento por prefixo."""
    norm = normalize_text(raw_query)
    raw_tokens = re.findall(r"[\w\.-]+", norm)
    filtered = [t for t in raw_tokens if t not in PORTUGUESE_STOPWORDS and len(t) > 2]
    if not filtered:
        filtered = [t for t in raw_tokens if len(t) > 2]

    if not filtered:
        return '""'

    terms = []
    for t in filtered:
        # Aplica truncamento para raiz caso a palavra seja longa (evita desinências flexionais)
        stem = t[:6] if len(t) > 6 else t
        terms.append(f'"{stem}"*')
    return " OR ".join(terms)


def search_retriever(db_path: Path, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Executa a busca lexical BM25 no índice SQLite FTS5."""
    if not db_path.exists():
        logger.warning("Banco de busca não encontrado: %s", db_path)
        return []

    fts_q = sanitize_fts_query(query)
    if fts_q == '""':
        return []

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    sql = """
        SELECT
            c.id,
            c.doc,
            c.section,
            c.title,
            c.page,
            c.text,
            bm25(chunks_fts, 1.5, 3.0, 2.0, 1.0) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score ASC
        LIMIT ?;
    """
    try:
        cur.execute(sql, (fts_q, top_k))
        rows = cur.fetchall()
        results = [
            {
                "id": r["id"],
                "doc": r["doc"],
                "section": r["section"],
                "title": r["title"],
                "page": r["page"],
                "text": r["text"],
                "bm25_score": round(float(r["score"]), 4),
            }
            for r in rows
        ]
        return results
    except Exception as e:
        logger.error("Erro ao consultar FTS5 (%s): %s", fts_q, e)
        return []
    finally:
        con.close()


def check_hit(retrieved_chunk: Dict[str, Any], pair: Dict[str, Any]) -> bool:
    """Verifica se o chunk recuperado corresponde ao gabarito ouro."""
    # 1. Correspondência exata por chunk_id primário ou lista de chunks relevantes
    if retrieved_chunk.get("id") == pair.get("chunk_id"):
        return True
    if pair.get("relevant_chunk_ids") and retrieved_chunk.get("id") in pair.get("relevant_chunk_ids", []):
        return True

    # 2. Correspondência estrutural por documento e seção/item
    target_doc = (pair.get("doc") or "").lower().strip()
    target_sec = (pair.get("section") or "").lower().strip()
    ret_doc = (retrieved_chunk.get("doc") or "").lower().strip()
    ret_sec = (retrieved_chunk.get("section") or "").lower().strip()
    ret_text = (retrieved_chunk.get("text") or "").lower().strip()

    if target_doc and ret_doc and target_doc == ret_doc:
        if target_sec and (target_sec in ret_sec or ret_sec in target_sec):
            return True
        if target_sec and target_sec in ret_text:
            return True

    return False


def extract_clean_query(raw_text: str, fallback: str) -> str:
    """Extrai e limpa os termos de busca gerados no Turno 1 do Modo C."""
    if not raw_text or not raw_text.strip():
        return fallback
    text = raw_text.strip()
    # Remove formatações como negrito, marcadores e aspas
    text = re.sub(r"[\*#_`\"]", " ", text)
    # Remove preâmbulos comuns gerados por assistentes
    text = re.sub(r"^(?:aqui estão|palavras-chave|termos de busca|termos|palavras)[\w\s]*[:\s\-]+", "", text, flags=re.IGNORECASE)
    lines = [re.sub(r"^[\-\•\d\.]+\s*", "", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not line.lower().startswith(("aqui ", "com base", "para buscar", "segue", "desculpe", "essas são", "como assistente"))]
    cleaned = " ".join(lines).strip()
    if not cleaned or len(cleaned) < 3:
        return fallback
    return cleaned


def find_free_port() -> int:
    """Encontra uma porta TCP local livre."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class LlamaServerManager:
    """Gerenciador do ciclo de vida do processo llama-server."""

    def __init__(
        self,
        model_path: Path,
        llama_dir: Path,
        port: int,
        extra_args: List[str]
    ):
        self.model_path = model_path
        self.llama_dir = llama_dir
        self.port = port
        self.extra_args = extra_args
        self.process: Optional[subprocess.Popen] = None

    def start(self, timeout: float = 30.0) -> bool:
        server_bin = self.llama_dir / "bin" / "llama-server"
        if not server_bin.exists():
            server_bin = self.llama_dir / "build" / "bin" / "llama-server"
        if not server_bin.exists():
            raise FileNotFoundError(f"llama-server não encontrado em: {server_bin}")

        cmd = [
            str(server_bin),
            "-m", str(self.model_path),
            "--port", str(self.port),
            *self.extra_args
        ]
        logger.info("Iniciando llama-server: %s (porta %d)", self.model_path.name, self.port)
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        t_start = time.time()
        url = f"http://localhost:{self.port}/health"
        while time.time() - t_start < timeout:
            try:
                r = requests.get(url, timeout=1)
                if r.status_code == 200:
                    logger.info("llama-server pronto em %.1fs", time.time() - t_start)
                    return True
            except requests.RequestException:
                pass
            time.sleep(0.5)

        logger.error("Timeout ao aguardar llama-server inicializar.")
        self.stop()
        return False

    def stop(self) -> None:
        if self.process:
            logger.info("Encerrando llama-server (PID %d)...", self.process.pid)
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None


def run_classic_rag(
    base_url: str,
    item: Dict[str, Any],
    db_path: Path,
    model_key: str = "",
    use_gold_query: bool = False,
    top_k: int = 3,
    suffix: str = "",
    engine: str = "cpu"
) -> Dict[str, Any]:
    """
    Modo A: RAG Clássico com contexto pré-recuperado e injetado no prompt.
    Se use_gold_query=False (padrão): busca com a fala bruta do operário (item["question"]).
    Se use_gold_query=True: braço de referência com query_terms curados (teto/oracle).
    """
    if use_gold_query:
        query = item.get("query_terms") or item["question"]
        mode_name = f"classic_gold{suffix}"
    else:
        query = item["question"]
        mode_name = f"classic_rag{suffix}"

    chunks = search_retriever(db_path, query, top_k=top_k)
    retrieval_hit = any(check_hit(c, item) for c in chunks)

    context_str = "\n\n".join(
        f"[{idx+1}] {c['text']}" for idx, c in enumerate(chunks)
    ) if chunks else "Nenhum contexto normativo recuperado para a consulta."

    system_prompt = (
        "Você é o assistente técnico de campo da CEMIG, especialista em Normas Regulamentadoras (NR-10, NR-06, NR-35, NR-12 e NR-18).\n"
        "Suas diretrizes mandatórias:\n"
        "1. Responda em português brasileiro com precisão técnica e objetividade.\n"
        "2. Baseie sua resposta EXCLUSIVAMENTE nas informações do contexto normativo fornecido abaixo. Não adicione fatos não contidos nas normas.\n"
        "3. É OBRIGATÓRIO citar expressamente a fonte técnica oficial (ex: 'NR-10, item 10.5.1' ou 'NR-06, item 6.3').\n"
        "4. Se a pergunta não puder ser respondida com o contexto fornecido, declare explicitamente: "
        "'Não sei com base nas normas consultadas.' Não tente adivinhar."
    )

    user_content = f"Contexto normativo consultado:\n{context_str}\n\nPergunta do eletricista:\n{item['question']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload = {
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 700,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    t0 = time.time()
    try:
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=45)
        dt = time.time() - t0
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
        timings = data.get("timings", {})

        return {
            "mode": mode_name,
            "engine": engine,
            "item_id": item["id"],
            "question": item["question"],
            "golden_answer": item.get("golden_answer", ""),
            "doc": item.get("doc", ""),
            "section": item.get("section", ""),
            "query_used": query,
            "retrieval_hit": retrieval_hit,
            "response": content.strip(),
            "reasoning": reasoning.strip(),
            "latency_s": round(dt, 3),
            "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            "predicted_tok_per_s": round(timings.get("predicted_per_second", 0), 2),
            "chunks_used": [c["id"] for c in chunks],
            "chunks_titles": [c["title"] for c in chunks],
            "chunks_docs": [c["doc"] for c in chunks],
            "chunks_sections": [c["section"] for c in chunks]
        }
    except Exception as e:
        logger.error("Erro no %s (item %s): %s", mode_name, item["id"], e)
        return {
            "mode": mode_name,
            "item_id": item["id"],
            "question": item["question"],
            "query_used": query,
            "retrieval_hit": retrieval_hit,
            "error": str(e),
            "latency_s": round(time.time() - t0, 3),
            "response": ""
        }


def run_query_rewrite_inject(
    base_url: str,
    item: Dict[str, Any],
    db_path: Path,
    model_key: str = "",
    top_k: int = 3,
    suffix: str = "",
    engine: str = "cpu"
) -> Dict[str, Any]:
    """
    Modo C: Query-rewriting em turno curto + injeção de chunks recuperados no turno 2.
    Turno 1 pede apenas palavras-chave técnicas reformuladas; executa BM25;
    Turno 2 sintetiza a resposta final com os chunks injetados.
    """
    mode_name = f"query_rewrite_inject{suffix}"
    rewrite_sys = (
        "Você é um especialista em busca técnica nas Normas Regulamentadoras (NRs de segurança do trabalho).\n"
        "Sua única tarefa é extrair e converter a dúvida do trabalhador em palavras-chave técnicas e conceituais para busca BM25 no acervo das NRs.\n"
        "Diretrizes mandatórias:\n"
        "1. Retorne APENAS de 3 a 6 palavras-chave técnicas na mesma linha, separadas por vírgula ou espaço.\n"
        "2. Foque nos termos específicos da situação, procedimentos, equipamentos, riscos ou medidas de proteção aplicáveis.\n"
        "3. NÃO inclua saudações, preâmbulos, justificativas ou listas longas de nomes de normas.\n"
        "4. NÃO responda à dúvida nesta etapa; retorne estritamente os termos de busca."
    )

    t0 = time.time()
    t1_start = time.time()
    raw_rewrite = ""
    query_rewritten = item["question"]
    try:
        payload_t1 = {
            "messages": [
                {"role": "system", "content": rewrite_sys},
                {"role": "user", "content": f"Dúvida do trabalhador: {item['question']}\nTermos técnicos para busca:"}
            ],
            "temperature": 0.1,
            "max_tokens": 50,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        resp1 = requests.post(f"{base_url}/v1/chat/completions", json=payload_t1, timeout=30)
        t_round1 = time.time() - t1_start
        data1 = resp1.json()
        msg1 = data1.get("choices", [{}])[0].get("message", {})
        raw_rewrite = msg1.get("content", "") or ""
        if not raw_rewrite.strip() and msg1.get("reasoning_content"):
            raw_rewrite = msg1.get("reasoning_content", "")
        query_rewritten = extract_clean_query(raw_rewrite, fallback=item["question"])
    except Exception as e:
        logger.warning("Erro no Turno 1 de Modo C (item %s): %s. Usando fallback.", item["id"], e)
        t_round1 = time.time() - t1_start
        query_rewritten = item["question"]

    # Executa busca BM25 com query reformulada
    chunks = search_retriever(db_path, query_rewritten, top_k=top_k)
    retrieval_hit = any(check_hit(c, item) for c in chunks)

    # Turno 2: Injeção dos chunks e síntese da resposta final
    context_str = "\n\n".join(
        f"[{idx+1}] {c['text']}" for idx, c in enumerate(chunks)
    ) if chunks else "Nenhum contexto normativo recuperado para a consulta."

    system_prompt = (
        "Você é o assistente técnico de campo da CEMIG, especialista em Normas Regulamentadoras (NR-10, NR-06, NR-35, NR-12 e NR-18).\n"
        "Suas diretrizes mandatórias:\n"
        "1. Responda em português brasileiro com precisão técnica e objetividade.\n"
        "2. Baseie sua resposta EXCLUSIVAMENTE nas informações do contexto normativo fornecido abaixo. Não adicione procedimentos não contidos nas normas.\n"
        "3. É OBRIGATÓRIO citar expressamente a fonte técnica oficial (ex: 'NR-10, item 10.5.1' ou 'NR-06, item 6.3').\n"
        "4. Se a pergunta não puder ser respondida com o contexto fornecido, declare explicitamente: "
        "'Não sei com base nas normas consultadas.' Não tente adivinhar."
    )

    user_content = f"Contexto normativo consultado:\n{context_str}\n\nPergunta do eletricista:\n{item['question']}"

    messages_t2 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload_t2 = {
        "messages": messages_t2,
        "temperature": 0.1,
        "max_tokens": 700,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    t2_start = time.time()
    try:
        resp2 = requests.post(f"{base_url}/v1/chat/completions", json=payload_t2, timeout=45)
        t_round2 = time.time() - t2_start
        data2 = resp2.json()
        choice2 = data2["choices"][0]
        msg2 = choice2["message"]
        content = msg2.get("content", "") or ""
        reasoning = msg2.get("reasoning_content", "") or ""
        timings = data2.get("timings", {})
        dt_total = time.time() - t0

        return {
            "mode": mode_name,
            "engine": engine,
            "item_id": item["id"],
            "question": item["question"],
            "golden_answer": item.get("golden_answer", ""),
            "doc": item.get("doc", ""),
            "section": item.get("section", ""),
            "raw_rewrite": raw_rewrite.strip(),
            "query_rewritten": query_rewritten,
            "retrieval_hit": retrieval_hit,
            "response": content.strip(),
            "reasoning": reasoning.strip(),
            "latency_round1_s": round(t_round1, 3),
            "latency_round2_s": round(t_round2, 3),
            "latency_total_s": round(dt_total, 3),
            "prompt_tokens": data2.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": data2.get("usage", {}).get("completion_tokens", 0),
            "predicted_tok_per_s": round(timings.get("predicted_per_second", 0), 2),
            "chunks_used": [c["id"] for c in chunks],
            "chunks_titles": [c["title"] for c in chunks],
            "chunks_docs": [c["doc"] for c in chunks],
            "chunks_sections": [c["section"] for c in chunks]
        }
    except Exception as e:
        logger.error("Erro no Turno 2 de Modo C (item %s): %s", item["id"], e)
        return {
            "mode": "query_rewrite_inject",
            "item_id": item["id"],
            "question": item["question"],
            "error": str(e),
            "raw_rewrite": raw_rewrite.strip(),
            "query_rewritten": query_rewritten,
            "retrieval_hit": retrieval_hit,
            "latency_round1_s": round(t_round1, 3),
            "latency_round2_s": round(time.time() - t2_start, 3),
            "latency_total_s": round(time.time() - t0, 3),
            "response": ""
        }


def parse_tool_call(msg: Dict[str, Any]) -> Tuple[bool, str, str, str]:
    """
    Detecta e extrai chamadas de ferramenta em formato OpenAI JSON ou Pythonic (Liquid AI docs.liquid.ai).
    Retorna: (called_tool: bool, format: 'json'|'pythonic'|'none', query: str, fn_name: str)
    """
    tc = msg.get("tool_calls")
    if tc and len(tc) > 0:
        fn = tc[0].get("function", {})
        fn_name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw)
            q = args.get("query", "")
        except Exception:
            q = str(args_raw)
        return True, "json", q, fn_name

    content = msg.get("content", "") or ""
    m = re.search(r'\[?\s*(\w+)\s*\(\s*(?:query\s*=\s*)?["\']([^"\']+)["\']\s*\)\s*\]?', content)
    if m:
        fn_name = m.group(1)
        q = m.group(2)
        return True, "pythonic", q, fn_name

    return False, "none", "", ""


def run_tool_calling(
    base_url: str,
    item: Dict[str, Any],
    db_path: Path,
    model_key: str = "",
    top_k: int = 3,
    suffix: str = "",
    engine: str = "cpu"
) -> Dict[str, Any]:
    """Modo B: RAG acionado dinamicamente pelo SLM via Tool-Calling (JSON ou Pythonic)."""
    mode_name = f"tool_calling{suffix}"
    cfg = AVAILABLE_MODELS.get(model_key, {})
    is_thinking = cfg.get("is_thinking", False)
    sampling = cfg.get("sampling", {})
    temp = sampling.get("temperature", 0.1)
    system_prompt = (
        "Você é o assistente técnico de campo da CEMIG, especialista em Normas Regulamentadoras.\n"
        "Você tem acesso à ferramenta `retriever(query)` para consultar o acervo de NRs de segurança (NR-10, NR-06, NR-35, NR-12, NR-18).\n"
        "Suas diretrizes mandatórias:\n"
        "1. Para qualquer dúvida operacional ou regulamentar sobre eletricidade, trabalho em altura, máquinas ou EPIs, chame a ferramenta `retriever(query)` com termos de busca técnicos e concisos em português.\n"
        "2. Após obter o retorno da ferramenta, elabore a resposta final em português brasileiro.\n"
        "3. É OBRIGATÓRIO citar expressamente a fonte (norma e item, ex: 'NR-10, item 10.2.9.3').\n"
        "4. Se a informação não constar no resultado ou se o tema for alheio às normas, declare: "
        "'Não sei com base nas normas consultadas.'"
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "retriever",
                "description": "Busca seções das Normas Regulamentadoras (NR-10, NR-06, NR-35, NR-12, NR-18) usando palavras-chave técnicas.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Termos de busca técnicos em português (ex: 'desenergizacao ordem sequencia')"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": item["question"]}
    ]

    t0 = time.time()
    t_round1 = 0.0
    t_round2 = 0.0
    called_tool = False
    tool_call_format = "none"
    tool_query = ""
    chunks_found: List[Dict[str, Any]] = []
    retrieval_hit = False
    final_content = ""
    reasoning_combined = ""
    thinking_tokens = 0
    answer_tokens = 0

    try:
        # Round 1: Decisão de chamar a ferramenta
        r1_t0 = time.time()
        payload1 = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temp,
            "max_tokens": 400
        }
        if not is_thinking:
            payload1["chat_template_kwargs"] = {"enable_thinking": False}

        resp1 = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload1,
            timeout=45
        )
        t_round1 = time.time() - r1_t0
        data1 = resp1.json()

        choice1 = data1.get("choices", [{}])[0]
        msg1 = choice1.get("message", {})
        reasoning1 = msg1.get("reasoning_content", "") or ""

        called_tool, tool_call_format, tool_query, fn_name = parse_tool_call(msg1)

        if called_tool:
            # Executa busca no banco SQLite FTS5
            chunks_found = search_retriever(db_path, tool_query or item["question"], top_k=top_k)
            retrieval_hit = any(check_hit(c, item) for c in chunks_found)

            tool_content = "\n\n".join(
                f"[{i+1}] {c['text']}" for i, c in enumerate(chunks_found)
            ) if chunks_found else "Nenhum resultado normativo encontrado para a busca."

            # Round 2: Envia resultado e instrui a sintetizar a resposta final
            if tool_call_format == "json":
                clean_msg1 = {
                    "role": "assistant",
                    "content": msg1.get("content", "") or "",
                    "tool_calls": msg1.get("tool_calls", [])
                }
                messages.append(clean_msg1)
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg1.get("tool_calls", [{}])[0].get("id", "call_1"),
                    "name": "retriever",
                    "content": tool_content
                })
            else:
                # Formato Pythonic (Liquid AI)
                messages.append({
                    "role": "assistant",
                    "content": msg1.get("content", "") or ""
                })
                messages.append({
                    "role": "tool",
                    "name": fn_name or "retriever",
                    "content": tool_content
                })

            r2_t0 = time.time()
            payload2 = {
                "messages": messages,
                "temperature": temp,
                "max_tokens": 1200 if is_thinking else 700
            }
            if not is_thinking:
                payload2["chat_template_kwargs"] = {"enable_thinking": False}

            resp2 = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload2,
                timeout=45
            )
            t_round2 = time.time() - r2_t0
            data2 = resp2.json()
            choice2 = data2.get("choices", [{}])[0]
            final_content = choice2.get("message", {}).get("content", "") or ""
            reasoning2 = choice2.get("message", {}).get("reasoning_content", "") or ""
            reasoning_combined = f"R1: {reasoning1} | R2: {reasoning2}".strip()

            total_tokens = (
                data1.get("usage", {}).get("total_tokens", 0) +
                data2.get("usage", {}).get("total_tokens", 0)
            )
            if is_thinking and reasoning2:
                thinking_tokens = len(re.findall(r"\w+", reasoning2))
                answer_tokens = len(re.findall(r"\w+", final_content))
        else:
            final_content = msg1.get("content", "") or ""
            reasoning_combined = reasoning1
            total_tokens = data1.get("usage", {}).get("total_tokens", 0)
            retrieval_hit = False

        dt_total = time.time() - t0

        # Análise da query gerada
        query_words = set(re.findall(r"\w+", tool_query.lower()))
        orig_words = set(re.findall(r"\w+", item["question"].lower()))
        reformulated = bool(tool_query and query_words != orig_words)

        # Detecta se gerou termos em inglês
        en_words = {"electrical", "panel", "substation", "gold", "safety", "work", "maintenance", "glove", "shoes"}
        is_english_query = bool(query_words & en_words)

        return {
            "mode": mode_name,
            "engine": engine,
            "item_id": item["id"],
            "question": item["question"],
            "golden_answer": item.get("golden_answer", ""),
            "doc": item.get("doc", ""),
            "section": item.get("section", ""),
            "requires_tool": item.get("requires_tool", True),
            "called_tool": called_tool,
            "tool_call_format": tool_call_format,
            "called_when_expected": (called_tool == item.get("requires_tool", True)),
            "tool_query": tool_query,
            "tool_query_reformulated": reformulated,
            "tool_query_is_english": is_english_query,
            "retrieval_hit": retrieval_hit,
            "response": final_content.strip(),
            "reasoning": reasoning_combined,
            "thinking_tokens": thinking_tokens,
            "answer_tokens": answer_tokens,
            "latency_total_s": round(dt_total, 3),
            "latency_round1_s": round(t_round1, 3),
            "latency_round2_s": round(t_round2, 3),
            "total_tokens": total_tokens,
            "chunks_retrieved_count": len(chunks_found),
            "chunks_used": [c["id"] for c in chunks_found],
            "chunks_docs": [c["doc"] for c in chunks_found],
            "chunks_sections": [c["section"] for c in chunks_found]
        }
    except Exception as e:
        logger.error("Erro no tool calling (item %s): %s", item["id"], e)
        return {
            "mode": mode_name,
            "item_id": item["id"],
            "question": item["question"],
            "error": str(e),
            "called_tool": False,
            "retrieval_hit": False,
            "latency_total_s": round(time.time() - t0, 3),
            "response": ""
        }


def load_dataset(qa_file: Path) -> List[Dict[str, Any]]:
    """Carrega dataset JSONL de avaliação."""
    if not qa_file.exists():
        raise FileNotFoundError(f"Arquivo de avaliação não encontrado: {qa_file}")

    items = []
    with open(qa_file, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s:
                items.append(json.loads(line_s))
    return items


def run_benchmark(
    models: List[str],
    mode: str,
    qa_file: Path,
    secondary_qa_file: Optional[Path],
    db_path: Path,
    models_dir: Path,
    llama_dir: Path,
    output_file: Path,
    limit: Optional[int] = None,
    stratified_split: bool = True,
    force: bool = False,
    include_topk2: bool = False,
    use_gpu: bool = False
) -> Dict[str, Any]:
    """
    Executa o benchmark completo para a lista de modelos selecionados.
    Se stratified_split=True: modelos-chave rodam no dataset principal (101 itens)
    e modelos secundários rodam no subset estratificado (40 itens).
    Se include_topk2=True: avalia também variante top-2 chunks nos modelos-chave.
    """
    key_dataset = load_dataset(qa_file)
    sec_dataset = load_dataset(secondary_qa_file) if secondary_qa_file and secondary_qa_file.exists() else key_dataset

    if limit and limit > 0:
        key_dataset = key_dataset[:limit]
        sec_dataset = sec_dataset[:limit]

    logger.info("Dataset principal carregado: %d questões (%s)", len(key_dataset), qa_file)
    if secondary_qa_file and secondary_qa_file.exists():
        logger.info("Dataset secundário (estratificado): %d questões (%s)", len(sec_dataset), secondary_qa_file)
    logger.info("Usando banco de busca SQLite FTS5: %s", db_path)

    # Carrega resultados existentes se houver
    all_results: Dict[str, Any] = {
        "metadata": {
            "qa_file": str(qa_file),
            "secondary_qa_file": str(secondary_qa_file) if secondary_qa_file else None,
            "db_path": str(db_path),
            "mode": mode,
            "stratified_split": stratified_split,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "models": {}
    }
    if output_file.exists() and not force:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                all_results["models"] = existing.get("models", {})
                logger.info("Carregados resultados prévios de %d modelos de %s", len(all_results["models"]), output_file)
        except Exception as e:
            logger.warning("Não foi possível ler resultados anteriores: %s", e)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Mapeia modos a executar
    execute_classic = mode in ("all_three", "all", "classic", "both", "with_topk2")
    execute_rewrite = mode in ("all_three", "all", "c", "rewrite", "query_rewrite_inject", "with_topk2")
    execute_tool = mode in ("all_three", "all", "tool", "both", "with_topk2")
    execute_gold = mode in ("all", "gold", "classic_gold")
    check_topk2 = include_topk2 or mode in ("with_topk2", "topk2", "topk2_only")

    for model_key in models:
        cfg = AVAILABLE_MODELS.get(model_key)
        if not cfg:
            logger.warning("Modelo desconhecido: %s. Pulando...", model_key)
            continue

        model_file = models_dir / cfg["file"]
        if not model_file.exists():
            logger.error("Arquivo do modelo não encontrado: %s", model_file)
            continue

        # Seleciona o dataset apropriado para o modelo
        if stratified_split and model_key in SECONDARY_MODELS:
            current_dataset = sec_dataset
            logger.info("Modelo %s é Tier 2 -> usando subset estratificado de %d questões", model_key, len(current_dataset))
        else:
            current_dataset = key_dataset
            logger.info("Modelo %s é Tier 1 -> usando conjunto completo de %d questões", model_key, len(current_dataset))

        # Verifica se já está completo
        existing_model_data = all_results["models"].get(model_key, {}).get("results", {})
        modes_to_run = []
        if execute_classic and (force or len(existing_model_data.get("classic_rag", [])) < len(current_dataset)):
            modes_to_run.append("classic_rag")
        if execute_rewrite and (force or len(existing_model_data.get("query_rewrite_inject", [])) < len(current_dataset)):
            modes_to_run.append("query_rewrite_inject")
        if execute_tool and (force or len(existing_model_data.get("tool_calling", [])) < len(current_dataset)):
            modes_to_run.append("tool_calling")
        if execute_gold and (force or len(existing_model_data.get("classic_gold", [])) < len(current_dataset)):
            modes_to_run.append("classic_gold")

        # Variante top-2 para modelos-chave (orçamento Galaxy S24+)
        is_key_model = model_key in KEY_MODELS
        if check_topk2 and is_key_model:
            if force or len(existing_model_data.get("query_rewrite_inject_topk2", [])) < len(current_dataset):
                modes_to_run.append("query_rewrite_inject_topk2")
            if force or len(existing_model_data.get("classic_rag_topk2", [])) < len(current_dataset):
                modes_to_run.append("classic_rag_topk2")

        if not modes_to_run:
            logger.info("Modelo %s já possui todos os modos avaliados. Pulando...", model_key)
            continue

        logger.info("=========================================================")
        logger.info("Iniciando avaliação do modelo: %s (%s)", model_key, cfg["description"])
        logger.info("Modos a executar: %s", ", ".join(modes_to_run))
        logger.info("=========================================================")

        port = find_free_port()
        server_llama_dir = Path.home() / "llama.cpp" / "build-cuda" if use_gpu else llama_dir
        extra_args = [*cfg["server_args"]]
        if use_gpu and "-ngl" not in extra_args:
            extra_args.extend(["-ngl", "99"])
        engine = "cuda" if use_gpu else "cpu"

        server = LlamaServerManager(
            model_path=model_file,
            llama_dir=server_llama_dir,
            port=port,
            extra_args=extra_args
        )

        started = server.start()
        if not started:
            logger.error("Não foi possível iniciar o servidor para %s. Pulando...", model_key)
            continue

        base_url = f"http://localhost:{port}"
        model_results: Dict[str, List[Dict[str, Any]]] = {
            "classic_rag": existing_model_data.get("classic_rag", []) if not force else [],
            "query_rewrite_inject": existing_model_data.get("query_rewrite_inject", []) if not force else [],
            "tool_calling": existing_model_data.get("tool_calling", []) if not force else [],
            "classic_gold": existing_model_data.get("classic_gold", []) if not force else [],
            "query_rewrite_inject_topk2": existing_model_data.get("query_rewrite_inject_topk2", []) if not force else [],
            "classic_rag_topk2": existing_model_data.get("classic_rag_topk2", []) if not force else []
        }

        try:
            for idx, item in enumerate(current_dataset, 1):
                logger.info("[%s] [%d/%d] (%s) %s", model_key, idx, len(current_dataset), item.get("id"), item["question"][:55])

                # 1. Modo Clássico-Bruto (sem query_terms)
                if "classic_rag" in modes_to_run:
                    # Só executa se ainda não tiver feito esse item
                    if len(model_results["classic_rag"]) < idx:
                        res_c = run_classic_rag(base_url, item, db_path, model_key=model_key, use_gold_query=False, top_k=3, engine=engine)
                        model_results["classic_rag"].append(res_c)
                        hit_str = "HIT" if res_c.get("retrieval_hit") else "MISS"
                        logger.info("  -> Classic-Raw: %.2fs | Retrieval: %s", res_c.get("latency_s", 0), hit_str)

                # 2. Modo C (Query Rewrite + Inject)
                if "query_rewrite_inject" in modes_to_run:
                    if len(model_results["query_rewrite_inject"]) < idx:
                        res_rw = run_query_rewrite_inject(base_url, item, db_path, model_key=model_key, top_k=3, engine=engine)
                        model_results["query_rewrite_inject"].append(res_rw)
                        hit_str = "HIT" if res_rw.get("retrieval_hit") else "MISS"
                        logger.info("  -> Modo C (Rewrite): %.2fs (T1: %.2fs, T2: %.2fs) | Retrieval: %s | Query: '%s'",
                                    res_rw.get("latency_total_s", 0), res_rw.get("latency_round1_s", 0),
                                    res_rw.get("latency_round2_s", 0), hit_str, res_rw.get("query_rewritten", "")[:40])

                # 3. Modo Tool-Calling
                if "tool_calling" in modes_to_run:
                    if len(model_results["tool_calling"]) < idx:
                        res_t = run_tool_calling(base_url, item, db_path, model_key=model_key, top_k=3, engine=engine)
                        model_results["tool_calling"].append(res_t)
                        tool_flag = "SIM" if res_t.get("called_tool") else "NÃO"
                        hit_str = "HIT" if res_t.get("retrieval_hit") else "MISS"
                        logger.info("  -> Tool-calling: %.2fs (Chamou: %s, Formato: %s, Retrieval: %s, Query: '%s')",
                                    res_t.get("latency_total_s", 0), tool_flag, res_t.get("tool_call_format", "none"),
                                    hit_str, res_t.get("tool_query", "")[:40])

                # 4. Modo Referência / Teto Oracle (opcional)
                if "classic_gold" in modes_to_run:
                    if len(model_results["classic_gold"]) < idx:
                        res_g = run_classic_rag(base_url, item, db_path, model_key=model_key, use_gold_query=True, top_k=3, engine=engine)
                        model_results["classic_gold"].append(res_g)

                # 5. Modo C Top-2 (Variante top-2 chunks para perfil S24+)
                if "query_rewrite_inject_topk2" in modes_to_run:
                    if len(model_results["query_rewrite_inject_topk2"]) < idx:
                        res_rw2 = run_query_rewrite_inject(base_url, item, db_path, model_key=model_key, top_k=2, suffix="_topk2", engine=engine)
                        model_results["query_rewrite_inject_topk2"].append(res_rw2)
                        hit_str = "HIT" if res_rw2.get("retrieval_hit") else "MISS"
                        logger.info("  -> Modo C (topk2): %.2fs (T1: %.2fs, T2: %.2fs) | Retrieval: %s",
                                    res_rw2.get("latency_total_s", 0), res_rw2.get("latency_round1_s", 0),
                                    res_rw2.get("latency_round2_s", 0), hit_str)

                # 6. Modo Clássico Top-2
                if "classic_rag_topk2" in modes_to_run:
                    if len(model_results["classic_rag_topk2"]) < idx:
                        res_c2 = run_classic_rag(base_url, item, db_path, model_key=model_key, use_gold_query=False, top_k=2, suffix="_topk2", engine=engine)
                        model_results["classic_rag_topk2"].append(res_c2)
                        hit_str = "HIT" if res_c2.get("retrieval_hit") else "MISS"
                        logger.info("  -> Classic (topk2): %.2fs | Retrieval: %s", res_c2.get("latency_s", 0), hit_str)

                # Salva a cada 10 itens como salvaguarda
                if idx % 10 == 0 or idx == len(current_dataset):
                    all_results["models"][model_key] = {
                        "description": cfg["description"],
                        "dataset_size": len(current_dataset),
                        "results": {k: v for k, v in model_results.items() if v}
                    }
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)

        finally:
            server.stop()
            time.sleep(1)

        all_results["models"][model_key] = {
            "description": cfg["description"],
            "dataset_size": len(current_dataset),
            "results": {k: v for k, v in model_results.items() if v}
        }

        # Salva resultados consolidados do modelo
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info("Resultados de %s gravados em: %s", model_key, output_file)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness de benchmark de SLMs Tier 1 (CEMIG POC v2)")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Modelos a avaliar separados por vírgula ou 'all' (ex: qwen3.5-0.8b,lfm2-1.2b,gemma-3-1b)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all_three",
        choices=["all_three", "all", "classic", "rewrite", "c", "tool", "gold", "both"],
        help="Modo de execução: all_three (clássico-bruto, Modo C e tool), all (+ oracle gold), classic, rewrite/c, tool, gold"
    )
    parser.add_argument(
        "--qa-file",
        type=str,
        default=None,
        help="Caminho do arquivo JSONL principal de perguntas (padrão: corpus/qa_pairs.jsonl)"
    )
    parser.add_argument(
        "--secondary-qa-file",
        type=str,
        default=None,
        help="Caminho do arquivo JSONL secundário estratificado (padrão: bench/data/qa_pairs_stratified_40.jsonl)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Caminho do banco SQLite FTS5 index.db (padrão: corpus/index.db)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help="Diretório contendo os GGUFs baixados"
    )
    parser.add_argument(
        "--llama-dir",
        type=str,
        default=str(DEFAULT_LLAMA_DIR),
        help="Diretório raiz de compilação do llama.cpp"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bench/results/v2/harness_results_v2.json",
        help="Arquivo JSON de saída para as respostas e métricas"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o número de questões a executar (útil para smoke test rápido)"
    )
    parser.add_argument(
        "--no-stratified-split",
        action="store_true",
        help="Desativa o split estratificado (força todos os modelos a rodar no dataset principal)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve avaliações anteriores mesmo se já existirem no arquivo de saída"
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Usa aceleracao GPU (RTX 5070 via llama-server build-cuda -ngl 99)"
    )
    parser.add_argument(
        "--include-topk2",
        action="store_true",
        help="Executa também as variantes top-2 chunks nos 3 modelos-chave (orçamento Galaxy S24+)"
    )

    args = parser.parse_args()

    # Seleção de modelos
    if args.models.strip().lower() == "all":
        selected_models = list(AVAILABLE_MODELS.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Seleção do dataset principal
    if args.qa_file:
        qa_path = Path(args.qa_file)
    elif DEFAULT_CORPUS_QA.exists():
        qa_path = DEFAULT_CORPUS_QA
    else:
        qa_path = DEFAULT_SMOKE_QA

    # Seleção do dataset secundário
    if args.secondary_qa_file:
        sec_qa_path = Path(args.secondary_qa_file)
    elif DEFAULT_STRATIFIED_40.exists():
        sec_qa_path = DEFAULT_STRATIFIED_40
    else:
        sec_qa_path = None

    # Seleção do DB
    if args.db:
        db_path = Path(args.db)
    elif DEFAULT_CORPUS_DB.exists():
        db_path = DEFAULT_CORPUS_DB
    else:
        db_path = DEFAULT_MINIMAL_DB

    run_benchmark(
        models=selected_models,
        mode=args.mode,
        qa_file=qa_path,
        secondary_qa_file=sec_qa_path,
        db_path=db_path,
        models_dir=Path(args.models_dir),
        llama_dir=Path(args.llama_dir),
        output_file=Path(args.output),
        limit=args.limit,
        stratified_split=not args.no_stratified_split,
        force=args.force,
        include_topk2=args.include_topk2,
        use_gpu=args.use_gpu
    )


if __name__ == "__main__":
    main()
