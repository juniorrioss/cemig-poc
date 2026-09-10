"""
harness.py — Protocolo único de benchmark para SLMs Tier 1 (Modo Clássico e Tool-Calling).

Executa:
1. Modo RAG Clássico: Injeção direta de top-3 chunks no system prompt.
2. Modo Tool-Calling: Exposição de retriever(query) via endpoint OpenAI-compat do llama-server.
Mede latência, fidelidade, reformulação de query e citação de fontes.
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
DEFAULT_MINIMAL_DB = Path("bench/data/minimal_index.db")
DEFAULT_CORPUS_DB = Path("corpus/index.db")

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
        "server_args": ["-c", "4096", "-t", "4", "--jinja"]
    },
    "lfm2.5-350m": {
        "file": "lfm2.5-350m-q4_k_m.gguf",
        "description": "Liquid LFM2.5 350M",
        "server_args": ["-c", "4096", "-t", "4", "--jinja"]
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
    model_key: str = ""
) -> Dict[str, Any]:
    """Modo A: RAG Clássico com contexto pré-recuperado e injetado no prompt."""
    query = item.get("query_terms") or item["question"]
    chunks = search_retriever(db_path, query, top_k=3)

    context_str = "\n\n".join(
        f"[{idx+1}] {c['text']}" for idx, c in enumerate(chunks)
    )

    system_prompt = (
        "Você é o assistente técnico de campo da CEMIG, especialista em Normas Regulamentadoras (NR-10 e NR-06).\n"
        "Suas diretrizes mandatórias:\n"
        "1. Responda em português brasileiro com precisão técnica e objetividade.\n"
        "2. Baseie sua resposta EXCLUSIVAMENTE nas informações do contexto fornecido abaixo. Não adicione fatos não contidos nas normas.\n"
        "3. É OBRIGATÓRIO citar expressamente a fonte técnica oficial (ex: 'NR-10, item 10.5.1' ou 'NR-06, item 6.3').\n"
        "4. Se a pergunta não puder ser respondida com o contexto fornecido, declare explicitamente: "
        "'Não sei com base nas normas consultadas.' Não tente adivinhar."
    )

    user_content = f"Contexto normativo consultado:\n{context_str}\n\nPergunta do eletricista:\n{item['question']}"

    if model_key.startswith("qwen3.5"):
        # Qwen 3.5 opera nativamente em modo grounded quando o contexto é entregue em role: tool
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["question"]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_ctx",
                    "type": "function",
                    "function": {
                        "name": "retriever",
                        "arguments": json.dumps({"query": item["question"]})
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_ctx",
                "name": "retriever",
                "content": context_str
            }
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    payload = {
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 700
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
            "mode": "classic_rag",
            "item_id": item["id"],
            "question": item["question"],
            "golden_answer": item.get("golden_answer", ""),
            "doc": item.get("doc", ""),
            "section": item.get("section", ""),
            "response": content.strip(),
            "reasoning": reasoning.strip(),
            "latency_s": round(dt, 3),
            "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            "predicted_tok_per_s": round(timings.get("predicted_per_second", 0), 2),
            "chunks_used": [c["id"] for c in chunks],
            "chunks_titles": [c["title"] for c in chunks]
        }
    except Exception as e:
        logger.error("Erro no classic RAG (item %s): %s", item["id"], e)
        return {
            "mode": "classic_rag",
            "item_id": item["id"],
            "question": item["question"],
            "error": str(e),
            "latency_s": round(time.time() - t0, 3),
            "response": ""
        }


def run_tool_calling(
    base_url: str,
    item: Dict[str, Any],
    db_path: Path,
    model_key: str = ""
) -> Dict[str, Any]:
    """Modo B: RAG acionado dinamicamente pelo SLM via Tool-Calling."""
    system_prompt = (
        "Você é o assistente técnico de campo da CEMIG, especialista em Normas Regulamentadoras.\n"
        "Você tem acesso à ferramenta `retriever(query)` para consultar o acervo de NRs de segurança.\n"
        "Suas diretrizes mandatórias:\n"
        "1. Para qualquer dúvida operacional ou regulamentar sobre eletricidade e EPIs, chame a ferramenta `retriever(query)` "
        "com termos de busca técnicos e concisos em português.\n"
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
                "description": "Busca seções das Normas Regulamentadoras (NR-10, NR-06, etc.) usando palavras-chave técnicas.",
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
    tool_query = ""
    tool_args = {}
    chunks_found: List[Dict[str, Any]] = []
    final_content = ""
    reasoning_combined = ""

    try:
        # Round 1: Decisão de chamar a ferramenta
        r1_t0 = time.time()
        resp1 = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.1,
                "max_tokens": 300
            },
            timeout=45
        )
        t_round1 = time.time() - r1_t0
        data1 = resp1.json()

        choice1 = data1.get("choices", [{}])[0]
        msg1 = choice1.get("message", {})
        tool_calls = msg1.get("tool_calls", [])
        reasoning1 = msg1.get("reasoning_content", "") or ""

        if tool_calls:
            called_tool = True
            tc = tool_calls[0]
            try:
                tool_args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                tool_query = tool_args.get("query", "")
            except Exception:
                tool_query = str(tc.get("function", {}).get("arguments", ""))

            # Executa busca no banco SQLite FTS5
            chunks_found = search_retriever(db_path, tool_query or item["question"], top_k=3)
            tool_content = "\n\n".join(
                f"[{i+1}] {c['text']}" for i, c in enumerate(chunks_found)
            ) if chunks_found else "Nenhum resultado normativo encontrado para a busca."

            # Round 2: Envia resultado e instrui a sintetizar a resposta final
            # Limpa reasoning da rodada anterior para não poluir o prompt da rodada 2
            clean_msg1 = {
                "role": "assistant",
                "content": msg1.get("content", "") or "",
                "tool_calls": msg1.get("tool_calls", [])
            }
            messages.append(clean_msg1)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", "call_1"),
                "name": "retriever",
                "content": tool_content
            })

            r2_t0 = time.time()
            resp2 = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 700
                },
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
        else:
            final_content = msg1.get("content", "") or ""
            reasoning_combined = reasoning1
            total_tokens = data1.get("usage", {}).get("total_tokens", 0)

        dt_total = time.time() - t0

        # Análise da query gerada
        query_words = set(re.findall(r"\w+", tool_query.lower()))
        orig_words = set(re.findall(r"\w+", item["question"].lower()))
        reformulated = bool(tool_query and query_words != orig_words)

        # Detecta se gerou termos em inglês
        en_words = {"electrical", "panel", "substation", "gold", "safety", "work", "maintenance", "glove", "shoes"}
        is_english_query = bool(query_words & en_words)

        return {
            "mode": "tool_calling",
            "item_id": item["id"],
            "question": item["question"],
            "golden_answer": item.get("golden_answer", ""),
            "doc": item.get("doc", ""),
            "section": item.get("section", ""),
            "requires_tool": item.get("requires_tool", True),
            "called_tool": called_tool,
            "called_when_expected": (called_tool == item.get("requires_tool", True)),
            "tool_query": tool_query,
            "tool_query_reformulated": reformulated,
            "tool_query_is_english": is_english_query,
            "response": final_content.strip(),
            "reasoning": reasoning_combined,
            "latency_total_s": round(dt_total, 3),
            "latency_round1_s": round(t_round1, 3),
            "latency_round2_s": round(t_round2, 3),
            "total_tokens": total_tokens,
            "chunks_retrieved_count": len(chunks_found),
            "chunks_used": [c["id"] for c in chunks_found]
        }
    except Exception as e:
        logger.error("Erro no tool calling (item %s): %s", item["id"], e)
        return {
            "mode": "tool_calling",
            "item_id": item["id"],
            "question": item["question"],
            "error": str(e),
            "called_tool": False,
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
    db_path: Path,
    models_dir: Path,
    llama_dir: Path,
    output_file: Path,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Executa o benchmark completo para a lista de modelos selecionados."""
    dataset = load_dataset(qa_file)
    if limit and limit > 0:
        dataset = dataset[:limit]

    logger.info("Carregado dataset com %d questões de %s", len(dataset), qa_file)
    logger.info("Usando banco de busca: %s", db_path)

    all_results: Dict[str, Any] = {
        "metadata": {
            "qa_file": str(qa_file),
            "db_path": str(db_path),
            "total_questions": len(dataset),
            "mode": mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "models": {}
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    for model_key in models:
        cfg = AVAILABLE_MODELS.get(model_key)
        if not cfg:
            logger.warning("Modelo desconhecido: %s. Pulando...", model_key)
            continue

        model_file = models_dir / cfg["file"]
        if not model_file.exists():
            logger.error("Arquivo do modelo não encontrado: %s", model_file)
            continue

        logger.info("=========================================================")
        logger.info("Iniciando avaliação do modelo: %s (%s)", model_key, cfg["description"])
        logger.info("=========================================================")

        port = find_free_port()
        server = LlamaServerManager(
            model_path=model_file,
            llama_dir=llama_dir,
            port=port,
            extra_args=cfg["server_args"]
        )

        started = server.start()
        if not started:
            logger.error("Não foi possível iniciar o servidor para %s. Pulando...", model_key)
            continue

        base_url = f"http://localhost:{port}"
        model_results: Dict[str, List[Dict[str, Any]]] = {
            "classic_rag": [],
            "tool_calling": []
        }

        try:
            for idx, item in enumerate(dataset, 1):
                logger.info("[%s] [%d/%d] Pergunta: %s", model_key, idx, len(dataset), item["question"][:60])

                if mode in ("classic", "both"):
                    res_c = run_classic_rag(base_url, item, db_path, model_key=model_key)
                    model_results["classic_rag"].append(res_c)
                    logger.info("  -> Classic RAG: %.2fs (%s tok/s)", res_c.get("latency_s", 0), res_c.get("predicted_tok_per_s", 0))

                if mode in ("tool", "both"):
                    res_t = run_tool_calling(base_url, item, db_path, model_key=model_key)
                    model_results["tool_calling"].append(res_t)
                    tool_flag = "SIM" if res_t.get("called_tool") else "NÃO"
                    logger.info("  -> Tool-calling: %.2fs (chamou tool: %s, query: '%s')", res_t.get("latency_total_s", 0), tool_flag, res_t.get("tool_query", ""))

        finally:
            server.stop()
            time.sleep(1)

        all_results["models"][model_key] = {
            "description": cfg["description"],
            "results": model_results
        }

        # Salva resultados parciais a cada modelo
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info("Resultados parciais gravados em: %s", output_file)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness de benchmark de SLMs Tier 1 (CEMIG POC)")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Modelos a avaliar separados por vírgula ou 'all' (ex: qwen3.5-0.8b,llama-3.2-1b)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "classic", "tool"],
        help="Modo de execução: classic (injetado), tool (tool-calling) ou both (ambos)"
    )
    parser.add_argument(
        "--qa-file",
        type=str,
        default=None,
        help="Caminho do arquivo JSONL de perguntas (padrão: corpus/qa_pairs.jsonl ou bench/data/smoke_qa_20.jsonl)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Caminho do banco SQLite FTS5 index.db (padrão: corpus/index.db ou bench/data/minimal_index.db)"
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
        default="bench/results/harness_results.json",
        help="Arquivo JSON de saída para as respostas e métricas"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o número de questões a executar (útil para smoke test rápido)"
    )

    args = parser.parse_args()

    # Seleção de modelos
    if args.models.strip().lower() == "all":
        selected_models = list(AVAILABLE_MODELS.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Seleção inteligente do QA file (usa corpus se existir, senão minimal smoke)
    if args.qa_file:
        qa_path = Path(args.qa_file)
    elif DEFAULT_CORPUS_QA.exists():
        qa_path = DEFAULT_CORPUS_QA
    else:
        qa_path = DEFAULT_SMOKE_QA

    # Seleção inteligente do DB
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
        db_path=db_path,
        models_dir=Path(args.models_dir),
        llama_dir=Path(args.llama_dir),
        output_file=Path(args.output),
        limit=args.limit
    )


if __name__ == "__main__":
    main()
