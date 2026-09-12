"""
01_batch_process.py — Upload dos PDFs, batch process no Document AI Form Parser e download dos shards JSON.

Fluxo (tudo async/LRO, paralelo, com checkpoint incremental):
  1. Sobe os PDFs de --pdf-dir para gs://<bucket>/<prefix>/input/.
  2. Dispara UM batch_process_documents (LRO) sobre todos os PDFs do input.
  3. Aguarda a LRO concluir (poll não-interativo).
  4. Baixa todos os shards JSON de gs://<bucket>/<prefix>/output/ para --out-dir.

O Form Parser já faz OCR internamente; PDFs com texto nativo não precisam de etapa separada.
Autenticação por ADC (authorized_user com refresh_token) — ver GOOGLE_APPLICATION_CREDENTIALS.

Comentários em português; identificadores em inglês (padrão do projeto).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Identificadores do projeto GCP (ordem permanente do capitão).
PROJECT_ID = os.environ.get("DOCAI_PROJECT", "sebrae-arquitetura-ia-teste")
LOCATION = os.environ.get("DOCAI_LOCATION", "us")
PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR", "83cc594eb5e2ea45")  # form-parser-documentprocessor
BUCKET = os.environ.get("DOCAI_BUCKET", "ceiaidp-raw-docs-dev")
GCS_PREFIX = os.environ.get("DOCAI_PREFIX", "temp-eletrico-documentai")


def upload_pdfs(bucket: storage.Bucket, pdf_paths: List[Path], input_prefix: str) -> List[str]:
    """Sobe os PDFs para o input/ do bucket e devolve os URIs gs://."""
    uris = []
    for p in pdf_paths:
        blob_name = f"{input_prefix}/{p.name}"
        blob = bucket.blob(blob_name)
        # Evita re-upload se o tamanho bater (checkpoint barato).
        if blob.exists() and blob.size == p.stat().st_size:
            logger.info("Já no bucket (mesmo tamanho): %s", blob_name)
        else:
            logger.info("Subindo %s -> gs://%s/%s", p.name, bucket.name, blob_name)
            blob.upload_from_filename(str(p), content_type="application/pdf")
        uris.append(f"gs://{bucket.name}/{blob_name}")
    return uris


def run_batch(client: documentai.DocumentProcessorServiceClient,
              input_uris: List[str], output_uri: str) -> None:
    """Dispara o batch_process_documents (LRO) e aguarda a conclusão."""
    name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    gcs_documents = documentai.GcsDocuments(
        documents=[
            documentai.GcsDocument(gcs_uri=uri, mime_type="application/pdf")
            for uri in input_uris
        ]
    )
    input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
    output_config = documentai.DocumentOutputConfig(
        gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=output_uri
        )
    )

    request = documentai.BatchProcessRequest(
        name=name,
        input_documents=input_config,
        document_output_config=output_config,
    )

    logger.info("Disparando batch_process_documents (LRO) sobre %d documentos...", len(input_uris))
    operation = client.batch_process_documents(request)
    logger.info("LRO: %s — aguardando (poll não-interativo)...", operation.operation.name)

    # Poll não-interativo com timeout amplo (NR-28 tem centenas de páginas).
    start = time.time()
    while not operation.done():
        time.sleep(15)
        logger.info("  ... LRO em andamento (%.0fs)", time.time() - start)
    # Propaga erro, se houver.
    result = operation.result(timeout=60)
    meta = documentai.BatchProcessMetadata(operation.metadata)
    logger.info("LRO concluída em %.0fs. Estado: %s", time.time() - start, meta.state.name)
    for ind in meta.individual_process_statuses:
        logger.info("  doc=%s status=%s out=%s",
                    ind.input_gcs_source, ind.status.message or "OK",
                    ind.output_gcs_destination)


def download_shards(bucket: storage.Bucket, output_prefix: str, out_dir: Path) -> int:
    """Baixa todos os shards JSON do output/ para o diretório local."""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for blob in bucket.list_blobs(prefix=output_prefix):
        if not blob.name.endswith(".json"):
            continue
        # Mantém a subestrutura de pastas do output (uma por doc).
        rel = blob.name[len(output_prefix):].lstrip("/")
        dest = out_dir / rel.replace("/", "__")
        logger.info("Baixando shard %s -> %s", blob.name, dest.name)
        blob.download_to_filename(str(dest))
        count += 1
    logger.info("Baixados %d shards JSON em %s", count, out_dir)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process de PDFs no Document AI Form Parser.")
    parser.add_argument("--pdf-dir", required=True, help="Diretório com os PDFs a processar.")
    parser.add_argument("--out-dir", required=True, help="Diretório local para os shards JSON.")
    parser.add_argument("--run-tag", default="pilot", help="Sufixo do prefixo de output (isola execuções).")
    parser.add_argument("--nrs", default=None, help="Lista de NRs (ex.: nr-10,nr-15). Padrão: todos os PDFs do dir.")
    parser.add_argument("--skip-batch", action="store_true", help="Só baixa shards (LRO já rodou).")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    input_prefix = f"{GCS_PREFIX}/input"
    output_prefix = f"{GCS_PREFIX}/output/{args.run_tag}"
    output_uri = f"gs://{BUCKET}/{output_prefix}/"

    # Seleção de PDFs.
    if args.nrs:
        wanted = {n.strip().lower() for n in args.nrs.split(",")}
        pdf_paths = sorted(p for p in pdf_dir.glob("*.pdf") if p.stem.lower() in wanted)
    else:
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"Nenhum PDF encontrado em {pdf_dir} (nrs={args.nrs})")
    logger.info("PDFs selecionados (%d): %s", len(pdf_paths), [p.name for p in pdf_paths])

    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET)

    docai_client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    )

    if not args.skip_batch:
        input_uris = upload_pdfs(bucket, pdf_paths, input_prefix)
        run_batch(docai_client, input_uris, output_uri)

    n = download_shards(bucket, output_prefix, out_dir)
    if n == 0:
        raise SystemExit("Nenhum shard baixado — verifique a LRO/output.")


if __name__ == "__main__":
    main()
