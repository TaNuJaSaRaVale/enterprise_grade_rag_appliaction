import os
import sys
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=60,  # the default few seconds is too short to upload ~1 MB batches on a home connection
)


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


UPSERT_BATCH_SIZE = 16  # ~1 MB of JSON per request (3072 floats per point); 64 (~4 MB) timed out


def _point_id(source_type: str, filename: str, index: int) -> str:
    """Same file + chunk number → same id, so re-ingesting a file overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_type}/{filename}#{index}"))


def _stored_count(filename: str, source_type: str) -> int:
    """How many points Qdrant currently holds for this file."""
    return qdrant_client.count(
        collection_name=settings.QDRANT_COLLECTION,
        count_filter=models.Filter(must=[
            models.FieldCondition(key="source", match=models.MatchValue(value=filename)),
            models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type)),
        ]),
        exact=True,
    ).count


def process_file(file_path: str, filename: str, source_type: str, dry_run: bool = False) -> dict:
    """
    Parse → chunk → save locally → embed → index in Qdrant.
    Returns a result row for the ingestion report — failures are REPORTED, not just logged
    (the old version logged and moved on, which silently dropped cronjobs.docx).
    """
    result = {"file": filename, "source_type": source_type, "status": "indexed",
              "chars": 0, "chunks": 0, "max_chunk": 0, "detail": ""}
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return {**result, "status": "skipped", "detail": f"unsupported type .{ext}"}

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return {**result, "status": "skipped", "detail": "no text extracted"}

            # 2. Chunk text
            chunks = chunk_text(full_text)
            result.update(chars=len(full_text), chunks=len(chunks), max_chunk=max(map(len, chunks), default=0))
            if not chunks:
                return {**result, "status": "skipped", "detail": "no chunks"}
            if dry_run:
                return {**result, "status": "dry-run"}

            # Re-runs after a partial failure only re-embed what is missing (saves Gemini quota).
            # An exact count match is required — a changed chunker or a half-uploaded file differs.
            if _stored_count(filename, source_type) == len(chunks):
                logfire.info(f"{filename} already fully in Qdrant — skipping embedding.")
                return {**result, "detail": "already in Qdrant, not re-embedded"}

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id=_point_id(source_type, filename, i),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                            "chunk_index": i,
                        },
                    )
                    for i, (chunk, vector) in enumerate(zip(chunks, embeddings))
                ]

                for start in range(0, len(points), UPSERT_BATCH_SIZE):
                    qdrant_client.upsert(
                        collection_name=settings.QDRANT_COLLECTION,
                        points=points[start:start + UPSERT_BATCH_SIZE],
                        wait=True,  # the verification count below must see these points
                    )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")
            return result

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            return {**result, "status": "FAILED", "detail": f"{type(e).__name__}: {str(e)[:200]}"}


def process_directory(dir_path: str, source_type: str, dry_run: bool = False) -> list[dict]:
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = sorted(f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)))
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        results = []
        for n, filename in enumerate(files, start=1):
            print(f"[{n}/{len(files)}] {filename}", flush=True)
            results.append(process_file(os.path.join(dir_path, filename), filename, source_type, dry_run))
        return results


def verify_in_qdrant(results: list[dict]) -> None:
    """Post-ingest check: the points Qdrant actually holds per file must equal the chunks produced."""
    for r in results:
        if r["status"] != "indexed":
            continue
        stored = _stored_count(r["file"], r["source_type"])
        r["in_qdrant"] = stored
        if stored != r["chunks"]:
            r["status"] = "MISMATCH"
            r["detail"] = f"{r['chunks']} chunks produced but {stored} points in Qdrant"


def print_report(results: list[dict]) -> None:
    print("\n=== Ingestion report ===")
    print(f"{'status':9} {'chars':>9} {'chunks':>6} {'max':>5} {'qdrant':>6}  file")
    for r in results:
        print(f"{r['status']:9} {r['chars']:9} {r['chunks']:6} {r['max_chunk']:5} {str(r.get('in_qdrant', '-')):>6}  "
              f"{r['source_type']}/{r['file']}" + (f"  — {r['detail']}" if r["detail"] else ""))
    bad = [r for r in results if r["status"] in ("FAILED", "MISMATCH")]
    total = sum(r["chunks"] for r in results if r["status"] in ("indexed", "dry-run"))
    print(f"\nFiles: {len(results)} · chunks: {total} · problems: {len(bad)}")


def _check_dimension() -> None:
    """Refuse to mix embedding models in one collection."""
    dim = get_embedding_dim()
    if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
        existing = qdrant_client.get_collection(settings.QDRANT_COLLECTION).config.params.vectors.size
        if existing != dim:
            # Happens when Gemini is unreachable and the 768-dim local fallback loads:
            # its vectors cannot be compared with the collection's Gemini vectors.
            sys.exit(f"Embedding dimension {dim} does not match collection '{settings.QDRANT_COLLECTION}' "
                     f"({existing}). Is Gemini reachable? Aborting before anything is changed.")


def _ensure_collection() -> None:
    """Create the collection if missing (dimension from the active embedding model)."""
    if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
        dim = get_embedding_dim()
        qdrant_client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        logfire.info(f"Created collection '{settings.QDRANT_COLLECTION}' ({dim}-dim, Cosine).")
    # Keyword indexes make the per-file verification count fast (and allowed on clusters
    # that reject filters on unindexed fields). Creating an existing index is a no-op.
    for field in ("source", "source_type"):
        qdrant_client.create_payload_index(settings.QDRANT_COLLECTION, field_name=field,
                                           field_schema=models.PayloadSchemaType.KEYWORD)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False,
                            dry_run: bool = False) -> list[dict]:
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion,
    --dry-run to only parse + chunk and print the report (no embeddings, no Qdrant).
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        if not dry_run:
            # Check the embedding model BEFORE wiping — otherwise a Gemini outage would
            # delete the Gemini collection and recreate it with the 768-dim fallback
            _check_dimension()
            if wipe:
                with logfire.span("Wiping Collection"):
                    if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                        qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                        logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")
            _ensure_collection()

        # Route to sub-folders or treat the whole dir as one source
        results = []
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            results += process_directory(base_dir, source_type, dry_run)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                results += process_directory(os.path.join(base_dir, subdir), source_type, dry_run)

        if not dry_run:
            verify_in_qdrant(results)
        return results


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA/true_data --dry-run   (parse + chunk only, print report)
    #   python -m app.ingestion.processor DATA/true_data --wipe      (drop collection, then ingest)
    #   python -m app.ingestion.processor DATA/noisy_sample_10       (add to the existing collection)
    flags = {"--wipe", "--dry-run"}
    wipe_requested = "--wipe" in sys.argv
    dry_run = "--dry-run" in sys.argv
    clean_args = [a for a in sys.argv if a not in flags]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    results = run_universal_ingestion(target_dir, explicit_source_type=explicit_type,
                                      wipe=wipe_requested, dry_run=dry_run)
    print_report(results)
    logfire.info("Ingestion job completed.")
    # Non-zero exit so scripts/CI notice a partial ingestion instead of trusting it
    if any(r["status"] in ("FAILED", "MISMATCH") for r in results):
        sys.exit(1)