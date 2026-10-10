# `app/ingestion/` — the offline ingestion job

This turns source files into searchable vectors. It runs by hand from a terminal, never inside the API or the Docker image.

```
file ─► loader (text) ─► chunker (≤1200-char chunks) ─► Gemini embeddings ─► Qdrant
                              │
                              └─► processed_data/<source_type>/<file>.json  (local copy of the chunks)
```

## Files

| File | Role |
|---|---|
| `processor.py` | The job: routing, skip logic, embedding, upsert, verification, report |
| `chunking/splitter.py` | `chunk_text()` — recursive, size-bounded chunker with overlap |
| `loaders/pdf.py` | `pypdf`, with `pdfplumber` as a fallback for pages that come back blank |
| `loaders/html.py` | BeautifulSoup: drops `script`, `style`, `meta`, `noscript`, collapses whitespace |
| `loaders/office.py` | `unstructured` for `.docx` and `.pptx` |
| `loaders/text.py` | Plain read for `.txt` |

## Running it

```bash
python -m app.ingestion.processor DATA/true_data --dry-run   # parse + chunk only, print the report
python -m app.ingestion.processor DATA/true_data             # add to the collection
python -m app.ingestion.processor DATA/true_data --wipe      # drop the collection first, then ingest
```

- The target collection is `QDRANT_COLLECTION` from `.env` (default `enterprise_rag`).
- **`--wipe` deletes the whole collection.** If that collection is the one the live demo reads, the demo has no data until ingestion finishes.
- Embedding uses the same `GEMINI_API_KEY` as the live demo's query embeddings. A large ingest can use up the free-tier quota and starve the demo.
- The process exits with code 1 if any file ended as `FAILED` or `MISMATCH`.

## `processor.py` step by step

For each file, `process_file()` does this and returns one report row:

1. **Parse** by extension (`pdf`, `html`/`htm`, `txt`, `docx`/`pptx`). Other types are reported as `skipped`.
2. **Chunk** with `chunk_text()`.
3. **Skip if already complete.** If Qdrant already holds exactly as many points for this file as chunks were produced, embedding is skipped.
4. **Save** the chunks to `processed_data/<source_type>/<file>.json`.
5. **Embed** with `embed_texts()` (see [`../services/retrieval/README.md`](../services/retrieval/README.md)).
6. **Upsert** in batches of 16 with `wait=True`.

Around that:

| Function | What it does |
|---|---|
| `_point_id()` | Deterministic UUID from `source_type/filename#index`. Re-ingesting a file overwrites its points instead of duplicating them. |
| `_check_dimension()` | Stops before anything is changed if the active embedding model's dimension differs from the collection's. |
| `_ensure_collection()` | Creates the collection (cosine distance) and keyword indexes on `source` and `source_type`. |
| `verify_in_qdrant()` | After ingestion, counts the points Qdrant holds per file. A difference from the chunk count becomes status `MISMATCH`. |
| `print_report()` | One line per file: status, characters, chunks, largest chunk, points in Qdrant. |

**Source types.** A sub-folder whose name contains `true` becomes source type `true`, one containing `noisy` becomes `noisy`; any other sub-folder name is used as is. Each point's payload is `{text, source, source_type, chunk_index}`.

## `chunking/splitter.py`

| Constant | Value | Reason |
|---|---|---|
| `CHUNK_SIZE` | 1200 characters | The reranker reads about 512 tokens (≈ 2,000 characters including the query). A chunk must fit so it is ranked as a whole. |
| `CHUNK_OVERLAP` | 200 characters | A fact that falls on a boundary appears whole in one of the two neighbouring chunks. |
| `SEPARATORS` | `"\n\n"`, `"\n"`, `". "`, `" "`, `""` | Split on the largest natural boundary first; use smaller ones only for pieces that are still too big. `""` is a hard cut for text with no spaces. |

How it works:

1. `_split_to_pieces()` breaks the text into pieces of at most `chunk_size - overlap`, recursing to a finer separator only where needed. Separators stay attached, so the text is rebuilt exactly.
2. `chunk_text()` packs pieces into a chunk until the next piece would exceed `chunk_size`. The next chunk starts with `_tail()` — the last ~200 characters of the previous one, cut at a word boundary.

No chunk is longer than `chunk_size`.

## What changed while building this, and why

| Change | Why |
|---|---|
| Paragraph-only splitter → recursive, size-bounded splitter with overlap | The old splitter only split on blank lines. PDFs and HTML without blank lines became single chunks of 17k–60k characters. The reranker saw only their first ~2,000 characters, so answers further in were never ranked. Example: VPA questions returned "not in the docs" because the VPA section sat deep inside one 17.7k-character chunk. |
| Failures are returned in a report, not only logged | The first eval run found that `cronjobs.docx` had never been indexed. Ingestion had logged the error and moved on. Now every file has a status and the job exits non-zero on any failure. |
| Post-ingest verification (`verify_in_qdrant`) | "No exception" is not proof that points arrived. The count in Qdrant is compared with the chunks produced. |
| Deterministic point ids | Re-running after a partial failure overwrites instead of creating duplicate chunks. |
| Skip files already fully in Qdrant | A re-run after a rate-limit failure re-embeds only what is missing, which saves Gemini quota. |
| Dimension check before `--wipe` | If Gemini is unreachable, the 768-dim local fallback loads. Without the check, `--wipe` would delete a 3072-dim collection and rebuild it with vectors the app cannot query. |
| Qdrant client timeout 60 s, upsert batch 64 → 16 | A batch of 64 points with 3072 floats each is about 4 MB and timed out on a home connection. 16 is about 1 MB. |
| Payload indexes on `source` and `source_type` | Make the per-file count fast, and some clusters reject filters on unindexed fields. |
| `--dry-run` flag | Check chunk counts and sizes before spending any embedding quota. |
| Ingest into `enterprise_rag_v2`, not `--wipe` on the live collection | Zero downtime: the demo kept serving the old collection while the new one was built and evaluated. |

**Measured result of the re-ingest** (same corpus, same goldens, same judge; baseline run `20260927-112048` → v2 run `20261003-211038`):

| Metric | Baseline | v2 |
|---|---|---|
| Source hit rate | 0.706 | 0.941 |
| Context recall | 0.691 | 0.941 |
| Context precision | 0.736 | 0.828 |
| Faithfulness | 0.780 | 0.924 |
| Answer relevancy | 0.734 | 0.789 |

One golden got worse: `pwq-002`, a two-part question whose answers now sit in two different small chunks. This was fixed afterwards in the agent, not in ingestion: the planner splits multi-part questions and the retriever searches once per part (run `20261009-143328`: source hit rate 17/17). See the `sub_queries` notes in [`../agents/README.md`](../agents/README.md).
