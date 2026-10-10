from typing import List
import logfire

# The reranker (ms-marco-MiniLM, 512 tokens ≈ 2000 chars incl. the query) must see a
# WHOLE chunk — the old paragraph-only splitter produced 17k–60k char chunks, so only
# their first ~2000 chars were ever ranked.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200  # a fact that falls on a boundary appears whole in one of the two chunks

# Try the largest natural boundary first; fall back to smaller ones only for pieces
# that are still too big ("" = hard cut, for text with no spaces at all).
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_to_pieces(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Break text into pieces that are each <= chunk_size, using the coarsest separator possible."""
    if len(text) <= chunk_size:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    if sep not in text:
        return _split_to_pieces(text, chunk_size, rest)

    parts = text.split(sep)
    pieces = []
    for i, part in enumerate(parts):
        # keep the separator attached so sentences/lines are rebuilt exactly
        part = part + sep if i < len(parts) - 1 else part
        pieces.extend(_split_to_pieces(part, chunk_size, rest) if len(part) > chunk_size else [part])
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Last ~overlap chars of a chunk, starting at a word boundary."""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Recursive, size-bounded chunker: no chunk is longer than chunk_size, and consecutive
    chunks share ~overlap chars of context.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        # Pieces leave room for the overlap prefix, so prefix + pieces still fit chunk_size
        pieces = _split_to_pieces(text, chunk_size - overlap, SEPARATORS)

        chunks: List[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > chunk_size:
                chunks.append(current.strip())
                current = _tail(current, overlap)
            current += piece
        if current.strip():
            chunks.append(current.strip())

        valid_chunks = [c for c in chunks if c]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
