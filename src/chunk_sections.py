"""
src/chunk_sections.py

Section-aware chunking for QASPER papers.

Instead of fixed-size token splitting, this module:
  1. Reads paper sections and paragraphs from QASPER structure
  2. Creates chunks that respect section boundaries
  3. Tags each chunk with its section name
  4. Merges short paragraphs, splits oversized ones

This produces semantically coherent chunks — a chunk about
methodology stays about methodology, not half methodology
and half results.

Output schema per chunk:
    {
        "paper_id":    str,
        "chunk_idx":   int,
        "section":     str,   # section heading this chunk came from
        "text":        str,
    }
"""

import tiktoken

CHUNK_SIZE = 512
MIN_CHUNK_TOKENS = 64   # paragraphs shorter than this get merged
enc = tiktoken.get_encoding("cl100k_base")

def token_count(text):
    """Return number of tokens in text."""
    return len(enc.encode(text))

def split_long_text(text, max_tokens=CHUNK_SIZE):
    """
    Split text that exceeds max_tokens into overlapping chunks.

    Used when a single paragraph is too long to embed as one unit.

    Args:
        text      (str): Text to split.
        max_tokens (int): Token limit per chunk.

    Returns:
        list[str]: List of sub-chunks with 50-token overlap.
    """
    tokens = enc.encode(text)
    chunks = []
    overlap = 50
    i = 0
    while i < len(tokens):
        chunks.append(enc.decode(tokens[i:i + max_tokens]))
        i += max_tokens - overlap
    return chunks

def chunk_paper_by_sections(paper_id, full_text_field):
     
    """
    Chunk a QASPER paper into section-aware chunks.

    Processes sections in order. Each paragraph becomes a candidate
    chunk. Short paragraphs are accumulated and merged. Oversized
    paragraphs are split with overlap.

    Args:
        paper_id        (str):  Arxiv paper ID.
        full_text_field (dict): QASPER full_text dict with keys
                                'section_name' and 'paragraphs'.

    Returns:
        list[dict]: Chunks with paper_id, chunk_idx, section, text.
    """

    section_name = full_text_field.get("section_name", [])
    paragraphs_list = full_text_field.get("paragraphs", [])

    chunks = []
    chunk_idx = 0
    buffer = ""
    buffer_section = ""

    def flush_buffer(buf, section):
        """Save buffer as a chunk if non-empty."""
        nonlocal chunk_idx
        buf = buf.strip()
        if not buf:
            return
        if token_count(buf) > CHUNK_SIZE:
            for sub in split_long_text(buf):
                chunks.append({
                    "paper_id": paper_id,
                    "chunk_idx": chunk_idx,
                    "section": section,
                    "text": sub,
                })
                chunk_idx += 1
        else:
            chunks.append({
                "paper_id": paper_id,
                "chunk_idx": chunk_idx,
                "section": section,
                "text": buf,
            })
            chunk_idx += 1
    
    for section, paragraphs in zip(section_name, paragraphs_list):
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        if not isinstance(paragraphs, list):
            continue

        
        # New section: flush existing buffer first
        if buffer and buffer_section != section:
            flush_buffer(buffer, buffer_section)
            buffer = ""

        buffer_section = section or "unknown"

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_text = f"[{buffer_section}] {para}" if buffer_section else para

            if token_count(buffer + " " + para_text) <= CHUNK_SIZE:
                buffer = (buffer + " " + para_text).strip()
            else:
                flush_buffer(buffer, buffer_section)
                buffer = para_text

    # flush any remaining buffer
    flush_buffer(buffer, buffer_section)

    return chunks

