# arxiv-rag

RAG pipeline for arXiv papers: ingest → parse → chunk → embed → retrieve.

## Pipeline
1. Ingest ~50 papers from cs.LG/cs.AI via arxiv API (`src/ingest.py`)
2. Parse PDFs to text with pymupdf (`src/parse.py`)
3. Fixed-size chunking: 512 tokens, 50 overlap (`src/chunk.py`)
4. Embed with sentence-transformers all-MiniLM-L6-v2, store in Qdrant (`src/embed.py`)
5. Naive cosine top-k retrieval, no reranking (`src/retrieve.py`)


## Run order
\`\`\`bash

docker compose up -d

python -m src.ingest

python -m src.parse

python -m src.chunk

python -m src.embed

    Verify: go to http://localhost:6333/dashboard, click into arxiv_chunks collection, confirm point count matches chunk count.

python -m src.retrieve

\`\`\`

## Known limitations (intentional, to be fixed later)
- No reranking
- Fixed-size chunking ignores semantic/section boundaries
- No quantitative eval yet — relevance judged by eyeballing only
- No query rewriting/expansion
