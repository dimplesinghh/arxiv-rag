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

## Eval Harness (Weekend 2)

**Benchmark**: QASPER (allenai/qasper) — human-annotated QA pairs over 
NLP/ML papers with gold evidence paragraphs.

**Metrics**:
- Recall@k: fraction of queries where at least one relevant chunk 
  appears in top-k results
- MRR: Mean Reciprocal Rank — rewards finding relevant chunks higher 
  in the ranking

**Baseline results (naive top-k, no reranking):**

| Config                  | Recall@1 | Recall@5 | Recall@10 | MRR   |
|-------------------------|----------|----------|-----------|-------|
| Naive cosine, 512/50    | X.XX     | X.XX     | X.XX      | X.XX  |

*Fill in X.XX with your actual numbers from running eval/run_eval.py*

**Run eval:**
```bash
python -m eval.build_qasper_index   # build index once
python -m eval.run_eval             # compute metrics
```

## Known limitations (intentional, to be fixed later)
- No reranking
- Fixed-size chunking ignores semantic/section boundaries
- No quantitative eval yet — relevance judged by eyeballing only
- No query rewriting/expansion

## Weekend- 3: Hybrid Search + Section-Aware Chunking

### Changes implemented
- **Section-aware chunking** (`src/chunk_sections.py`): chunks respect 
  paragraph and section boundaries from QASPER structure, tagged with 
  section name. Replaces fixed-size 512/50 token splitting.
- **Hybrid retrieval** (`src/hybrid_retrieve.py`): combines dense cosine 
  similarity (Qdrant) with BM25 keyword search via weighted fusion 
  (alpha=0.6 dense, 0.4 BM25).

### Critical bug fixed
Both Weekend 2 baseline and Weekend 3 indexes were built from the `train` 
split while eval ran on `validation` split — completely different papers. 
Fixed by rebuilding both indexes from the validation split. This is why 
original baseline numbers (Recall@10=0.1176) were artificially low.

### Results after bug fix

| Config | Recall@1 | Recall@5 | Recall@10 | MRR |
|--------|----------|----------|-----------|-----|
| Naive dense, fixed 512/50 | 0.1359 | 0.2697 | 0.3517 | 0.1945 |
| Hybrid dense+BM25, section-aware | 0.1090 | 0.2600 | 0.3247 | 0.1741 |

### Finding
Dense-only retrieval outperformed hybrid on this benchmark. BM25 added 
noise rather than signal — likely because section headers prepended to 
chunks during section-aware chunking (`[Section ::: Subsection]` prefix) 
polluted BM25 keyword matching with non-query-relevant tokens. 
