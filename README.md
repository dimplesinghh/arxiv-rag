# arxiv-rag

RAG pipeline for arXiv papers: ingest → parse → chunk → embed → retrieve.

## Status
Work in progress.



python -m src.embed
Verify: go to http://localhost:6333/dashboard, click into arxiv_chunks collection, confirm point count matches chunk count.