"""
eval/run_eval.py

Runs the retrieval eval harness against the QASPER benchmark.

For each question in QASPER:
  1. Retrieve top-10 chunks from 'qasper_chunks' collection
  2. Check retrieved chunks against gold evidence text
  3. Compute Recall@1, Recall@5, Recall@10, MRR

Output: eval/results/baseline_results.json

Requires:
  - Qdrant running with 'qasper_chunks' collection populated
  - Run eval/build_qasper_index.py first
"""

import json
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from eval.metrics import recall_at_k, mean_reciprocal_rank, reciprocal_rank

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "qasper_chunks"
TOP_K = 10
RESULTS_PATH = "eval/results/baseline_results.json"


def get_gold_evidence(answer_objects):
    """
    Extract gold evidence strings from a list of QASPER answer objects.

    Each answer object is a dict with an 'evidence' key containing
    a list of relevant paragraph strings.

    Args:
        answer_objects (list): List of answer dicts, each with 'evidence'.

    Returns:
        list[str]: All non-empty evidence strings.
    """
    evidence = []
    for answer in answer_objects:
        for e in answer.get("evidence", []):
            if e and e.strip():
                evidence.append(e.strip())
    return evidence


def chunk_contains_evidence(chunk_text, evidence_list):
    """
    Check if a chunk contains any gold evidence via substring match.

    Args:
        chunk_text    (str):       Text of the retrieved chunk.
        evidence_list (list[str]): Gold evidence paragraphs.

    Returns:
        bool: True if chunk contains at least one evidence string.
    """
    chunk_lower = chunk_text.lower()
    for ev in evidence_list:
        # use first 100 chars of evidence to match — avoids
        # exact whitespace mismatch issues
        snippet = ev[:100].lower().strip()
        if snippet and snippet in chunk_lower:
            return True
    return False


def main():
    os.makedirs("eval/results", exist_ok=True)

    print("Loading QASPER...")
    dataset = load_dataset("allenai/qasper", split="validation")

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(host="localhost", port=6333)

    recall_1_scores = []
    recall_5_scores = []
    recall_10_scores = []
    rr_scores = []
    query_results = []

    total_questions = 0
    skipped = 0

    for paper in dataset:
        paper_id = paper["id"]
        questions = paper.get("qas", {}).get("question", [])
        answers_list = paper.get("qas", {}).get("answers", [])

        for question, answers in zip(questions, answers_list):
            answer_objects = answers.get("answer", [])
            evidence = get_gold_evidence(answer_objects)
            if not evidence:
                skipped += 1
                continue

            total_questions += 1

            # embed query and retrieve
            qvec = model.encode(question).tolist()
            hits = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=qvec,
                limit=TOP_K,
            )

            retrieved_texts = [h.payload["text"] for h in hits]

            # build binary relevance list: 1 if chunk contains evidence
            relevance = [
                1 if chunk_contains_evidence(t, evidence) else 0
                for t in retrieved_texts
            ]

            # compute per-query metrics
            r1  = 1.0 if any(relevance[:1])  else 0.0
            r5  = 1.0 if any(relevance[:5])  else 0.0
            r10 = 1.0 if any(relevance[:10]) else 0.0

            rr = 0.0
            for rank, rel in enumerate(relevance, start=1):
                if rel == 1:
                    rr = 1.0 / rank
                    break

            recall_1_scores.append(r1)
            recall_5_scores.append(r5)
            recall_10_scores.append(r10)
            rr_scores.append(rr)

            query_results.append({
                "paper_id": paper_id,
                "question": question,
                "recall@1": r1,
                "recall@5": r5,
                "recall@10": r10,
                "rr": rr,
            })

    results = {
        "config": {
            "collection": COLLECTION_NAME,
            "model": MODEL_NAME,
            "top_k": TOP_K,
            "chunking": "fixed-size 512/50",
            "retrieval": "naive cosine top-k, no reranking",
        },
        "summary": {
            "total_questions": total_questions,
            "skipped_no_evidence": skipped,
            "recall@1":  sum(recall_1_scores)  / len(recall_1_scores),
            "recall@5":  sum(recall_5_scores)  / len(recall_5_scores),
            "recall@10": sum(recall_10_scores) / len(recall_10_scores),
            "mrr":       sum(rr_scores)        / len(rr_scores),
        },
        "per_query": query_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== BASELINE EVAL RESULTS ===")
    print(f"Questions evaluated : {total_questions}")
    print(f"Recall@1            : {results['summary']['recall@1']:.4f}")
    print(f"Recall@5            : {results['summary']['recall@5']:.4f}")
    print(f"Recall@10           : {results['summary']['recall@10']:.4f}")
    print(f"MRR                 : {results['summary']['mrr']:.4f}")
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()