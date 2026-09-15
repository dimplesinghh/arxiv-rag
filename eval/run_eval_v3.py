"""
Eval harness for dense retrieval + cross-encoder reranking.

Pipeline:
    1. Retrieve top-20 candidates from qasper_chunks_v2 (dense)
    2. Rerank with cross-encoder
    3. Return top-10
    4. Compute Recall@1, Recall@5, Recall@10, MRR

Output: eval/results/reranked_results.json

Requires:
    - Qdrant running with qasper_chunks_v2 collection populated
    - Run eval/build_qasper_index_v2.py first if not already done
"""

import json
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from src.rerank import rerank

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "qasper_chunks_v2"
RETRIEVE_K = 20   # retrieve more candidates for reranker to work with
FINAL_K = 10      # return top-10 after reranking
RESULTS_PATH = "eval/results/reranked_results.json"

model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(host="localhost", port=6333)


def get_gold_evidence(answer_objects):
    """
    Extract gold evidence strings from QASPER answer objects.

    Args:
        answer_objects (list): List of answer dicts with 'evidence'.

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
    Check relevance via word overlap (>50% overlap = relevant).

    Args:
        chunk_text    (str):       Retrieved chunk text.
        evidence_list (list[str]): Gold evidence paragraphs.

    Returns:
        bool: True if chunk overlaps sufficiently with any evidence.
    """
    chunk_words = set(chunk_text.lower().split())
    for ev in evidence_list:
        ev_words = set(ev.lower().split())
        if not ev_words:
            continue
        overlap = len(chunk_words & ev_words) / len(ev_words)
        if overlap > 0.5:
            return True
    return False


def main():
    os.makedirs("eval/results", exist_ok=True)

    print("Loading QASPER validation set...")
    dataset = load_dataset("allenai/qasper", split="validation")

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

            # step 1: dense retrieval — fetch more candidates
            qvec = model.encode(question).tolist()
            hits = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=qvec,
                limit=RETRIEVE_K,
            )
            candidates = [
                {
                    "text": h.payload["text"],
                    "paper_id": h.payload["paper_id"],
                    "chunk_idx": h.payload["chunk_idx"],
                    "section": h.payload.get("section", ""),
                    "dense_score": h.score,
                }
                for h in hits
            ]

            # step 2: rerank candidates
            reranked = rerank(question, candidates, top_k=FINAL_K)
            retrieved_texts = [r["text"] for r in reranked]

            # step 3: measure relevance
            relevance = [
                1 if chunk_contains_evidence(t, evidence) else 0
                for t in retrieved_texts
            ]

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
            "chunking": "section-aware",
            "retrieval": f"dense top-{RETRIEVE_K} → cross-encoder rerank → top-{FINAL_K}",
            "retrieval_model": MODEL_NAME,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
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

    print("\n=== RERANKED EVAL RESULTS ===")
    print(f"Questions evaluated : {total_questions}")
    print(f"Recall@1            : {results['summary']['recall@1']:.4f}")
    print(f"Recall@5            : {results['summary']['recall@5']:.4f}")
    print(f"Recall@10           : {results['summary']['recall@10']:.4f}")
    print(f"MRR                 : {results['summary']['mrr']:.4f}")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()