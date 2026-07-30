"""
eval/run_eval_v2.py

Eval harness for hybrid retrieval on section-aware QASPER index.

Compares against baseline (run_eval.py) results.
Uses hybrid dense + BM25 retrieval on qasper_chunks_v2 collection.

Output: eval/results/hybrid_results.json
"""

import json
import os
from datasets import load_dataset
from src.hybrid_retrieve import load_all_chunks, build_bm25_index, hybrid_retrieve

RESULTS_PATH = "eval/results/hybrid_results.json"
TOP_K = 10


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

    print("Loading chunks and building BM25 index...")
    chunks = load_all_chunks()
    bm25 = build_bm25_index(chunks)
    print(f"BM25 index built over {len(chunks)} chunks.")

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

            # hybrid retrieval
            hits = hybrid_retrieve(question, chunks, bm25, k=TOP_K)
            retrieved_texts = [h["text"] for h in hits]

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
            "collection": "qasper_chunks_v2",
            "chunking": "section-aware",
            "retrieval": "hybrid dense + BM25 (alpha=0.6)",
            "model": "all-MiniLM-L6-v2",
            "top_k": TOP_K,
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

    print("\n=== HYBRID EVAL RESULTS ===")
    print(f"Questions evaluated : {total_questions}")
    print(f"Recall@1            : {results['summary']['recall@1']:.4f}")
    print(f"Recall@5            : {results['summary']['recall@5']:.4f}")
    print(f"Recall@10           : {results['summary']['recall@10']:.4f}")
    print(f"MRR                 : {results['summary']['mrr']:.4f}")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()