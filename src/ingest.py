import arxiv 
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder holding this script (src/)
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # one level up (project root)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
METADATA_PATH = os.path.join(RAW_DIR, "metadata.json")

MAX_RESULTS = 170

def main():
    os.makedirs(RAW_DIR, exist_ok = True)
    search = arxiv.Search(
        query="cat:cs.LG OR cat:cs.AI",
        max_results = MAX_RESULTS,
        sort_by = arxiv.SortCriterion.SubmittedDate,
    )

    client = arxiv.Client()
    metadata = []

    for i, result in enumerate(client.results(search)):
        paper_id = result.get_short_id()
        filename = f"{paper_id.replace('/', '_')}.pdf"

        # print(i, paper_id, result.title)
        # print(dir(result))

        print(f"[{i+1}/{MAX_RESULTS}] downloading {paper_id}")
        result.download_pdf(dirpath=RAW_DIR, filename=filename)

        metadata.append({
            "id": paper_id,
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "categories": result.categories,
            "published": str(result.published),
            "pdf_filename": filename
        })

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. {len(metadata)} papers saved to {RAW_DIR}")

if __name__ == "__main__":
    main()