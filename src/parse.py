import fitz
import json
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
METADATA_PATH = "data/raw/metadata.json"

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    for page in doc:
        # sort blocks top-to-bottom, left-to-right — fixes 2-column ordering
        blocks = page.get_text("blocks")
        for b in blocks:
            # using only 4 here because columns 0–3 are just the coordinates of the bounding box — they tell you where the text is on the page, not what the text says.
            text = b[4].strip() 
            if text:
                full_text.append(text)

    doc.close()
    return "\n".join(full_text)

def main():
    os.makedirs(PROCESSED_DIR, exist_ok = True)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    for i, paper in enumerate(metadata):
        pdf_path = os.path.join(RAW_DIR, paper["pdf_filename"])
        out_path = os.path.join(PROCESSED_DIR, paper["id"].replace("/", "_") + ".txt")

        if not os.path.exists(pdf_path):
            print(f"missing PDF: {pdf_path}, skipping")
            continue

        print(f"[{i+1}/{len(metadata)}] parsing {paper['id']}")

        try:
            text = extract_text(pdf_path)
            with open(out_path, "w") as f:
                f.write(text)

        except Exception as e:
            print(f"failed on {paper['id']}: {e}")

    print("Done parsing.")

if __name__ == "__main__":
    main()

