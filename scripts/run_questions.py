from __future__ import annotations
import os
import json
import argparse

from src.config import RagConfig
from src.vector_store import VectorIndex
from src.rag import RagSystem

QUESTIONS = [
    (1, "What was Apples total revenue for the fiscal year ended September 28, 2024?"),
    (2, "How many shares of common stock were issued and outstanding as of October 18, 2024?"),
    (3, "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"),
    (4, "On what date was Apples 10-K report for 2024 signed and filed with the SEC?"),
    (5, "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"),
    (6, "What was Teslas total revenue for the year ended December 31, 2023?"),
    (7, "What percentage of Teslas total revenue in 2023 came from Automotive Sales (excluding Leasing)?"),
    (8, "What is the primary reason Tesla states for being highly dependent on Elon Musk?"),
    (9, "What types of vehicles does Tesla currently produce and deliver?"),
    (10, "What is the purpose of Teslas 'lease pass-through fund arrangements'?"),
    (11, "What is Teslas stock price forecast for 2025?"),
    (12, "Who is the CFO of Apple as of 2025?"),
    (13, "What color is Teslas headquarters painted?")
]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="artifacts/index", help="Directory containing faiss.index and store.jsonl")
    ap.add_argument("--out_json", default="outputs/answers.json", help="Output JSON path")
    ap.add_argument("--single_question", default="", help="Ask a single custom question instead of the 13")
    args = ap.parse_args()

    cfg = RagConfig()
    index = VectorIndex.load(args.index_dir)
    rag = RagSystem(cfg, index)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    if args.single_question:
        result = rag.answer_question(args.single_question)
        print(json.dumps(result, indent=2))
        return

    results = []
    for item in QUESTIONS:
        # Backwards compatible:
        #   - (qid, "question") tuples (default)
        #   - {"question_id": qid, "question": "..."} dicts
        if isinstance(item, dict):
            qid = item.get("question_id")
            q = item.get("question") or item.get("query") or item.get("q") or ""
        else:
            qid, q = item

        r = rag.answer_question(item if isinstance(item, dict) else q)
        results.append({
            "question_id": qid,
            "answer": r.get("answer", ""),
            "sources": r.get("sources", []),
        })
        print(f"Q{qid}: {r.get('answer','')}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.out_json}")

if __name__ == "__main__":
    main()
