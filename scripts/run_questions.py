from __future__ import annotations
import os
import json
import argparse

from src.config import RagConfig
from src.vector_store import VectorIndex
from src.rag import RagSystem

QUESTIONS = [
    (1, "What is the total net sales/revenue for Apple Inc. for FY2024?"),
    (2, "How many shares of common stock of Apple Inc. were outstanding as of October 18, 2024?"),
    (3, "What is the total term debt (current and non-current) for Apple Inc. as of September 28, 2024?"),
    (4, "On what date did Apple file its 2024 10-K?"),
    (5, "Does Apple have any unresolved staff comments from the SEC?"),
    (6, "What is the total revenue for Tesla for FY2023?"),
    (7, "What percentage of Tesla's total revenue in FY2023 came from automotive sales (excluding leasing)?"),
    (8, "Why is Tesla dependent on Elon Musk?"),
    (9, "Which vehicles does Tesla currently produce and deliver?"),
    (10, "What is the purpose of Tesla's lease pass-through fund arrangements?"),
    (11, "What is the forecasted stock price for Tesla in 2025?"),
    (12, "Who is Apple's CFO as of 2025?"),
    (13, "What color is Tesla's headquarters?"),
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
    for qid, q in QUESTIONS:
        r = rag.answer_question(q)
        results.append({"question_id": qid, "answer": r["answer"], "sources": r["sources"]})
        print(f"Q{qid}: {r['answer']}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.out_json}")

if __name__ == "__main__":
    main()
