import os
import sys

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from rag_pipeline import answer_query


def print_sources(docs):
    seen = set()
    for i, doc in enumerate(docs, 1):
        meta = getattr(doc, "metadata", {})
        key = (meta.get("source"), meta.get("page"))

        if key in seen:
            continue
        seen.add(key)

        source = meta.get("source", "Unknown")
        domain = meta.get("domain", "Unknown")
        page = meta.get("page", "?")

        print(f"{i}. {source} ({domain}, page {page})")


if __name__ == "__main__":
    queries = [
        "What is volatility clustering?",
        "Explain market microstructure noise",
        "Why do momentum strategies decay over time?"
    ]

    for q in queries:
        print("\n" + "=" * 90)
        print(f"QUESTION: {q}")
        print("=" * 90)

        answer, sources = answer_query(q)

        print("\nANSWER:")
        print(answer)

        print("\nSOURCES USED:")
        if sources:
            print_sources(sources)
        else:
            print("None")
