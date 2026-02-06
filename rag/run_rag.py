import warnings

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
)

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from rag_pipeline import answer_query

QUERIES = [
    "What is volatility clustering?",
    "Explain market microstructure noise",
    "Why do momentum strategies decay over time?",
]

if __name__ == "__main__":
    for q in QUERIES:
        print("\n" + "=" * 90)
        print(f"QUESTION: {q}")
        print("=" * 90)

        answer, sources = answer_query(q)

        print("\nANSWER:")
        print(answer)

        print("\nSOURCES USED:")
        if not sources:
            print("None")
        else:
            for i, src in enumerate(sources, 1):
                print(f"{i}. {src}")
