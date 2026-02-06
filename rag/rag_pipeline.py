import os
import sys
import warnings
from typing import List, Tuple

# Silence noisy but harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Make backend importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from backend.vector_store import load_faiss_index
from transformers import pipeline


# ---------------- CONFIG ----------------
INDEX_DIR = os.path.join(PROJECT_ROOT, "vector_store", "faiss_index")
TOP_K = 3
MAX_CONTEXT_CHARS = 1200   # enough for theory papers
MAX_CHUNK_CHARS = 400      # handled upstream but documented here

GEN_MODEL = "google/flan-t5-base"
# ---------------------------------------


def _build_context(docs) -> str:
    """Safely build context string from retrieved documents."""
    context_parts = []
    total_chars = 0

    for d in docs:
        text = d.page_content.strip()
        if not text:
            continue

        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(text)
        total_chars += len(text)

    return "\n\n".join(context_parts)


def answer_query(question: str) -> Tuple[str, List]:
    """
    Run retrieval + generation.
    Returns:
        answer (str)
        source_docs (List[Document])
    """

    # 1. Load vector store
    vectorstore = load_faiss_index(INDEX_DIR)

    # 2. Retrieve documents
    docs = vectorstore.similarity_search(question, k=TOP_K)

    if not docs:
        return "Insufficient evidence in the provided documents.", []

    # 3. Build context
    context = _build_context(docs)

    if not context.strip():
        return "Insufficient evidence in the provided documents.", docs

    # 4. Prompt (relaxed but grounded)
    prompt = f"""
You are answering using ONLY the provided academic context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Explain how the concept is described, defined, or analyzed in the papers
- Summarize the mechanism or empirical pattern discussed
- Do NOT add external knowledge
- If the papers do not meaningfully discuss the concept, answer exactly:
  "Insufficient evidence in the provided documents."

ANSWER:
""".strip()

    # 5. Load model lazily
    generator = pipeline(
        "text2text-generation",
        model=GEN_MODEL,
        max_new_tokens=120,
        do_sample=False
    )

    # 6. Generate
    result = generator(prompt)[0]["generated_text"].strip()

    # 7. Final guardrail (semantic, not word-count)
    if result.lower().startswith("insufficient"):
        return "Insufficient evidence in the provided documents.", docs

    return result, docs
