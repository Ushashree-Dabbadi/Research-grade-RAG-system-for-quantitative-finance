import warnings

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
)


import os
from transformers import pipeline
from backend.vector_store import load_faiss_index

# -----------------------------
# HARD LIMITS (TOKEN-SAFE)
# -----------------------------
TOP_K = 3
MAX_CONTEXT_CHARS = 800      # total context
MAX_CHUNK_CHARS = 220        # per chunk (critical)

INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "vector_store",
    "faiss_index",
)

# -----------------------------
# Load model ONCE
# -----------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=160,
    do_sample=False,
)

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query: str):
    vectorstore = load_faiss_index(INDEX_DIR)
    return vectorstore.similarity_search(query, k=TOP_K)

# -----------------------------
# Prompt construction
# -----------------------------
def build_prompt(question: str, docs):
    context_blocks = []
    sources = []

    total_chars = 0

    for doc in docs[:TOP_K]:
        text = doc.page_content.strip()
        meta = doc.metadata

        if not text:
            continue

        snippet = text[:MAX_CHUNK_CHARS]

        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
            break

        context_blocks.append(snippet)
        total_chars += len(snippet)

        sources.append(
            f"{meta.get('source')} "
            f"({meta.get('domain')}, page {meta.get('page')})"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a quantitative finance research assistant.

CONTEXT (verbatim excerpts from academic papers):
{context}

QUESTION:
{question}

ANSWER INSTRUCTIONS:
- Use ONLY the context above
- Describe how the concept is operationalized or empirically used in the papers
- Do NOT give generic or textbook definitions
- Do NOT speculate or generalize
- If the context does not explicitly explain the concept, respond EXACTLY with:
  "Insufficient evidence in the provided documents."

ANSWER:
"""

    return prompt, sources

# -----------------------------
# Public API
# -----------------------------
def answer_query(question: str):
    docs = retrieve(question)
    prompt, sources = build_prompt(question, docs)

    output = generator(prompt)[0]["generated_text"].strip()

    # Enforce hard refusal if model tries to be vague
    if len(output.split()) < 6:
        output = "Insufficient evidence in the provided documents."

    return output, sources
