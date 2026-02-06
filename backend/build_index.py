import os

from ingest import load_pdfs
from preprocess import preprocess_documents
from chunk import chunk_documents
from vector_store import build_faiss_index


def main():
    print(">>> build_index.py started")

    # Resolve project paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "Data")
    index_dir = os.path.join(project_root, "vector_store", "faiss_index")

    print(f"Project root: {project_root}")
    print(f"Data dir: {data_dir}")
    print(f"Index dir: {index_dir}")

    # 1. Ingest
    print("\n[1/4] Loading PDFs...")
    raw_docs = load_pdfs(data_dir)

    # 2. Preprocess
    print("\n[2/4] Preprocessing documents...")
    cleaned_docs = preprocess_documents(raw_docs)

    # 3. Chunk
    print("\n[3/4] Chunking documents...")
    chunks = chunk_documents(cleaned_docs)

    # 4. Vector index
    print("\n[4/4] Building FAISS index...")
    build_faiss_index(chunks, index_dir)

    print("\n FAISS index built successfully")


if __name__ == "__main__":
    main()
