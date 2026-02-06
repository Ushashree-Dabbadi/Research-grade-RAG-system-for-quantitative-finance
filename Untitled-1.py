from pathlib import Path

# Base directory = current working directory (rag_quant_finance)
BASE = Path(".").resolve()

FOLDERS = [
    "backend",
    "Data",
    "Data/Statistical-Finance",
    "Data/Trading-Market-Microstructure",
    "vector_store",
    "vector_store/faiss_index",
]

FILES = [
    "backend/ingest.py",
    "backend/preprocess.py",
    "backend/chunk.py",
    "backend/vector_store.py",
    "backend/build_index.py",
    "backend/query.py",
    "README.md",
    "requirements.txt",
    ".gitignore",
]

def main():
    for folder in FOLDERS:
        (BASE / folder).mkdir(parents=True, exist_ok=True)

    for file in FILES:
        path = BASE / file
        if not path.exists():
            path.touch()

    print("✔ Project folders and empty files created successfully")

if __name__ == "__main__":
    main()
