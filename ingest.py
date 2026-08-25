"""
STEP 2 — Compute abstract embeddings and populate ChromaDB

Usage: python ingest.py

ChromaDB is a vector store: instead of keyword search,
it retrieves the documents most semantically similar to your query vector.
"""

import glob
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, COLLECTION_NAME, DATA_DIR, EMBED_MODEL


def load_papers() -> list[dict]:
    """Load all JSON files from the data/ directory."""
    files = glob.glob(str(Path(DATA_DIR) / "*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in '{DATA_DIR}/'. "
            "Run first: python fetch_papers.py"
        )

    all_papers = []
    seen_pmids = set()
    for path in files:
        with open(path, encoding="utf-8") as f:
            papers = json.load(f)
        for p in papers:
            if p["pmid"] not in seen_pmids:
                all_papers.append(p)
                seen_pmids.add(p["pmid"])
        print(f"  Loaded '{path}' ({len(papers)} papers)")

    print(f"Total unique papers: {len(all_papers)}")
    return all_papers


def build_vectorstore(papers: list[dict]) -> None:
    """Compute embeddings and insert all documents into ChromaDB."""
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop existing collection so we can re-index cleanly
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Existing collection dropped (re-indexing).")
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    # Embed title + abstract together for richer semantic representation
    texts = [f"{p['title']}\n\n{p['abstract']}" for p in papers]
    ids = [p["pmid"] for p in papers]
    metadatas = [
        {
            "pmid": p["pmid"],
            "title": p["title"],
            "year": p.get("year", ""),
            "url": p["url"],
        }
        for p in papers
    ]

    print(f"Computing embeddings for {len(texts)} documents...")
    # batch_size=64 is a good speed/memory tradeoff
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    # Insert into ChromaDB in chunks of 500
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
        )
        print(f"  Inserted {min(i + batch_size, len(ids))}/{len(ids)} into ChromaDB...")

    print(f"\nChromaDB populated with {collection.count()} documents.")
    print(f"Vector store saved to '{CHROMA_PATH}/'")
    print("Next step: streamlit run app.py")


def main():
    print("=== STEP 2: Ingesting papers into ChromaDB ===\n")
    papers = load_papers()
    build_vectorstore(papers)


if __name__ == "__main__":
    main()
