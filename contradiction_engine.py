"""
Core of the project: RAG + LLM pipeline to detect scientific contradictions.

Pipeline:
  claim → embedding → ChromaDB retrieval → LLM stance classification → LLM synthesis
"""

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_MODEL,
    TOP_K_RETRIEVAL,
)

_STANCE_PROMPT = (
    "Read the scientific abstract and determine if it SUPPORTS, CONTRADICTS, "
    "or is NEUTRAL toward the claim below.\n\n"
    "Claim: \"{claim}\"\n\n"
    "Abstract: \"{abstract}\"\n\n"
    "Reply with exactly one word — SUPPORT, CONTRADICT, or NEUTRAL — and nothing else."
)

_SYNTHESIS_PROMPT = PromptTemplate.from_template(
    """You are a rigorous scientific analyst. Your task is to assess the scientific consensus on the claim below, based exclusively on the evidence provided.

CLAIM: {claim}

SUPPORTING EVIDENCE ({n_support} papers):
{supporting}

CONTRADICTING EVIDENCE ({n_contra} papers):
{contradicting}

Write a concise scientific report (3-5 paragraphs) that:
1. Summarizes what the supporting papers collectively argue
2. Summarizes what the contradicting papers collectively argue
3. Identifies key nuances, conditions, or populations where results diverge
4. Concludes with an overall consensus label from this scale:
   STRONG SUPPORT | MODERATE SUPPORT | CONTESTED | MODERATE CONTRADICTION | STRONG CONTRADICTION

Be precise. Do not add information beyond what is in the evidence. Answer in the same language as the claim.
"""
)


def _format_papers_for_prompt(papers: list[dict], max_n: int = 5) -> str:
    """Format the highest-confidence papers as context snippets for the LLM."""
    top = sorted(papers, key=lambda x: x["confidence"], reverse=True)[:max_n]
    lines = []
    for p in top:
        snippet = p["text"][:400].replace("\n", " ")
        lines.append(f"- [{p['year']}] {p['title']}\n  {snippet}...")
    return "\n".join(lines) if lines else "None found."


class ContradictionDetector:
    def __init__(self):
        print("Loading models (may download from HuggingFace on first run)...")

        self.embedder = SentenceTransformer(EMBED_MODEL)
        print(f"  ✓ Embedding model: {EMBED_MODEL}")

        self.llm = OllamaLLM(model=LLM_MODEL, temperature=0)
        print(f"  ✓ LLM: {LLM_MODEL} via Ollama")

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"  ✓ ChromaDB: {self.collection.count()} documents loaded\n")

    def retrieve(self, claim: str) -> list[dict]:
        """Retrieve the TOP_K_RETRIEVAL papers most semantically similar to the claim."""
        query_vec = self.embedder.encode([claim]).tolist()
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=TOP_K_RETRIEVAL,
            include=["documents", "metadatas", "distances"],
        )

        papers = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            papers.append(
                {
                    "text": doc,
                    "title": meta.get("title", ""),
                    "pmid": meta.get("pmid", ""),
                    "year": meta.get("year", ""),
                    "url": meta.get("url", ""),
                    # cosine distance → similarity score (higher = more similar)
                    "similarity": round(1 - float(dist), 3),
                }
            )
        return papers

    def classify_stance(self, claim: str, papers: list[dict]) -> list[dict]:
        """
        Classify each paper as SUPPORT / CONTRADICT / NEUTRAL using the LLM.
        More accurate than a generic NLI model on biomedical domain text.
        """
        for i, paper in enumerate(papers):
            snippet = paper["text"][:900].replace("\n", " ")
            prompt = _STANCE_PROMPT.format(claim=claim, abstract=snippet)
            raw = self.llm.invoke(prompt).strip().upper()

            if "SUPPORT" in raw:
                stance = "SUPPORT"
            elif "CONTRADICT" in raw:
                stance = "CONTRADICT"
            else:
                stance = "NEUTRAL"

            paper["stance"] = stance
            # LLM does not output calibrated probabilities; use fixed display values
            paper["confidence"] = 0.88
            paper["probs"] = {
                "support": 0.88 if stance == "SUPPORT" else 0.06,
                "contradict": 0.88 if stance == "CONTRADICT" else 0.06,
                "neutral": 0.88 if stance == "NEUTRAL" else 0.06,
            }
            print(f"      [{i+1}/{len(papers)}] {stance}")

        return papers

    def synthesize(self, claim: str, classified_papers: list[dict]) -> str:
        """Generate a narrative report using the LLM, grounded in the classified papers."""
        supporting = [p for p in classified_papers if p["stance"] == "SUPPORT"]
        contradicting = [p for p in classified_papers if p["stance"] == "CONTRADICT"]

        prompt = _SYNTHESIS_PROMPT.format(
            claim=claim,
            n_support=len(supporting),
            n_contra=len(contradicting),
            supporting=_format_papers_for_prompt(supporting),
            contradicting=_format_papers_for_prompt(contradicting),
        )
        return self.llm.invoke(prompt)

    def analyze(self, claim: str) -> dict:
        """Full pipeline: retrieval + LLM stance classification + LLM synthesis."""
        print(f"[1/3] Retrieval for: '{claim}'")
        papers = self.retrieve(claim)

        print(f"[2/3] Stance classification on {len(papers)} papers...")
        classified = self.classify_stance(claim, papers)

        supporting = [p for p in classified if p["stance"] == "SUPPORT"]
        contradicting = [p for p in classified if p["stance"] == "CONTRADICT"]
        neutral = [p for p in classified if p["stance"] == "NEUTRAL"]

        print(
            f"      → {len(supporting)} support | "
            f"{len(contradicting)} contradict | "
            f"{len(neutral)} neutral"
        )

        print("[3/3] LLM synthesis...")
        report = self.synthesize(claim, classified)

        total_non_neutral = len(supporting) + len(contradicting)
        consensus_score = len(supporting) / total_non_neutral if total_non_neutral > 0 else 0.5

        return {
            "claim": claim,
            "report": report,
            "supporting": supporting,
            "contradicting": contradicting,
            "neutral": neutral,
            "consensus_score": consensus_score,
            "total_retrieved": len(classified),
        }
