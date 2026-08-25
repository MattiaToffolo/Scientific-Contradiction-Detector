"""
STEP 1 — Download abstracts from PubMed and save them to data/

Usage: python fetch_papers.py --query "intermittent fasting lifespan" --max 300
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from config import PUBMED_API_KEY, PUBMED_BASE_URL, DATA_DIR


def search_pubmed(query: str, max_results: int) -> list[str]:
    """Return a list of PubMed IDs relevant to the query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    resp = requests.get(f"{PUBMED_BASE_URL}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    ids = resp.json()["esearchresult"]["idlist"]
    print(f"Found {len(ids)} papers for query: '{query}'")
    return ids


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Download title + abstract for each PMID, in batches of 200."""
    papers = []
    batch_size = 200

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if PUBMED_API_KEY:
            params["api_key"] = PUBMED_API_KEY

        resp = requests.get(f"{PUBMED_BASE_URL}/efetch.fcgi", params=params, timeout=60)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", "")
            title = article.findtext(".//ArticleTitle", "").strip()

            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (p.text or "").strip() for p in abstract_parts if p.text
            )

            pub_year = article.findtext(".//PubDate/Year", "")

            if abstract and len(abstract) > 100:
                papers.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "year": pub_year,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    }
                )

        processed = min(i + batch_size, len(pmids))
        print(f"  Processed {processed}/{len(pmids)} papers ({len(papers)} with abstract)...")

        # Respect PubMed rate limit: 0.4s without key, 0.15s with key
        time.sleep(0.15 if PUBMED_API_KEY else 0.4)

    return papers


def main():
    parser = argparse.ArgumentParser(description="Fetch papers from PubMed")
    parser.add_argument(
        "--query",
        type=str,
        default="intermittent fasting lifespan longevity",
        help="PubMed search query",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=300,
        help="Maximum number of papers to download",
    )
    args = parser.parse_args()

    Path(DATA_DIR).mkdir(exist_ok=True)

    pmids = search_pubmed(args.query, args.max)
    if not pmids:
        print("No papers found. Try a different query.")
        return

    papers = fetch_abstracts(pmids)

    safe_query = args.query.replace(" ", "_")[:50]
    output_path = Path(DATA_DIR) / f"papers_{safe_query}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(papers)} papers to '{output_path}'")
    print("Next step: python ingest.py")


if __name__ == "__main__":
    main()
