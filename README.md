# Scientific Contradiction Detector

A local AI prototype designed to help researchers and analysts organise biomedical evidence before a manual review. Enter a claim to explore potentially supporting, contradicting and neutral abstracts, with PubMed links and an LLM-generated summary. Built with Python, RAG, ChromaDB, LangChain, llama3.2 via Ollama, and Streamlit.

**Research prototype, not a validated scientific or medical decision tool.** Classification errors remain substantial. All labels and summaries require checking against the original sources.

## Why this exists

Finding and comparing biomedical studies is time-consuming. Relevant evidence can be scattered across publications with different populations, interventions and outcomes.

This project automates document organisation and produces a first-pass summary for manual investigation. It searches only the corpus you have downloaded and indexed, not all of PubMed at query time. It does not replace a systematic literature review, and time savings have not been measured in a user study.

**What you gain:**
- A single interface for exploring retrieved abstracts and their predicted stances
- Source links for checking the underlying studies
- A reproducible evaluation script for measuring classification errors
- Local inference without paid LLM APIs; hardware and electricity costs still apply

**Who this is for:**
- Researchers who need a first-pass scan of evidence before diving deep
- Consultants and analysts who need to quickly validate or challenge a scientific claim
- Anyone who needs to separate marketing claims from peer-reviewed evidence

**Example topics to investigate:**
- Does intermittent fasting actually extend lifespan in humans, or mainly in mice?
- What findings have been reported about metformin and aging?
- How contested is the link between gut microbiome and neurological disease?

LLM inference runs locally through Ollama. Downloading papers sends search terms to PubMed, model downloads contact external registries, and evaluation downloads SciFact. This is not an entirely offline workflow. Review dependency telemetry settings before using sensitive data.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Ollama](https://img.shields.io/badge/LLM-llama3.2-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## How it works

1. **Fetch**: downloads abstracts from PubMed via the free E-utilities API
2. **Ingest**: computes sentence embeddings and stores them in a local ChromaDB vector store
3. **Analyse**: retrieves the 20 most relevant papers, asks an LLM to classify each one as SUPPORT / CONTRADICT / NEUTRAL, then generates a narrative synthesis report

The app currently uses the original zero-shot classifier on the first 900 characters of each retrieved text. An experimental few-shot classifier uses full abstracts and structured JSON containing evidence, reasoning and a stance label. It is available for evaluation but is not enabled in the app: its development gains did not generalise to the held-out sample.

```
User claim
    │
    ▼
[all-MiniLM-L6-v2]  ── BiEncoder embedding
    │
    ▼
[ChromaDB HNSW]  ────── top-20 semantic search
    │
    ▼
[llama3.2 via Ollama]  ── stance classification x 20 papers
    │
    ▼
[llama3.2 via Ollama]  ── narrative synthesis report
    │
    ▼
[Streamlit UI]  ──────── metrics + consensus bar + expandable papers
```

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/download) with `llama3.2` pulled (`ollama pull llama3.2`)
- All Python dependencies in `requirements.txt`

> **Windows note:** create the virtual environment at a short path (e.g. `C:\sc\venv`) to avoid the 260-character path limit with PyTorch's deep file tree.

---

## Installation

```bash
# 1. Pull the LLM model (~2 GB, one-time)
ollama pull llama3.2

# 2. Create venv at a short path (Windows) or standard path (Mac/Linux)
python -m venv C:\sc\venv          # Windows
python -m venv venv                 # Mac/Linux

# 3. Activate the venv (required every new terminal session)
C:\sc\venv\Scripts\Activate.ps1    # Windows PowerShell
source venv/bin/activate            # Mac/Linux

# 4. Install dependencies
python -m pip install -r requirements.txt
```

---

## Usage

### Every time you open a new terminal

```powershell
C:\sc\venv\Scripts\Activate.ps1   # activate venv (Windows)
cd path/to/Scientific-Contradiction-Detector
```

Make sure Ollama is running (system tray on Windows, `ollama serve` on Mac/Linux).

### Download papers for a topic

```bash
python fetch_papers.py --query "intermittent fasting lifespan longevity" --max 300
```

### Index the papers into ChromaDB

```bash
python ingest.py
```

### Launch the app

```bash
python -m streamlit run app.py
# Opens at http://localhost:8501
```

Activate the virtual environment in every new terminal. Fetching additional papers can run separately. Before re-indexing, stop Streamlit with Ctrl+C, run `python ingest.py`, then restart the app: indexing deletes and recreates the collection, so cached database handles should not be assumed to refresh safely.

Download papers for each topic before trying the example claims below. The repository does not include the generated corpus or vector database.

## Evaluation and Limitations

Evaluation uses [SciFact](https://github.com/allenai/scifact), a human-annotated scientific claim verification dataset. These are custom, class-balanced subsets of its public development split, not official leaderboard results. Each example is a claim/abstract pair. Cited documents without annotated supporting or contradicting evidence are mapped to NEUTRAL.

| Configuration | Sample | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Original zero-shot baseline, 900 characters | Development: 60 pairs | 53.3% | 0.487 |
| Original prompt with full abstracts | Same development pairs | 55.0% | 0.525 |
| Few-shot prompt, full abstracts, structured JSON | Same development pairs | 66.7% | 0.612 |
| Original baseline | Held out: 45 pairs | 48.9% | 0.448 |
| Selected structured variant | Same held-out pairs | 42.2% | 0.339 |

The development sample contains 20 examples per class; the held-out sample contains 15 per class, with no claim IDs or document IDs shared with development. The structured variant was selected on development data before evaluating the holdout. The holdout result rejected it as an app upgrade. Another development experiment with stricter entity rules and quote-based label filtering reached 53.3% accuracy and 0.535 macro-F1 and was also not selected.

The selected variant improved contradiction recall but missed every NEUTRAL example in the holdout. Its 66.7% development accuracy must not be presented as general test accuracy. With only 45 held-out pairs, accuracy uncertainty is substantial: approximate 95% Wilson intervals are 34.9%-63.0% for the baseline and 29.0%-56.7% for the structured variant. These results do not establish a statistically significant difference.

Additional limitations:

* Evaluation covers stance classification, not retrieval relevance or report faithfulness.
* These subsets do not establish performance on the application's PubMed corpus, and public benchmark exposure during model pretraining cannot be excluded.
* The displayed consensus score is the fraction of predicted SUPPORT among non-neutral retrieved papers. It is not scientific consensus, a confidence estimate, or a measure of study quality. If every paper is neutral, the current implementation returns 50% as a fallback, not evidence of disagreement.
* The baseline's internal confidence and probability fields are fixed placeholders, not measured probabilities. The structured variant's quote-presence flag checks only text matching, not whether the quote logically supports the label.
* Only abstracts are assessed. A report is generated from at most five supporting and five contradicting snippets, each limited to 400 characters. Findings, populations and safety trade-offs can therefore be missed.

### Reproduce the Comparison

Run these from the project root with the virtual environment active and Ollama running. Each output directory must be new: the script refuses to overwrite saved predictions. Evaluation does not require rebuilding ChromaDB.

```powershell
python evaluate_stance.py --self-test

# Original baseline on the development sample
python evaluate_stance.py --baseline --per-class 20 --seed 20261017 --output evaluation_results/reproduction_dev_baseline

# Experimental classifier on the exact same development pairs
python evaluate_stance.py --input-sample evaluation_results/reproduction_dev_baseline/sample.json --output evaluation_results/reproduction_dev_structured

# New pairs excluding development claims and documents
python evaluate_stance.py --baseline --per-class 15 --seed 20261018 --exclude-sample evaluation_results/reproduction_dev_baseline/sample.json --output evaluation_results/reproduction_holdout_baseline

# Selected experimental configuration on the same held-out pairs
python evaluate_stance.py --input-sample evaluation_results/reproduction_holdout_baseline/sample.json --output evaluation_results/reproduction_holdout_structured
```

Each run saves its sample, raw responses, predictions, per-class metrics, confusion matrix, accuracy interval, prompt, runtime settings and source hash. Local inference can take tens of minutes per run. Fixed seeds aid repeatability but do not guarantee identical outputs across software versions or hardware. Do not tune on the holdout and then describe it as an untouched test set.

---

## Example claims to try

```
intermittent fasting extends lifespan in humans
metformin has anti-aging effects
gut microbiome influences Parkinson's disease progression
low-dose aspirin reduces cardiovascular mortality
high-protein diet increases cancer risk
```

---

## Tech stack

| Component | Tool |
|---|---|
| Local LLM | Ollama + llama3.2 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB |
| LLM pipeline | LangChain + langchain-ollama |
| UI | Streamlit |
| Data source | PubMed E-utilities API (free) |

---

## Project structure

```
├── fetch_papers.py          # Step 1: download abstracts from PubMed
├── ingest.py                # Step 2: embeddings + ChromaDB indexing
├── contradiction_engine.py  # Core RAG + LLM pipeline
├── app.py                   # Streamlit UI
├── config.py                # Centralised configuration constants
├── evaluate_stance.py       # SciFact stance evaluation and metric/parser checks
├── requirements.txt
└── .env.example             # Copy to .env and add your PubMed API key
```

---

## Optional: PubMed API key

Register for a free key at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/). It raises the rate limit from 3 to 10 req/sec and speeds up `fetch_papers.py` significantly.

```bash
cp .env.example .env
# Edit .env and add your key
```
