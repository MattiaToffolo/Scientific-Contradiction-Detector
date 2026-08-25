# Scientific Contradiction Detector

End-to-end AI system for automated scientific literature review. Type a biomedical claim, get a structured evidence report: which papers support it, which contradict it, and a written synthesis. Built with local LLMs (Ollama + llama3.2), RAG (ChromaDB + sentence-transformers), and Streamlit. Zero cost, fully private.

## Why this exists

Scientific literature is enormous and contradictory. For almost every health claim, there are studies that support it and studies that refute it, scattered across thousands of papers published over decades. Reading them manually takes days. Hiring a researcher to do it costs money. And even then, human review is selective and prone to confirmation bias.

This tool automates that process. You type a claim like "metformin slows aging", "high-protein diets increase cancer risk", "gut microbiome influences Parkinson's disease", and within minutes you get a structured report that tells you what the current scientific literature actually says: how many papers support the claim, how many contradict it, and a written synthesis of the evidence.

**What you gain:**
- Hours of manual literature review compressed into a few minutes
- An objective, evidence-based starting point for any health or biomedical decision
- A repeatable, auditable process (every result is traceable to specific papers with PubMed links)
- Zero subscription cost: the system runs entirely on your own machine with open-source models

**Who this is for:**
- Researchers who need a first-pass scan of evidence before diving deep
- Consultants and analysts who need to quickly validate or challenge a scientific claim
- Anyone who needs to separate marketing claims from peer-reviewed evidence

**Example questions you can answer in minutes:**
- Does intermittent fasting actually extend lifespan in humans, or mainly in mice?
- Is the evidence for metformin as an anti-aging drug strong enough to act on?
- How contested is the link between gut microbiome and neurological disease?

Everything runs locally. No data is sent to any external service. No API subscription required.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Ollama](https://img.shields.io/badge/LLM-llama3.2-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## How it works

1. **Fetch**: downloads abstracts from PubMed via the free E-utilities API
2. **Ingest**: computes sentence embeddings and stores them in a local ChromaDB vector store
3. **Analyse**: retrieves the 20 most relevant papers, asks an LLM to classify each one as SUPPORT / CONTRADICT / NEUTRAL, then generates a narrative synthesis report

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
pip install -r requirements.txt
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

You can run `fetch_papers.py` and `ingest.py` in a second terminal while the app is open. No restart needed.

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
├── requirements.txt
├── .env.example             # Copy to .env and add your PubMed API key
```

---

## Optional: PubMed API key

Register for a free key at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/). It raises the rate limit from 3 to 10 req/sec and speeds up `fetch_papers.py` significantly.

```bash
cp .env.example .env
# Edit .env and add your key
```
