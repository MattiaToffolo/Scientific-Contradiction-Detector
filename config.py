import os
from dotenv import load_dotenv

load_dotenv()

# Free PubMed API key: https://www.ncbi.nlm.nih.gov/account/
# Without key: 3 req/sec | With key: 10 req/sec
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Embedding model: converts text into semantic vectors
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Local LLM via Ollama for the final synthesis
LLM_MODEL = "llama3.2"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "scientific_papers"

TOP_K_RETRIEVAL = 20
DATA_DIR = "./data"
