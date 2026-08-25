"""
STEP 3 — Streamlit UI

Run with: streamlit run app.py
"""

import streamlit as st
from contradiction_engine import ContradictionDetector

st.set_page_config(
    page_title="Scientific Contradiction Detector",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Scientific Contradiction Detector")
st.caption(
    "Analyse the scientific consensus on any biomedical claim "
    "using RAG + LLM over PubMed papers."
)

# Load models once for the entire Streamlit session
@st.cache_resource(show_spinner="Loading AI models (only on first run)...")
def load_detector() -> ContradictionDetector:
    return ContradictionDetector()


try:
    detector = load_detector()
except Exception as e:
    st.error(
        f"Failed to load the system: {e}\n\n"
        "Make sure you have run:\n"
        "1. `python fetch_papers.py`\n"
        "2. `python ingest.py`\n"
        "3. `ollama pull llama3.2`"
    )
    st.stop()

# --- Input ---
st.subheader("Enter a scientific claim")
col_input, col_btn = st.columns([5, 1])

with col_input:
    claim = st.text_input(
        label="Claim",
        placeholder="e.g. intermittent fasting extends lifespan in humans",
        label_visibility="collapsed",
    )

with col_btn:
    analyze_btn = st.button("Analyse", type="primary", use_container_width=True)

# Clickable examples
st.caption("Examples:")
example_cols = st.columns(3)
examples = [
    "intermittent fasting extends human lifespan",
    "metformin has anti-aging effects in humans",
    "gut microbiome influences Parkinson's disease",
]
for col, ex in zip(example_cols, examples):
    if col.button(ex, use_container_width=True):
        claim = ex
        analyze_btn = True

# --- Analysis ---
if analyze_btn and claim:
    with st.spinner("Analysing..."):
        result = detector.analyze(claim)

    st.divider()

    # --- Summary metrics ---
    n_sup = len(result["supporting"])
    n_con = len(result["contradicting"])
    n_neu = len(result["neutral"])
    score = result["consensus_score"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("✅ Supporting papers", n_sup)
    m2.metric("❌ Contradicting papers", n_con)
    m3.metric("⬜ Neutral papers", n_neu)
    m4.metric("Consensus score", f"{score:.0%}")

    # Colour-coded consensus bar
    bar_color = (
        "green" if score >= 0.65
        else "red" if score <= 0.35
        else "orange"
    )
    st.markdown(
        f"""
        <div style="background:#eee; border-radius:8px; height:18px; width:100%;">
          <div style="background:{bar_color}; border-radius:8px; height:18px; width:{score*100:.0f}%;"></div>
        </div>
        <p style="color:{bar_color}; font-size:0.85em; margin-top:4px;">
          {'Strong Support' if score >= 0.65 else 'Strong Contradiction' if score <= 0.35 else 'Contested'}
          &nbsp;({score:.0%} of non-neutral papers support the claim)
        </p>
        """,
        unsafe_allow_html=True,
    )

    # --- LLM report ---
    st.subheader("Scientific report")
    st.write(result["report"])

    # --- Paper detail ---
    st.subheader("Retrieved papers")
    tab_sup, tab_con, tab_neu = st.tabs(
        [f"✅ Supporting ({n_sup})", f"❌ Contradicting ({n_con})", f"⬜ Neutral ({n_neu})"]
    )

    def render_papers(papers: list[dict]):
        if not papers:
            st.info("No papers in this category.")
            return
        sorted_papers = sorted(papers, key=lambda x: x["confidence"], reverse=True)
        for p in sorted_papers:
            label = f"**{p['title']}** ({p['year']})"
            with st.expander(label):
                st.write(p["text"])
                st.link_button("Open on PubMed", p["url"])

    with tab_sup:
        render_papers(result["supporting"])
    with tab_con:
        render_papers(result["contradicting"])
    with tab_neu:
        render_papers(result["neutral"])

elif analyze_btn and not claim:
    st.warning("Please enter a claim before analysing.")
