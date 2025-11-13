import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Capstone Similarity Checker", page_icon="🔎", layout="wide")
st.title("🔎 Capstone Similarity Checker")
st.caption("Live Google Sheet → Hybrid Similarity (SBERT + TF-IDF) → Top matches with % scores")

# ===================== DEFAULTS =====================
DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQOt3ScW1TkCpKVCP2vNMbNSahbMkaFZBARjoTRe267tQdX_E_hC8o3bXTjwkhPxdXKtKfq1_dWLZMU/pub?gid=1929751519&single=true&output=csv"
)
PREFERRED_TITLE_NAMES = ["Title", "Project Title", "Capstone Title", "title", "project_title", "Project"]

# ===================== HELPERS =====================
def expand_query(q: str) -> str:
    """Tiny typo fixes + micro expansions for short queries."""
    q2 = q.lower().strip()
    fixes = {
        "artifical": "artificial",
        "intellegence": "intelligence",
        "machne": "machine",
        "lern": "learning",
    }
    for wrong, right in fixes.items():
        q2 = q2.replace(wrong, right)
    if q2 == "ai" or " ai " in f" {q2} ":
        q2 += " artificial intelligence"
    if q2 == "ml" or " ml " in f" {q2} ":
        q2 += " machine learning"
    if "nlp" in q2:
        q2 += " natural language processing"
    return q2

@st.cache_resource(show_spinner=False)
def load_sbert_model():
    from sentence_transformers import SentenceTransformer
    # small & fast; good for short titles
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

@st.cache_data(show_spinner=False)
def detect_title_column(df: pd.DataFrame) -> str:
    for c in PREFERRED_TITLE_NAMES:
        if c in df.columns:
            return c
    # fallback: first text-like column
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            return c
    raise ValueError("No suitable text column found.")

@st.cache_data(show_spinner=False)
def clean_titles(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna("").str.strip()
    s = s[s.str.len() > 0].reset_index(drop=True)
    return s

@st.cache_data(show_spinner=False)
def build_tfidf(titles: pd.Series):
    # Character n-grams are robust to short titles & typos
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=1,
        max_df=1.0,
    )
    mat = vec.fit_transform(titles.tolist())
    return vec, mat

@st.cache_data(show_spinner=False)
def embed_titles(titles: pd.Series):
    model = load_sbert_model()
    embs = model.encode(titles.tolist(), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs)

def compute_hybrid(query: str, titles: pd.Series, tfidf_vec, tfidf_mat, sbert_embs,
                   w_sbert: float = 0.80, w_tfidf: float = 0.20):
    if not query or not query.strip():
        return None
    q = expand_query(query)

    # TF-IDF cosine (rescale to [0,1])
    q_tfidf = tfidf_vec.transform([q])
    sim_tfidf = cosine_similarity(q_tfidf, tfidf_mat).ravel()
    sim_tfidf = (sim_tfidf - sim_tfidf.min()) / (sim_tfidf.max() - sim_tfidf.min() + 1e-12)

    # SBERT cosine in [-1,1] -> normalize to [0,1]
    model = load_sbert_model()
    q_emb = model.encode([q], normalize_embeddings=True, show_progress_bar=False)
    sim_sbert = (sbert_embs @ q_emb[0]).ravel()
    sim_sbert = (sim_sbert + 1.0) / 2.0

    sim_hybrid = w_sbert * sim_sbert + w_tfidf * sim_tfidf

    # Return as percentages
    return (
        np.nan_to_num(sim_tfidf * 100.0, nan=0.0),
        np.nan_to_num(sim_sbert * 100.0, nan=0.0),
        np.nan_to_num(np.clip(sim_hybrid, 0, 1) * 100.0, nan=0.0),
    )

# ===================== UI: DATA SOURCE =====================
with st.expander("📄 Data Source (Google Sheet CSV)"):
    sheet_url = st.text_input(
        "Paste your **published-to-web CSV** link (must end with `output=csv`):",
        value=DEFAULT_SHEET_URL,
    )
    st.caption("Google Sheets → File → Share → Publish to web → CSV → copy link.")

# Load the sheet
df = None
err_box = st.empty()
try:
    df = fetch_csv(sheet_url)
except Exception as e:
    err_box.error(f"Could not load CSV. Ensure the link is public & ends with `output=csv`. Error: {e}")

if df is not None:
    try:
        picked_col = detect_title_column(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    titles = clean_titles(df[picked_col])
    if titles.empty:
        st.error("No non-empty titles found in the detected column.")
        st.stop()

    st.info(f"Loaded **{len(titles)}** titles from column **{picked_col}**.")
    with st.expander("Preview (first 5 titles)"):
        st.write(pd.DataFrame(titles.head(5), columns=["Title"]))

    # Build models (cached)
    tfidf_vec, tfidf_mat = build_tfidf(titles)
    sbert_embs = embed_titles(titles)

    # ===================== UI: QUERY & RESULTS =====================
    qcol1, qcol2 = st.columns([2, 1])
    with qcol1:
        query_title = st.text_input(
            "Enter a new capstone title to check:",
            placeholder="e.g., AI-driven anomaly detection for retail supply chains",
        )
    with qcol2:
        top_k = st.number_input("Top matches", min_value=1, max_value=50, value=5, step=1)

    if st.button("Check Similarity", type="primary", use_container_width=True):
        sims = compute_hybrid(query_title, titles, tfidf_vec, tfidf_mat, sbert_embs)
        if sims is None:
            st.warning("Please enter a title.")
        else:
            sim_tfidf_pct, sim_sbert_pct, sim_hybrid_pct = sims
            order = np.argsort(-sim_hybrid_pct)
            k = int(min(top_k, len(titles)))
            idx = order[:k]

            results = pd.DataFrame({
                "Rank": np.arange(1, k + 1),
                "Existing Title": titles.iloc[idx].values,
                "Hybrid %": np.round(sim_hybrid_pct[idx], 2),
                "SBERT %": np.round(sim_sbert_pct[idx], 2),
                "TF-IDF %": np.round(sim_tfidf_pct[idx], 2),
            })

            st.subheader("Top Matches")
            st.dataframe(results, use_container_width=True)

            # Download
            st.download_button(
                "⬇️ Download results (CSV)",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="similarity_results.csv",
                mime="text/csv",
            )

            with st.expander("How scoring works"):
                st.markdown(
                    "- **SBERT (80%)**: semantic similarity (meaning), cosine normalized to 0–100%.\n"
                    "- **TF-IDF (20%)**: character n-gram overlap (robust to typos & short titles).\n"
                    "- **Hybrid %** = 0.8 × SBERT % + 0.2 × TF-IDF %."
                )

st.markdown("---")
st.caption("Tip: Keep titles concise but specific: domain + method + context (e.g., 'AI-based demand forecasting for retail').")
