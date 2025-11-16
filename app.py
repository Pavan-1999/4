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
    """Small typo fixes + expansions for common abbreviations."""
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
    # Compact, fast SBERT model suitable for sentence/title similarity
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
    """
    Build a character n-gram TF-IDF model.
    Character n-grams make the model robust to typos and short titles.
    """
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
    """
    Encode all existing titles once with SBERT.
    """
    model = load_sbert_model()
    embs = model.encode(titles.tolist(), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs)

def compute_hybrid(
    query: str,
    titles: pd.Series,
    tfidf_vec,
    tfidf_mat,
    sbert_embs,
    tfidf_cutoff: float = 0.10,
    w_sbert: float = 0.80,
    w_tfidf: float = 0.20,
):
    """
    Compute TF-IDF %, SBERT %, and an adaptive Hybrid % for a query.

    Design choices for interpretability:
    - TF-IDF: use raw cosine similarity (0–1), NOT forced max=1.
    - SBERT: cosine in [-1,1] mapped to [0,1].
    - Hybrid:
        * If TF-IDF < tfidf_cutoff (little lexical overlap) -> Hybrid = SBERT.
        * Else -> Hybrid = 0.8*SBERT + 0.2*TF-IDF.
      This prevents unrelated titles (low TF-IDF) from getting artificially high Hybrid scores.
    """
    if not query or not query.strip():
        return None

    q = expand_query(query)

    # ---------- TF-IDF SIMILARITY (raw cosine, no per-query 0–1 scaling) ----------
    q_tfidf = tfidf_vec.transform([q])
    sim_tfidf_raw = cosine_similarity(q_tfidf, tfidf_mat).ravel()
    # In practice cosine for tf-idf is already [0,1], but clip for safety
    sim_tfidf = np.clip(sim_tfidf_raw, 0.0, 1.0)

    # ---------- SBERT SIMILARITY ----------
    model = load_sbert_model()
    q_emb = model.encode([q], normalize_embeddings=True, show_progress_bar=False)
    sim_sbert_raw = (sbert_embs @ q_emb[0]).ravel()          # in [-1, 1]
    sim_sbert = (sim_sbert_raw + 1.0) / 2.0                  # -> [0, 1]
    sim_sbert = np.clip(sim_sbert, 0.0, 1.0)

    # ---------- ADAPTIVE HYBRID ----------
    # If lexical overlap is very small, trust semantic similarity only.
    use_sbert_only = sim_tfidf < tfidf_cutoff
    hybrid = np.empty_like(sim_sbert)

    # Case 1: very low lexical overlap -> Hybrid = SBERT
    hybrid[use_sbert_only] = sim_sbert[use_sbert_only]

    # Case 2: reasonable lexical overlap -> blend SBERT + TF-IDF
    hybrid[~use_sbert_only] = (
        w_sbert * sim_sbert[~use_sbert_only]
        + w_tfidf * sim_tfidf[~use_sbert_only]
    )

    # ---------- CONVERT TO PERCENTAGES ----------
    tfidf_pct = np.nan_to_num(sim_tfidf * 100.0, nan=0.0)
    sbert_pct = np.nan_to_num(sim_sbert * 100.0, nan=0.0)
    hybrid_pct = np.nan_to_num(np.clip(hybrid, 0, 1) * 100.0, nan=0.0)

    return tfidf_pct, sbert_pct, hybrid_pct

# ===================== UI: DATA SOURCE =====================
with st.expander("📄 Data Source (Google Sheet CSV)"):
    sheet_url = st.text_input(
        "Paste your **published-to-web CSV** link (must end with `output=csv`):",
        value=DEFAULT_SHEET_URL,
    )
    st.caption("Google Sheets → File → Share → Publish to web → CSV → copy the link.")

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

    # ---------- DATASET OVERVIEW ----------
    st.markdown("### Dataset overview")
    col_a, col_b = st.columns(2)
    col_a.metric("Total projects loaded", len(titles))

    # If there's a Year column, show projects per year
    year_col = None
    for cand in ["Year", "year", "Academic Year", "Year_of_Completion"]:
        if cand in df.columns:
            year_col = cand
            break

    if year_col:
        year_counts = df[year_col].value_counts().sort_index()
        col_b.bar_chart(year_counts, use_container_width=True)
        col_b.caption(f"Projects per {year_col}")
    else:
        col_b.write("Add a 'Year' column in your sheet to see projects per year.")

    with st.expander("Preview (first 5 titles)"):
        st.write(pd.DataFrame(titles.head(5), columns=["Title"]))

    # Build models (cached)
    tfidf_vec, tfidf_mat = build_tfidf(titles)
    sbert_embs = embed_titles(titles)

    # ===================== UI: QUERY & RESULTS =====================
    st.markdown("### Check a new capstone title")
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
            tfidf_pct, sbert_pct, hybrid_pct = sims
            order = np.argsort(-hybrid_pct)
            k = int(min(top_k, len(titles)))
            idx = order[:k]

            results = pd.DataFrame({
                "Rank": np.arange(1, k + 1),
                "Existing Title": titles.iloc[idx].values,
                "Hybrid %": np.round(hybrid_pct[idx], 2),
                "SBERT %": np.round(sbert_pct[idx], 2),
                "TF-IDF %": np.round(tfidf_pct[idx], 2),
            })

            st.subheader("Top Matches")
            st.dataframe(results, use_container_width=True)

            # ---------- METRICS FOR THIS QUERY ----------
            st.markdown("#### Similarity summary (Top matches)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Highest Hybrid %", f"{results['Hybrid %'].max():.1f}")
            m2.metric("Lowest Hybrid %", f"{results['Hybrid %'].min():.1f}")
            m3.metric("Average Hybrid %", f"{results['Hybrid %'].mean():.1f}")

            # ---------- BAR CHART OF HYBRID SCORES ----------
            st.markdown("#### Hybrid similarity scores for Top matches")
            chart_df = results[["Existing Title", "Hybrid %"]].set_index("Existing Title")
            st.bar_chart(chart_df, use_container_width=True)

            # Download
            st.download_button(
                "⬇️ Download results (CSV)",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="similarity_results.csv",
                mime="text/csv",
            )

            with st.expander("How to interpret the scores"):
                st.markdown(
                    """
                    **TF-IDF % (lexical similarity)**  
                    - Based on character n-gram cosine similarity.  
                    - Values close to 0% → very little word/character overlap.  
                    - Values above ~30–40% → strong overlap in phrases or spelling.  

                    **SBERT % (semantic similarity)**  
                    - Based on SBERT cosine similarity, mapped from [-1,1] to [0–100%].  
                    - Captures meaning, synonyms, and paraphrasing.  

                    **Hybrid % (adaptive blend)**  
                    - If TF-IDF < 10% (almost no lexical overlap), Hybrid = SBERT %.  
                    - If TF-IDF ≥ 10%, Hybrid = 0.8 × SBERT % + 0.2 × TF-IDF %.  
                    - Rough guideline:  
                        * < 40% → weak similarity  
                        * 40–70% → moderate topic overlap (worth reviewing)  
                        * > 70% → strong similarity; potential duplication risk  

                    Scores are always **relative to the projects in the current Google Sheet**.  
                    The tool is designed to **assist faculty**, not replace academic judgement.
                    """
                )

st.markdown("---")
st.caption("Tip: Keep titles concise but specific: domain + method + context (e.g., 'AI-based demand forecasting for retail').")
