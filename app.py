import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Capstone Similarity Checker",
    page_icon="🔎",
    layout="wide",
)
st.title("🔎 Capstone Similarity Checker")
st.caption(
    "Live Google Sheet → Lexical gate (TF-IDF) + Semantic refinement (SBERT) → "
    "Top matches with interpretable similarity labels."
)

# ===================== DEFAULTS =====================
DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQOt3ScW1TkCpKVCP2vNMbNSahbMkaFZBARjoTRe267tQdX_E_hC8o3bXTjwkhPxdXKtKfq1_dWLZMU/pub?gid=1929751519&single=true&output=csv"
)
PREFERRED_TITLE_NAMES = [
    "Project Title",
    "Title",
    "Capstone Title",
    "title",
    "project_title",
    "Project",
]

# ===================== TEXT HELPERS =====================
def normalize_text(text: str) -> str:
    return str(text).strip().lower()


def expand_query(q: str) -> str:
    """Small typo fixes + expansions for common abbreviations."""
    q2 = normalize_text(q)
    fixes = {
        "artifical": "artificial",
        "intellegence": "intelligence",
        "machne": "machine",
        "lern": "learning",
    }
    for wrong, right in fixes.items():
        q2 = q2.replace(wrong, right)
    # simple expansions
    if q2 == "ai" or " ai " in f" {q2} ":
        q2 += " artificial intelligence"
    if q2 == "ml" or " ml " in f" {q2} ":
        q2 += " machine learning"
    if "nlp" in q2:
        q2 += " natural language processing"
    return q2


def strength_label(score: float) -> str:
    """Convert hybrid score in [0,1] to a verbal band."""
    if score < 0.30:
        return "Weak"
    elif score < 0.60:
        return "Moderate"
    elif score < 0.80:
        return "Strong"
    else:
        return "Very strong"


# ===================== MODEL LOADERS =====================
@st.cache_resource(show_spinner=False)
def load_sbert_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def build_lexical_index(titles: pd.Series):
    """
    Word-level TF-IDF for lexical similarity.
    Uses unigrams + bigrams and English stopwords.
    """
    titles_norm = titles.astype(str).apply(normalize_text)
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        max_df=1.0,
    )
    mat = vec.fit_transform(titles_norm.tolist())
    return vec, mat


@st.cache_data(show_spinner=False)
def embed_titles_sbert(titles: pd.Series):
    model = load_sbert_model()
    titles_norm = titles.astype(str).apply(normalize_text).tolist()
    embs = model.encode(
        titles_norm, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(embs)


# ===================== DATA HELPERS =====================
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
    raise ValueError("No suitable text column found for titles.")


@st.cache_data(show_spinner=False)
def clean_titles(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna("").str.strip()
    s = s[s.str.len() > 0].reset_index(drop=True)
    return s


# ===================== CORE SIMILARITY ENGINE =====================
def find_similar_titles(
    query: str,
    titles: pd.Series,
    lex_vec,
    lex_mat,
    sbert_embs,
    top_k: int = 5,
    lex_gate_strong: float = 0.15,   # strong lexical match threshold
    lex_gate_candidate: float = 0.10,  # min lexical sim to be considered candidate
    w_semantic: float = 0.70,
    w_lexical: float = 0.30,
):
    """
    Lexical gate + semantic refinement.

    Returns:
      results_df: DataFrame with matches and scores
      info: dict with summary flags / messages
    """
    q = expand_query(query)
    if not q:
        return None, {"error": "Please enter a non-empty title."}

    # ---------- LEXICAL SIMILARITY (WORD TF-IDF) ----------
    q_vec = lex_vec.transform([q])
    lex_sim = cosine_similarity(q_vec, lex_mat).ravel()  # in [0,1]
    max_lex = float(lex_sim.max())

    strong_lexical_match = max_lex >= lex_gate_strong

    # Candidates: titles with at least some lexical overlap
    cand_mask = lex_sim >= lex_gate_candidate
    cand_idx = np.where(cand_mask)[0]

    info = {
        "max_lexical_sim": max_lex,
        "strong_lexical_match": strong_lexical_match,
        "num_candidates": int(len(cand_idx)),
    }

    if len(cand_idx) == 0:
        info["message"] = (
            "No titles share enough wording with the query. "
            "We consider this title lexically novel in this dataset."
        )
        empty = pd.DataFrame(
            columns=["Existing Title", "Lexical", "Semantic", "Hybrid", "Hybrid %", "Strength"]
        )
        return empty, info

    # ---------- SEMANTIC SIMILARITY (SBERT) ON CANDIDATES ----------
    model = load_sbert_model()
    q_emb = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
    cand_sbert = sbert_embs[cand_idx]                       # embeddings for candidates
    sem_sim_raw = cand_sbert @ q_emb                        # cosine in [-1,1]
    sem_sim = np.clip((sem_sim_raw + 1.0) / 2.0, 0.0, 1.0)  # -> [0,1]

    cand_lex = lex_sim[cand_idx]                            # lexical sims for candidates

    # ---------- HYBRID SCORE ----------
    hybrid = (
        w_semantic * sem_sim +
        w_lexical * cand_lex
    )

    # ---------- RANK & FORMAT RESULTS ----------
    order = np.argsort(-hybrid)
    k = min(top_k, len(order))
    sel = order[:k]
    idx_sel = cand_idx[sel]

    results = pd.DataFrame({
        "Existing Title": titles.iloc[idx_sel].values,
        "Lexical": np.round(cand_lex[sel], 4),
        "Semantic": np.round(sem_sim[sel], 4),
        "Hybrid": np.round(hybrid[sel], 4),
    })
    results["Hybrid %"] = (results["Hybrid"] * 100).round(2)
    results["Strength"] = results["Hybrid"].apply(strength_label)

    if not strong_lexical_match:
        info["message"] = (
            "Only weak lexical overlaps found. These are the closest titles in the dataset, "
            "but the new title is likely a novel topic."
        )
    else:
        info["message"] = (
            "At least one existing title shares meaningful wording with the query. "
            "Review matches labelled 'Strong' or 'Very strong' for potential overlap."
        )

    return results, info


# ===================== UI: DATA SOURCE =====================
with st.expander("📄 Data Source (Google Sheet CSV)"):
    sheet_url = st.text_input(
        "Paste your **published-to-web CSV** link (must end with `output=csv`):",
        value=DEFAULT_SHEET_URL,
    )
    st.caption("Google Sheets → File → Share → Publish to web → CSV → copy the link.")

df = None
err_box = st.empty()
try:
    df = fetch_csv(sheet_url)
except Exception as e:
    err_box.error(
        "Could not load CSV. Ensure the link is public & ends with `output=csv`.\n\n"
        f"Error: {e}"
    )

if df is not None:
    try:
        title_col = detect_title_column(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    titles = clean_titles(df[title_col])
    if titles.empty:
        st.error("No non-empty titles found in the detected column.")
        st.stop()

    # ---------- DATASET OVERVIEW ----------
    st.markdown("### Dataset overview")
    col_a, col_b = st.columns(2)
    col_a.metric("Total projects loaded", len(titles))

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

    # Build indices once (cached)
    lex_vec, lex_mat = build_lexical_index(titles)
    sbert_embs = embed_titles_sbert(titles)

    # ===================== UI: QUERY & RESULTS =====================
    st.markdown("### Check a new capstone title")
    qcol1, qcol2 = st.columns([2, 1])
    with qcol1:
        query_title = st.text_input(
            "Enter a new capstone title to check:",
            placeholder="e.g., AI-based demand forecasting for retail supply chains",
        )
    with qcol2:
        top_k = st.number_input("Top matches", min_value=1, max_value=50, value=5, step=1)

    if st.button("Check Similarity", type="primary", use_container_width=True):
        results_df, info = find_similar_titles(
            query_title,
            titles,
            lex_vec,
            lex_mat,
            sbert_embs,
            top_k=int(top_k),
        )

        if "error" in info:
            st.warning(info["error"])
        else:
            st.markdown(f"**Summary:** {info['message']}")
            st.caption(
                f"Max lexical similarity in dataset: {info['max_lexical_sim']:.3f} "
                f"(candidates: {info['num_candidates']})"
            )

            if results_df.empty:
                st.info("No candidate titles to display.")
            else:
                st.subheader("Top candidate matches")
                st.dataframe(results_df, use_container_width=True)

                st.download_button(
                    "⬇️ Download results (CSV)",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="similarity_results.csv",
                    mime="text/csv",
                )

                with st.expander("How to interpret the scores"):
                    st.markdown(
                        """
                        - **Lexical**: word-level TF-IDF cosine similarity in [0,1].  
                          Higher = more overlap in actual words/phrases.
                        - **Semantic**: SBERT cosine similarity mapped from [-1,1] to [0,1].  
                          Captures paraphrasing / meaning.
                        - **Hybrid**: 0.7 × Semantic + 0.3 × Lexical.  
                          Used for ranking and for the **Strength** label.  

                        Rough guideline for **Hybrid** (and Hybrid %):

                        - < 0.30 (~0–30%) → **Weak** similarity  
                        - 0.30–0.60 (~30–60%) → **Moderate** topic overlap  
                        - 0.60–0.80 (~60–80%) → **Strong** overlap (investigate)  
                        - > 0.80 (>80%) → **Very strong**; likely duplicate or very close topic  

                        The tool is designed to **assist faculty** in spotting potential overlaps.  
                        Final decisions should always be made by supervisors using academic judgement.
                        """
                    )

st.markdown("---")
st.caption(
    "Tip: Good titles specify domain + method + context "
    "(e.g., 'AI-based demand forecasting for Canadian grocery retail')."
)
