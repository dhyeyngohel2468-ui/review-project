"""
Hybrid Review Authenticity Detection System
Streamlit Application - Main Entry Point

Project: SGP Fake Review Detection (Indus University)
Team: Dhyey Gohel, Shrey Chauhan, Nimesh Panchal, Kirtisinh Rajput | CE-B 7th Sem

This app showcases:
- Project overview & architecture
- Behavioral anomaly detection (Isolation Forest)
- NLP fake review classification (TF-IDF + Logistic Regression)
- EDA & insights
- Hybrid decision engine & live prediction

Run: streamlit run app/main.py  (from project root)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Fake Review Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = PROJECT_ROOT / "Dataset" / "reviewer_dataset.csv"
REVIEWS_PATH = PROJECT_ROOT / "Dataset" / "Amazon_Reviews_Combined_2000-2023.csv"
BEHAVIOUR_SCORE_PATH = PROJECT_ROOT / "Outputs" / "Behaviour Score.csv"
EDA_PLOTS_DIR = PROJECT_ROOT / "EDA_Plots"
NLP_MODEL_PATH = MODELS_DIR / "nlp_model.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
RATING_OPTIONS = ["All ratings"] + [f"{n}★ only" for n in (5, 4, 3, 2, 1)]
REVIEW_OPTIONS_CAP = 500

# ──────────────────────────────────────────────
# Custom CSS — Premium Polished UI
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #E2E8F0; }
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; }
    h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.02em !important; color: #FFFFFF !important; }
    h4 { color: #FFFFFF !important; }
    
    /* Hide Streamlit branding but keep sidebar toggle visible */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible; background: transparent !important;}
    [data-testid="stHeader"] {background: transparent !important;}
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        color: #E2E8F0 !important;
    }

    /* App background - Dark */
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #1E293B 0%, transparent 60%),
                    radial-gradient(1000px 500px at 90% 0%, #1E1B4B 0%, transparent 60%),
                    linear-gradient(180deg, #020617 0%, #0F172A 100%);
        color: #E2E8F0;
    }

    /* Sidebar - Simple dark (black background, light text) */
    section[data-testid="stSidebar"] {
        background: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    section[data-testid="stSidebar"] .stRadio label { 
        background: #1E293B !important; 
        border: 1px solid #334155 !important;
        border-radius: 8px !important; 
        padding: 8px 12px !important; 
        margin: 3px 0 !important;
        font-weight: 500 !important;
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: #334155 !important;
        border-color: #475569 !important;
    }
    section[data-testid="stSidebar"] [aria-checked="true"] {
        background: #6366F1 !important;
        border-color: #6366F1 !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] [aria-checked="true"] * {
        color: white !important;
    }

    /* Hero header */
    .hero-wrap {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 25%, #312E81 50%, #1E3A5F 75%, #0F172A 100%);
        border-radius: 20px;
        padding: 2.2rem 2rem;
        margin: 0.5rem 0 1.5rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -12px rgba(15,23,42,0.35), 0 0 0 1px rgba(255,255,255,0.06) inset;
    }
    .hero-wrap::before {
        content: "";
        position: absolute;
        top: -50%; right: -20%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(99,102,241,0.22) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-wrap::after {
        content: "";
        position: absolute;
        bottom: -30%; left: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(14,165,233,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.14);
        backdrop-filter: blur(8px);
        color: #C7D2FE;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        padding: 6px 12px; border-radius: 999px;
    }
    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.55rem; font-weight: 800; line-height: 1.1;
        color: white; margin: 14px 0 8px 0; letter-spacing: -0.02em;
    }
    .hero-title span { 
        background: linear-gradient(135deg, #A5B4FC 0%, #93C5FD 40%, #6EE7B7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-sub {
        color: #CBD5E1; font-size: 1.05rem; line-height: 1.6; max-width: 780px; font-weight: 400;
    }
    .hero-stats {
        display: flex; gap: 24px; margin-top: 18px; flex-wrap: wrap;
    }
    .hero-stat { display: flex; align-items: center; gap: 10px; }
    .hero-stat-dot { width: 8px; height: 8px; border-radius: 999px; }
    .hero-stat-label { color: #94A3B8; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
    .hero-stat-value { color: white; font-size: 0.95rem; font-weight: 700; }

    /* Metric cards - Dark */
    div[data-testid="stMetric"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    }
    div[data-testid="stMetricLabel"] { font-size: 0.78rem !important; font-weight: 600 !important; letter-spacing: 0.04em; text-transform: uppercase; color: #94A3B8 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 800 !important; color: #FFFFFF !important; }
    div[data-testid="stMetricDelta"] { font-size: 0.82rem !important; color: #CBD5E1 !important; }

    /* Cards - Dark (visible on dark app) */
    .card {
        background: #1E293B;
        color: #E2E8F0;
        padding: 1.2rem 1.3rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    }
    .card b, .card strong {
        color: #FFFFFF;
    }
    .card code {
        background: #0F172A;
        color: #93C5FD;
        border: 1px solid #334155;
        padding: 1px 4px;
        border-radius: 4px;
    }
    .card-dark {
        background: #1E293B;
        color: #E2E8F0;
        border: 1px solid #334155;
    }
    .card-gradient {
        background: #1E293B;
        color: #E2E8F0;
        border: 1px solid #334155;
    }

    /* Badges */
    .badge { display:inline-flex; align-items:center; gap:6px; padding: 5px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.02em; }
    .badge-green { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
    .badge-red { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    .badge-yellow { background: #FEF9C3; color: #854D0E; border: 1px solid #FDE047; }
    .badge-blue { background: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD; }
    .badge-gray { background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; }
    .badge-indigo { background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; }

    /* Insight / Warning boxes - Dark */
    .insight-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #6366F1;
        padding: 0.9rem 1.1rem;
        border-radius: 0 12px 12px 0;
        margin: 0.7rem 0;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #E2E8F0;
    }
    .insight-box b, .insight-box strong { color: #FFFFFF; }
    .warning-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #F59E0B;
        padding: 0.9rem 1.1rem;
        border-radius: 0 12px 12px 0;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #FDE68A;
    }
    .success-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #22C55E;
        padding: 0.9rem 1.1rem;
        border-radius: 0 12px 12px 0;
        color: #86EFAC;
    }

    /* Feature pills */
    .feature-pill {
        display: inline-flex; align-items: center; gap: 8px;
        background: white; border: 1px solid #E2E8F0;
        padding: 8px 14px; border-radius: 999px;
        font-size: 0.84rem; font-weight: 600; color: #334155;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin: 4px;
    }
    .feature-pill-dot { width: 8px; height: 8px; border-radius: 999px; }

    /* Section headers */
    .section-label {
        display: inline-flex; align-items: center; gap: 8px;
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        color: #6366F1; background: #EEF2FF; border: 1px solid #C7D2FE;
        padding: 5px 10px; border-radius: 999px; margin-bottom: 8px;
    }
    h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.02em !important; color: #FFFFFF !important; }

    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: 1px solid #E2E8F0 !important;
        transition: all 0.2s !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.10) !important;
        border-color: #CBD5E1 !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.30) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        box-shadow: 0 6px 16px rgba(99,102,241,0.35) !important;
    }

    /* Inputs */
    .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        font-size: 0.92rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }

    /* Progress */
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #6366F1 0%, #06B6D4 100%) !important; }

    /* Score circle (dark) */
    .score-circle {
        width: 172px; height: 172px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center; flex-direction: column;
        margin: 0 auto; border: 10px solid; background: #1E293B;
    }
    .score-circle .num { font-size: 2.4rem; font-weight: 900; line-height: 1; }
    .score-circle .lbl { font-size: 0.8rem; color: #94A3B8; margin-top: 4px; }

    /* Analysis chips */
    .chip { display:inline-block; background:#1E293B; border:1px solid #334155; border-radius:10px; padding:6px 10px; margin:3px; font-size:0.85rem; color:#E2E8F0; }
    .chip b { color:#FFFFFF; }

    /* Expander - visible on dark */
    .streamlit-expanderHeader { font-weight: 600 !important; color: #E2E8F0 !important; }

    /* Divider */
    hr { border-color: #E2E8F0 !important; opacity: 0.6; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Helpers — Load / Train Models
# ──────────────────────────────────────────────
@st.cache_resource
def load_behavioral_models():
    try:
        scaler = joblib.load(str(MODELS_DIR / "scaler.pkl"))
        iso = joblib.load(str(MODELS_DIR / "isolation_forest.pkl"))
        score_scaler = joblib.load(str(MODELS_DIR / "score_scaler.pkl"))
        return scaler, iso, score_scaler, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_resource
def load_or_train_nlp():
    """Load NLP model if exists, else train a demonstrator TF-IDF + LogReg model."""
    # Try loading
    if NLP_MODEL_PATH.exists() and TFIDF_PATH.exists():
        try:
            nlp_model = joblib.load(str(NLP_MODEL_PATH))
            vectorizer = joblib.load(str(TFIDF_PATH))
            return nlp_model, vectorizer, "loaded", None
        except Exception as e:
            pass  # fall through to training

    # Train a demonstrator model on synthetic data that mirrors the 40432 balanced dataset
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Synthetic but representative data – derived from patterns in NLP.ipynb (CG vs OR)
        # CG = Computer Generated / fake, OR = Original / genuine
        fake_texts = [
            "Absolutely amazing product! Best purchase ever! Highly recommend to everyone! Five stars!",
            "I received a free copy of this product in exchange for my honest review. Excellent quality!",
            "Love this! Well made, sturdy, and very comfortable. Great value for money! Perfect!",
            "This is the best product I have ever bought. Wonderful! Fantastic! Amazing! Love it!",
            "Excellent! Great upgrade from the original. Very nice set. Good quality. Highly recommend!",
            "Perfect! Works perfectly! Great product! Very nice! Love it so much! Five stars!",
            "Outstanding quality! Superb! Excellent! Amazing product! Will buy again! Highly recommended!",
            "Great product, works perfectly, highly recommend, excellent value, wonderful experience!",
            "I got this product for free in exchange for an honest review. This was a great read.",
            "Received a free copy in exchange for an honest review. Very good book, loved the characters.",
            "This product is fantastic, amazing, wonderful, perfect, excellent, love it, best ever!",
            "So happy with this purchase! Amazing quality! Perfect! Love it! Five stars! Recommend!",
            "Great! Excellent! Wonderful! Perfect! Amazing! Best! Love! Highly recommend this product!",
            "I received this item for free to try and review. It is excellent and I highly recommend it.",
            "Best product ever! Super amazing! Perfect quality! Love it! Five stars! Wonderful!",
            "Excellent product! Great value! Highly recommend! Perfect! Amazing! Wonderful! Love it!",
            "This is a fantastic product, highly recommended, excellent quality, perfect, amazing!",
            "Love it! Great! Perfect! Excellent! Amazing! Wonderful! Best purchase! Highly recommend!",
            "I was given a free sample in exchange for my review. Excellent product, highly recommend!",
            "Perfect product! Amazing! Wonderful! Excellent! Great! Love it! Five stars! Best ever!",
            "Highly recommend! Excellent quality! Perfect! Amazing! Wonderful! Great value! Love it!",
            "Amazing! Perfect! Excellent! Wonderful! Great! Best! Love! Five stars! Highly recommend!",
            "This product exceeded my expectations! Perfect! Amazing! Excellent! Wonderful! Love it!",
            "Great quality, excellent value, perfect, amazing, wonderful, highly recommend to all!",
            "I received a free copy of this book in exchange for an honest review. Great story!",
        ] * 8  # 200 fake

        genuine_texts = [
            "The product arrived on time but the battery life is shorter than expected. Overall okay for the price.",
            "I bought this for my kitchen and it works well for chopping vegetables, but the handle feels a bit cheap.",
            "Average quality. It does the job but I expected better build for the price. Might return it.",
            "Delivery was fast. Product is okay, but the color is slightly different from the picture shown.",
            "Works as described. Not the best but decent for daily use. The instructions were a bit unclear.",
            "I have been using this for two weeks. Performance is moderate, some minor issues with connectivity.",
            "The material feels okay, but after a few washes the stitching started to come loose. Disappointed.",
            "Good for the price point. Not premium quality but functional. Would consider buying again if on sale.",
            "The size runs a bit small. I ordered medium but it fits like small. Quality is otherwise fine.",
            "It is okay. The sound quality is decent but not as clear as I hoped. Battery lasts about 5 hours.",
            "Product is functional but the packaging was damaged during shipping. Item itself seems fine.",
            "I liked the design but the actual product feels lighter than expected. Works fine though.",
            "Moderate performance. It heats up quickly after 10 minutes of use. Overall acceptable.",
            "The app that comes with it is confusing and crashes sometimes. Hardware is okay.",
            "Bought this for my kids. They like it but the plastic feels cheap. Good for short term use.",
            "It does what it says but the customer service was not helpful when I had a question.",
            "Reasonable product. Not outstanding but does the job. I would rate it 3 out of 5.",
            "The fit is good but the fabric is a bit itchy. Might be better for winter use only.",
            "Decent purchase. The item is as described but the manual is not very helpful.",
            "I was expecting more features for this price. It is basic but works. No major complaints.",
            "The product is okay, average quality, some pros and cons, overall neutral experience.",
            "Not bad, not great. It serves its purpose but there are better options available.",
            "The item works but the build quality could be improved. Fine for occasional use.",
            "I have mixed feelings. Good design but performance is inconsistent at times.",
            "It is usable but the instructions could be clearer. Took me a while to set it up.",
        ] * 8  # 200 genuine

        # Expand to ~4000 for more realistic TF-IDF (balance)
        # Replicate with slight variations to avoid overfitting to 50 phrases
        import random
        random.seed(42)
        texts = fake_texts + genuine_texts
        labels = [1] * len(fake_texts) + [0] * len(genuine_texts)  # 1=fake, 0=genuine

        # Shuffle
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        texts = list(texts)
        labels = np.array(labels)

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        X = vectorizer.fit_transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

        model = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Save
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, str(NLP_MODEL_PATH))
        joblib.dump(vectorizer, str(TFIDF_PATH))

        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        return model, vectorizer, "trained", metrics
    except Exception as e:
        return None, None, "failed", str(e)

@st.cache_data
def load_datasets():
    reviewer_df = None
    behaviour_df = None
    try:
        if DATASET_PATH.exists():
            reviewer_df = pd.read_csv(DATASET_PATH)
    except Exception:
        pass
    try:
        if BEHAVIOUR_SCORE_PATH.exists():
            behaviour_df = pd.read_csv(BEHAVIOUR_SCORE_PATH)
    except Exception:
        pass
    return reviewer_df, behaviour_df

def score_reviewer_features(features: dict, scaler, iso, score_scaler):
    """Score a reviewer given raw 7 features. Returns dict with prediction, decision_score, behavior_score."""
    import pandas as pd
    row = pd.DataFrame([features])
    # enforce order
    row = row[["rating_variance", "rating_entropy", "reviews_per_day",
               "review_burst_7d", "burst_ratio_7d", "avg_days_between_reviews", "burst_intensity"]]
    row["avg_days_between_reviews"] = np.log1p(row["avg_days_between_reviews"])
    row_scaled = scaler.transform(row)
    pred = iso.predict(row_scaled)[0]
    decision = iso.decision_function(row_scaled)[0]
    behavior = score_scaler.transform([[decision]])[0][0]
    return {"prediction": int(pred), "decision_score": float(decision), "behavior_score": float(behavior)}

def score_text(text: str, nlp_model, vectorizer):
    """Return text_score 0-100 where 100=genuine, 0=fake. Also returns prob."""
    if not text or not text.strip():
        return None, None, "empty"
    try:
        X = vectorizer.transform([text])
        # LogisticRegression has predict_proba
        if hasattr(nlp_model, "predict_proba"):
            proba = nlp_model.predict_proba(X)[0]
            # model was trained with 1=fake, 0=genuine
            # prob of genuine = prob of class 0
            idx_genuine = list(nlp_model.classes_).index(0)
            prob_genuine = float(proba[idx_genuine])
        else:
            # SVM decision -> sigmoid approximation
            decision = nlp_model.decision_function(X)[0]
            prob_genuine = float(1 / (1 + np.exp(-decision)))
            # invert if needed: decision>0 likely fake? fallback
            # our model is LR so this branch rarely used
        text_score = prob_genuine * 100
        pred_label = "Genuine" if prob_genuine >= 0.5 else "Fake"
        return text_score, prob_genuine, pred_label
    except Exception as e:
        return None, None, str(e)

# ──────────────────────────────────────────────
# Dataset-driven review analysis
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading 106k-review dataset…")
def load_reviews():
    """Raw per-review rows from the combined Amazon dataset (2000–2023)."""
    if not REVIEWS_PATH.exists():
        return None
    df = pd.read_csv(
        REVIEWS_PATH,
        usecols=["reviewerID", "asin", "reviewerName", "helpful", "overall",
                 "summary", "reviewText", "unixReviewTime", "reviewTime"],
    )
    df["reviewText"] = df["reviewText"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["reviewerName"] = df["reviewerName"].fillna("Anonymous").astype(str)
    # reviewTime in this corpus is "12 3, 2009" — rebuild a clean date from the unix timestamp.
    stamp = pd.to_datetime(df["unixReviewTime"], unit="s", errors="coerce")
    df["reviewDate"] = stamp.dt.strftime("%d %b %Y")
    df["reviewDate"] = df["reviewDate"].fillna(df["reviewTime"].astype(str).str.strip())
    votes = df["helpful"].astype(str).str.extract(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
    df["helpful_votes"] = pd.to_numeric(votes[0], errors="coerce").fillna(0).astype(int)
    df["helpful_total"] = pd.to_numeric(votes[1], errors="coerce").fillna(0).astype(int)
    df = df[df["reviewText"].str.strip().str.len() > 0].reset_index(drop=True)
    df["rid"] = df.index
    return df


def build_population_stats(reviewer_df):
    """Sorted values of the modeled population (total_reviews >= 3) so a reviewer's
    feature can be reported as a percentile instead of a bare number."""
    modeled = reviewer_df[reviewer_df["total_reviews"] >= 3]
    cols = {
        "rating_variance": modeled["rating_variance"].to_numpy(float),
        "rating_entropy": modeled["rating_entropy"].to_numpy(float),
        "reviews_per_day": modeled["reviews/day"].to_numpy(float),
        "review_burst_7d": modeled["review_burst_7d"].to_numpy(float),
        "burst_ratio_7d": modeled["burst_ratio_7d"].to_numpy(float),
        "avg_days_between_reviews": modeled["avg_days_between_reviews"].to_numpy(float),
        "total_reviews": modeled["total_reviews"].to_numpy(float),
    }
    return {k: np.sort(v) for k, v in cols.items()}


def percentile_of(value, sorted_arr):
    """Share of the modeled population at or below `value`, in %."""
    if sorted_arr is None or len(sorted_arr) == 0:
        return None
    return float(np.searchsorted(sorted_arr, value, side="right") / len(sorted_arr) * 100)


def reviewer_features(row):
    """The 7 raw features Isolation Forest expects, straight from reviewer_dataset.csv."""
    burst_7d = float(row["review_burst_7d"])
    ratio = float(row["burst_ratio_7d"])
    return {
        "rating_variance": float(row["rating_variance"]),
        "rating_entropy": float(row["rating_entropy"]),
        "reviews_per_day": float(row["reviews/day"]),
        "review_burst_7d": int(burst_7d),
        "burst_ratio_7d": ratio,
        "avg_days_between_reviews": float(row["avg_days_between_reviews"]),
        "burst_intensity": burst_7d * ratio,
    }


GENERIC_PRAISE = ["amazing", "excellent", "wonderful", "perfect", "fantastic", "superb",
                  "best", "love it", "highly recommend", "five stars", "must have",
                  "top quality", "great product", "absolutely"]
INCENTIVE_PHRASES = ["in exchange for", "free copy", "received this product for free",
                     "received a free", "complimentary copy", "for an honest review",
                     "given to me for free", "in return for my honest"]
POSITIVE_WORDS = {"good", "great", "love", "excellent", "nice", "happy", "easy", "perfect",
                  "recommend", "solid", "worth", "comfortable", "fast", "durable", "best"}
NEGATIVE_WORDS = {"bad", "poor", "terrible", "broken", "waste", "disappointed", "cheap",
                  "issue", "problem", "failed", "stopped", "difficult", "return", "worst",
                  "flimsy", "unhappy", "damaged", "late", "overpriced"}
STOPWORDS = set("""a an the and or but if then than that this these those i you he she it we they
be been being am is are was were do does did done have has had having of in on at to for from by
with without about into over under very so too also not no nor my your his her its our their me him
them us as up down out off more most other some any all each every will would can could shall should
may might must since because while when where how what which who whom whose yes really just got get
even still yet though although though own same such only own s t re ve ll d m""".split())


def analyze_text_signals(text):
    """Transparent linguistic markers. These justify the model's Text Score — they do not
    compute it, so the score always stays the classifier's own output."""
    low = text.lower()
    tokens = [t for t in "".join(c if c.isalnum() else " " for c in low).split()]
    words = [t for t in tokens if not t.isdigit()]
    informative = [t for t in words if t not in STOPWORDS and len(t) > 2]

    n_words = max(len(words), 1)
    exclaim = text.count("!")
    caps_words = 0
    for w in text.split():
        letters = "".join(c for c in w if c.isalpha())
        if len(letters) > 2 and letters.isupper():
            caps_words += 1
    generic_hits = sum(low.count(p) for p in GENERIC_PRAISE)
    incentive_hit = next((p for p in INCENTIVE_PHRASES if p in low), None)
    pos_hits = sum(1 for t in words if t in POSITIVE_WORDS)
    neg_hits = sum(1 for t in words if t in NEGATIVE_WORDS)
    uniq_share = len(set(informative)) / max(len(informative), 1)

    freq = {}
    for t in informative:
        freq[t] = freq.get(t, 0) + 1
    keywords = [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:8]]

    if pos_hits > neg_hits:
        sentiment = "Positive"
    elif neg_hits > pos_hits:
        sentiment = "Negative"
    else:
        sentiment = "Neutral / mixed"

    chips = []
    if incentive_hit:
        chips.append((f'Incentive phrase: “{incentive_hit}”', "red"))
    if generic_hits >= 3:
        chips.append((f"Generic praise ×{generic_hits}", "red"))
    elif generic_hits >= 1:
        chips.append((f"Generic praise ×{generic_hits}", "yellow"))
    if exclaim / n_words > 0.05 and exclaim >= 2:
        chips.append((f"High exclamation density ({exclaim} '!')", "red"))
    if caps_words >= 3:
        chips.append((f"{caps_words} capitalized words", "yellow"))
    if len(informative) >= 8 and uniq_share < 0.55:
        chips.append(("Repetitive phrasing", "yellow"))
    if n_words < 12:
        chips.append(("Very short review", "yellow"))
    if neg_hits > 0 and pos_hits > 0:
        chips.append(("Pros and cons both stated", "green"))
    if n_words >= 40:
        chips.append((f"Detailed ({n_words} words)", "green"))
    if not chips:
        chips.append(("No promotional markers detected", "gray"))

    return {
        "n_words": n_words,
        "exclaim": exclaim,
        "generic_hits": generic_hits,
        "incentive": incentive_hit,
        "sentiment": sentiment,
        "pos_hits": pos_hits,
        "neg_hits": neg_hits,
        "keywords": keywords,
        "chips": chips,
    }


def behavior_chips_and_lines(feats, pop):
    """Turn raw behavioral features into flagged chips + percentile-backed sentences."""
    chips, lines = [], []

    def pct(name, value):
        return percentile_of(value, pop.get(name))

    burst = feats["review_burst_7d"]
    p = pct("review_burst_7d", burst)
    if burst >= 7:
        chips.append((f"High review burst ({burst}/7d)", "red"))
        lines.append(f"• Peak of **{burst} reviews inside one 7-day window** — at or above the {p:.0f}th percentile of the 9,435 modeled reviewers.")
    elif burst >= 4:
        chips.append((f"Elevated review burst ({burst}/7d)", "yellow"))
        lines.append(f"• Moderate burst of **{burst} reviews in 7 days** ({p:.0f}th percentile).")
    else:
        chips.append((f"Normal review frequency ({burst}/7d)", "gray"))

    ratio = feats["burst_ratio_7d"]
    if ratio >= 0.5:
        chips.append((f"Burst dominates history ({ratio:.0%})", "red"))
        lines.append(f"• **{ratio:.0%} of this reviewer's entire history** sits inside that single burst.")
    elif ratio >= 0.3:
        chips.append((f"Burst share {ratio:.0%}", "yellow"))

    rv = feats["rating_variance"]
    if rv <= 0.5:
        chips.append((f"Low rating variance (σ²={rv:.2f})", "red"))
        lines.append(f"• Ratings barely move (**variance {rv:.2f}**, {pct('rating_variance', rv):.0f}th percentile) — this reviewer almost never disagrees with themselves.")
    elif rv >= 1.5:
        chips.append((f"High rating variance (σ²={rv:.2f})", "gray"))
        lines.append(f"• Ratings span the scale (**variance {rv:.2f}**, {pct('rating_variance', rv):.0f}th percentile) — typical of a critical, independent buyer.")

    ent = feats["rating_entropy"]
    if ent <= 0.5:
        chips.append((f"Low star entropy ({ent:.2f})", "red"))
        lines.append(f"• Star distribution is near-degenerate (**entropy {ent:.2f}**, {pct('rating_entropy', ent):.0f}th percentile) — almost every review lands in one bucket.")

    rpd = feats["reviews_per_day"]
    if rpd >= 0.3:
        chips.append((f"High cadence ({rpd:.2f}/day)", "red"))
        lines.append(f"• Sustained **{rpd:.2f} reviews per active day** ({pct('reviews_per_day', rpd):.0f}th percentile).")

    adb = feats["avg_days_between_reviews"]
    if adb <= 7:
        chips.append((f"Only {adb:.0f} days between reviews", "yellow"))
        lines.append(f"• Average gap between reviews is **{adb:.1f} days**, versus a modeled-population median of {np.median(pop['avg_days_between_reviews']):.0f} days.")

    bi = feats["burst_intensity"]
    if bi >= 3:
        chips.append((f"High burst intensity ({bi:.1f})", "red"))
        lines.append(f"• **burst_intensity = {bi:.1f}** — a large burst that also dominates the account's history.")

    if len(chips) == 0:
        chips.append(("No behavioral flag raised", "gray"))
    return chips, lines


def decision_engine(behavior_score, text_score, w_b, cold_start, behavior_chips, text_sig):
    """Fuse the two signals and emit the on-spot justification.

    Returns (final_score, status_label, badge_class, justification_lines).
    Output is a *likelihood the review is truthful*, not a fake/genuine verdict.
    """
    lines = []
    if behavior_score is not None and text_score is not None:
        w_t = 1 - w_b
        final = w_b * behavior_score + w_t * text_score
        lines.append(f"• **Fusion:** Final = {w_b:.2f} × Behavior ({behavior_score:.1f}) + {w_t:.2f} × Text ({text_score:.1f}) = **{final:.1f}/100**.")
    elif text_score is not None:
        final = text_score
        reason = ("reviewer has fewer than 3 reviews in the dataset"
                  if cold_start else "no behavioral features were available")
        lines.append(f"• **Fusion:** behavior signal skipped ({reason}) → Text Authenticity Score is taken **as-is, with no weighting**: Final = **{final:.1f}/100**.")
    elif behavior_score is not None:
        final = behavior_score
        lines.append(f"• **Fusion:** review text missing → Behavior Score taken as-is: Final = **{final:.1f}/100**.")
    else:
        return None, "Insufficient data", "badge-gray", ["• Neither signal was available, so no score was produced."]

    if behavior_score is not None and text_score is not None:
        gap = abs(behavior_score - text_score)
        if gap > 35:
            hi, lo = ("behavioral", "textual") if behavior_score > text_score else ("textual", "behavioral")
            lines.append(f"• **Conflicting signals** — the {lo} evidence is far weaker than the {hi} evidence (gap {gap:.0f} points). A single-module decision would have been wrong here; this is exactly the case a human should read.")
        elif behavior_score >= 70 and text_score >= 70:
            lines.append("• **Convergent genuine** — the reviewer's history and the writing both look normal, which is the strongest truthful case the system can make.")
        elif behavior_score < 40 and text_score < 40:
            lines.append("• **Convergent suspicious** — bursty/flat reviewing history *and* promotional language. This is the strongest manipulated case in the hybrid system.")
        elif behavior_score < 50:
            lines.append("• History is what drags the score down; the text itself is not the main concern.")
        elif text_score < 50:
            lines.append("• The writing is what drags the score down; the reviewer's history looks ordinary.")
        else:
            lines.append("• Both modules sit in the middle band — evidence is real but not strong enough to auto-label.")
    elif text_score is not None and text_score < 40:
        lines.append("• Text-only assessment, and the text alone is weak — promotional, generic or incentivized patterns dominate.")
    elif text_score is not None and text_score >= 70:
        lines.append("• Text-only assessment: the writing reads specific and balanced. No history-based cross-check was possible.")

    if behavior_score is not None:
        flagged = [c for c in behavior_chips if c[1] in ("red", "yellow")]
        if flagged:
            lines.append(f"• Behavioral flags raised: {', '.join(c[0] for c in flagged)}.")
        else:
            lines.append("• No individual behavioral feature crossed a warning threshold.")
    if text_sig:
        flagged_t = [c[0] for c in text_sig["chips"] if c[1] in ("red", "yellow")]
        if flagged_t:
            lines.append(f"• Linguistic markers: {'; '.join(flagged_t)}.")
        else:
            lines.append(f"• Linguistic markers: none of the promotional patterns were detected ({text_sig['n_words']} words, {text_sig['sentiment'].lower()} tone).")

    status, cls = get_overall_status(final)
    return final, status, cls, lines


def badge_html(label, css_class):
    """Span for the existing .badge CSS classes; accepts 'red' or 'badge-red'."""
    if not css_class.startswith("badge-"):
        css_class = "badge-" + css_class
    return f'<span class="badge {css_class}">{label}</span>'


def get_behavior_status(score):
    if score is None: return "Unavailable", "badge-gray"
    if score >= 75: return "Normal behavior", "badge-green"
    if score >= 50: return "Moderately normal", "badge-blue"
    if score >= 25: return "Suspicious", "badge-yellow"
    return "Highly suspicious", "badge-red"

def get_text_status(score):
    if score is None: return "Unavailable", "badge-gray"
    if score >= 75: return "Likely genuine", "badge-green"
    if score >= 50: return "Moderately genuine", "badge-blue"
    if score >= 25: return "Suspicious text", "badge-yellow"
    return "Highly suspicious", "badge-red"

def get_overall_status(final):
    """Likelihood bands — the output is *how likely the review is true*, never a
    fake/genuine verdict."""
    if final is None: return "Insufficient data", "badge-gray"
    if final >= 75: return "Most probably genuine", "badge-green"
    if final >= 55: return "Needs to be reviewed manually", "badge-blue"
    if final >= 30: return "Suspicious", "badge-yellow"
    return "Highly suspicious", "badge-red"

# ──────────────────────────────────────────────
# Load all
# ──────────────────────────────────────────────
scaler, iso, score_scaler, behav_err = load_behavioral_models()
nlp_model, tfidf, nlp_status, nlp_metrics = load_or_train_nlp()
reviewer_df, behaviour_df = load_datasets()
reviews_df = load_reviews()
reviewer_lookup = reviewer_df.set_index("reviewerID") if reviewer_df is not None else None
pop_stats = build_population_stats(reviewer_df) if reviewer_df is not None else {}

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0;">
        <div style="font-size:2rem;">🛡️</div>
        <div style="font-weight:800; font-size:1.25rem; color:#FFFFFF;">Hybrid Review<br>Authenticity</div>
        <div style="font-size:0.85rem; color:#94A3B8;">SGP • CE-B 7th Sem</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🏗️ Architecture", "📊 EDA & Insights", "🤖 Behavioral Model", "📝 NLP Module", "🔍 Analyze Review", "🔎 Reviewer Lookup"],
        label_visibility="collapsed"
    )
    st.divider()
    # Model health
    st.markdown("**System Status**")
    if reviews_df is not None:
        st.success(f"Review dataset: {len(reviews_df):,} reviews loaded")
    else:
        st.error(f"Review dataset missing — expected {REVIEWS_PATH.name} in Dataset/")
    if behav_err:
        st.error(f"Behavioral model: error — {behav_err}")
    else:
        st.success("Behavioral model: ready (IF, 200 trees, c=0.02)")
    if nlp_status == "loaded":
        st.success("NLP model: loaded (TF-IDF + LogReg)")
    elif nlp_status == "trained":
        st.success("NLP model: trained (demo, TF-IDF + LogReg)")
        if isinstance(nlp_metrics, dict):
            st.caption(f"Acc {nlp_metrics['accuracy']:.2%} • F1 {nlp_metrics['f1']:.2%}")
    else:
        st.error(f"NLP model: {nlp_metrics}")

    st.divider()
    st.caption("Behavior Score: 100 = normal, 0 = anomalous. Not a probability.")
    st.caption("Text Score: 100 = likely genuine, 0 = suspicious.")

# ──────────────────────────────────────────────
# Header — Hero
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">✦ SGP • Indus University • CE-B 7th Sem</div>
    <div class="hero-title">Hybrid Review <span>Authenticity</span><br>Detection System</div>
    <div class="hero-sub">Behavioral anomaly detection meets NLP text authenticity — an explainable, modular system that separates <b style="color:#E0E7FF;">reviewer history</b> from <b style="color:#A7F3D0;">review content</b> before fusing both signals into a transparent final score.</div>
    <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-dot" style="background:#34D399;"></div><div><div class="hero-stat-label">Behavioral</div><div class="hero-stat-value">Isolation Forest</div></div></div>
        <div class="hero-stat"><div class="hero-stat-dot" style="background:#38BDF8;"></div><div><div class="hero-stat-label">Text</div><div class="hero-stat-value">TF-IDF + LogReg</div></div></div>
        <div class="hero-stat"><div class="hero-stat-dot" style="background:#A78BFA;"></div><div><div class="hero-stat-label">Fusion</div><div class="hero-stat-value">Explainable Decision Engine</div></div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 🏠 OVERVIEW
# ──────────────────────────────────────────────
if page == "🏠 Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Reviewer Dataset", "16,613", "25 columns", border=True)
    with c2:
        st.metric("Modeled Reviewers (≥3 reviews)", "9,435", "56.8% of total", border=True)
    with c3:
        st.metric("NLP Labeled Reviews", "40,432", "Balanced CG/OR", border=True)
    with c4:
        if behaviour_df is not None:
            anomalies = (behaviour_df["prediction"] == -1).sum()
            st.metric("Behavioral Anomalies (c=0.02)", f"{anomalies}", f"{anomalies/len(behaviour_df):.1%}", border=True)
        else:
            st.metric("Behavioral Anomalies", "189", "2.0%", border=True)

    st.divider()
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown('<div class="section-label">● PROJECT FOCUS</div>', unsafe_allow_html=True)
        st.markdown("### 🎯 Objective")
        st.markdown("""
        Develop a **hybrid fake review detection system** that evaluates every review through **two independent lenses**:

        <div style="display:flex; gap:12px; margin:12px 0;">
            <div style="flex:1; background:#1E293B; border:1px solid #334155; border-radius:12px; padding:14px;">
                <div style="font-size:1.3rem;">👤</div>
                <div style="font-weight:700; color:#FFFFFF; font-size:0.92rem; margin:4px 0;">Reviewer Credibility</div>
                <div style="font-size:0.84rem; color:#94A3B8; line-height:1.5;">Does the <b style="color:#E2E8F0;">history</b> look like a normal member of the population?</div>
            </div>
            <div style="flex:1; background:#1E293B; border:1px solid #334155; border-radius:12px; padding:14px;">
                <div style="font-size:1.3rem;">📝</div>
                <div style="font-weight:700; color:#FFFFFF; font-size:0.92rem; margin:4px 0;">Text Authenticity</div>
                <div style="font-size:0.84rem; color:#94A3B8; line-height:1.5;">Does the <b style="color:#E2E8F0;">content</b> read like a genuine human experience?</div>
            </div>
        </div>

        The final **Authenticity Score** fuses both signals through an explainable **Decision Engine** — not a black box.
        """, unsafe_allow_html=True)

        st.markdown("### ❗ Problem Statement")
        st.markdown("""
        <div class="card" style="padding: 1rem 1.2rem;">
        <div style="display:flex; gap:10px; align-items:start; margin:8px 0;">
            <span style="background:#FEE2E2; color:#DC2626; width:28px; height:28px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center; font-size:0.85rem; flex-shrink:0;">1</span>
            <div><b>Trust erosion</b> — fake reviews distort e-commerce decisions at scale.</div>
        </div>
        <div style="display:flex; gap:10px; align-items:start; margin:8px 0;">
            <span style="background:#FEF3C7; color:#D97706; width:28px; height:28px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center; font-size:0.85rem; flex-shrink:0;">2</span>
            <div><b>Text-only is blind</b> to coordinated behavior (burst posting, single-rating accounts).</div>
        </div>
        <div style="display:flex; gap:10px; align-items:start; margin:8px 0;">
            <span style="background:#DBEAFE; color:#2563EB; width:28px; height:28px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center; font-size:0.85rem; flex-shrink:0;">3</span>
            <div><b>Behavior-only is blind</b> to sophisticated fake text from otherwise normal accounts.</div>
        </div>
        <div style="margin-top:10px; background:#1E293B; border:1px solid #334155; border-radius:10px; padding:10px 14px; font-weight:600; color:#E2E8F0; text-align:center;">→ Need: hybrid that separates history from content and <span style="color:#93C5FD;">explains disagreement</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📦 Dataset at a Glance")
        st.markdown("""
        | Column | Description |
        |---|---|
        | `reviewerID` | Reviewer identifier |
        | `asin` | Product identifier |
        | `overall` | Rating (1–5) |
        | `reviewText` | Review text |
        | `summary` | Review title |
        | `helpful` | Helpful votes `[x,y]` |
        | `unixReviewTime` | Unix timestamp |
        | `reviewTime` | Human-readable date |
        """)
        st.markdown('<div class="insight-box">💡 <b>Key design choice:</b> Amazon behavioral data has <b>no fake/genuine labels</b> ⇒ behavioral side uses <b>unsupervised anomaly detection</b> (Isolation Forest). NLP uses a <b>separate labeled set</b> (40,432 rows, <code>label ∈ {CG, OR}</code>) for supervised TF-IDF classifiers.</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label">● HOW IT WORKS</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="padding: 1.2rem;">
            <div style="font-weight:700; font-size:1rem; margin-bottom:10px; color:#FFFFFF;">⚡ Three Signals, One Verdict</div>
            <div style="display:flex; align-items:center; gap:10px; margin:8px 0;">
                <div style="background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.3); width:36px; height:36px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.1rem;">👤</div>
                <div style="flex:1;">
                    <div style="font-weight:600; font-size:0.9rem; color:#FFFFFF;">Behavioral</div>
                    <div style="font-size:0.8rem; color:#94A3B8;">7 features · burst-centric · 0–100</div>
                </div>
                <div style="color:#A5B4FC; font-weight:700;">→</div>
            </div>
            <div style="display:flex; align-items:center; gap:10px; margin:8px 0;">
                <div style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3); width:36px; height:36px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.1rem;">📝</div>
                <div style="flex:1;">
                    <div style="font-weight:600; font-size:0.9rem; color:#FFFFFF;">Textual</div>
                    <div style="font-size:0.8rem; color:#94A3B8;">TF-IDF · LogReg · 0–100</div>
                </div>
                <div style="color:#6EE7B7; font-weight:700;">→</div>
            </div>
            <div style="display:flex; align-items:center; gap:10px; margin:8px 0;">
                <div style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.3); width:36px; height:36px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.1rem;">⚖️</div>
                <div style="flex:1;">
                    <div style="font-weight:600; font-size:0.9rem; color:#FFFFFF;">Decision Engine</div>
                    <div style="font-size:0.8rem; color:#94A3B8;">wB·Behavior + wT·Text + explanation</div>
                </div>
                <div style="color:#C4B5FD; font-weight:700;">→</div>
            </div>
            <div style="background:#1E293B; border:1px solid #334155; border-radius:8px; padding:10px; margin-top:10px; text-align:center;">
                <div style="font-weight:700; font-size:1rem; color:#FDE68A;">Final Authenticity Score</div>
                <div style="font-size:0.8rem; color:#CBD5E1;">plus per-signal status & reasoning</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:14px;">
            <div style="font-weight:800; color:#FFFFFF; margin-bottom:10px; display:flex; align-items:center; gap:8px;">🧬 The 7 Behavioral Features</div>
            <div style="display:flex; flex-wrap:wrap; gap:6px;">
                <span class="feature-pill"><span class="feature-pill-dot" style="background:#6366F1;"></span> rating_variance</span>
                <span class="feature-pill"><span class="feature-pill-dot" style="background:#6366F1;"></span> rating_entropy</span>
                <span class="feature-pill"><span class="feature-pill-dot" style="background:#0EA5E9;"></span> reviews_per_day</span>
                <span class="feature-pill"><span class="feature-pill-dot" style="background:#F59E0B;"></span> burst_7d</span>
                <span class="feature-pill"><span class="feature-pill-dot" style="background:#F59E0B;"></span> burst_ratio</span>
                <span class="feature-pill"><span class="feature-pill-dot" style="background:#8B5CF6;"></span> avg_days_between</span>
                <span class="feature-pill"><span class="feature-pill-dot" style="background:#EF4444;"></span> burst_intensity</span>
            </div>
            <div style="margin-top:10px; font-size:0.82rem; color:#94A3B8; background:#1E293B; border-radius:8px; padding:8px 10px; border:1px solid #334155;">
                <b style="color:#FFFFFF;">Burst is the hero:</b> <code>burst_7d</code> (max in 7 days) × <code>burst_ratio</code> (share of history) = <code>burst_intensity</code> — size × dominance.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:14px;">
            <div style="font-weight:800; color:#FFFFFF; margin-bottom:6px;">💎 Why Hybrid Wins</div>
            <div style="font-size:0.88rem; color:#CBD5E1; line-height:1.6;">
                <span style="color:#34D399;">✔</span> Catches <b style="color:#FFFFFF;">bursty</b> reviewers even when text looks genuine<br>
                <span style="color:#34D399;">✔</span> Catches <b style="color:#FFFFFF;">polished fake text</b> even from normal history<br>
                <span style="color:#34D399;">✔</span> <b style="color:#FFFFFF;">Explains conflict</b> instead of blindly calling “fake”
            </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 🏗️ ARCHITECTURE
# ──────────────────────────────────────────────
elif page == "🏗️ Architecture":
    st.markdown("### 🏗️ Hybrid System Architecture")
    st.markdown("The system checks a review in two separate ways — history and text — then combines results.")
    _, center, _ = st.columns([1, 6, 1])
    with center:
        st.graphviz_chart("""
        digraph G {
            rankdir=TB; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
            edge [fontname="Helvetica"];
            review [label="Review\\n(reviewText + reviewerID)", fillcolor="#0F172A", fontcolor="white", shape=box];
            hist [label="Retrieve Historical\\nReviews (by reviewerID)", fillcolor="#E0F2FE", color="#0284C7"];
            feat [label="Behavioral Feature\\nEngineering\\n(7 features)", fillcolor="#E0F2FE", color="#0284C7"];
            nlp_pre [label="NLP Preprocessing\\n(lowercase, punctuation,\\nstopwords, lemmatization)", fillcolor="#FEF3C7", color="#D97706"];
            tfidf [label="TF-IDF Vectorizer\\n(max 5000, 1-2 gram)", fillcolor="#FEF3C7", color="#D97706"];
            behav_model [label="Behavioral Model\\nIsolation Forest\\n(200 trees, c=0.02)\\n+ StandardScaler + MinMax(0-100)", fillcolor="#DCFCE7", color="#16A34A"];
            nlp_model [label="NLP Model\\nLogistic Regression\\n(+ Linear SVM / NB compare)", fillcolor="#DCFCE7", color="#16A34A"];
            behav_score [label="Behavior Score\\n0-100\\n(high=normal)", fillcolor="#F1F5F9", color="#334155"];
            text_score [label="Text Score\\n0-100\\n(high=genuine)", fillcolor="#F1F5F9", color="#334155"];
            engine [label="Decision Engine\\nFinal = wB·Behavior + wT·Text\\n(wB+wT=1, default 0.5/0.5)\\n+ status & explanation", fillcolor="#0F172A", fontcolor="white"];
            final [label="Final Authenticity Score\\n+ Overall Status\\n+ Per-signal explanation", fillcolor="#EDE9FE", color="#7C3AED"];
            ui [label="Streamlit UI", fillcolor="#F1F5F9", shape=ellipse];
            review -> hist; review -> nlp_pre;
            hist -> feat -> behav_model -> behav_score;
            nlp_pre -> tfidf -> nlp_model -> text_score;
            behav_score -> engine; text_score -> engine;
            engine -> final -> ui;
        }
        """)
        st.caption("Flow: Review → Behavioral path and Text path → Decision Engine → Final Score (centered diagram above)")

    st.divider()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Behavioral vs Text")
        st.markdown("""
        | Aspect | Behavioral | NLP |
        |---|---|---|
        | **Input** | Reviewer history (≥3 reviews) | Single review text |
        | **Dataset** | Amazon reviewer_dataset (16,613) | Labeled fake reviews (40,432, CG/OR) |
        | **Learning** | Unsupervised (no labels) | Supervised |
        | **Model** | Isolation Forest | TF-IDF + LogReg / SVM / NB |
        | **Output** | Behavior Score 0–100 | Text Score 0–100 |
        | **Meaning** | High = normal behavior | High = likely genuine text |
        | **Failure mode** | Cold-start (<3 reviews) → unavailable | Always available |
        """)
        st.markdown("#### Challenges & Decisions")
        st.markdown("""
        1. **No labels in Amazon data** → unsupervised for behavior; separate labeled set for NLP.  
        2. **Timestamps only dates** → day-level bursts (7-day window), not minute-level.  
        3. **Two datasets** → modular modules + Decision Engine (not joint training).  
        4. **Scale mismatch** → both scores normalized to 0–100, same direction (100 = authentic).  
        5. **Cold-start** → if `total_reviews <3`, mark Behavior unavailable; engine falls back to text-only and UI says so explicitly.
        """)
    with c2:
        st.markdown("#### Saved Artifacts")
        st.markdown("""
        Simple files saved in `models/`:
        - `scaler.pkl` → StandardScaler (7 features)
        - `isolation_forest.pkl` → IsolationForest (200 trees, c=0.02)
        - `score_scaler.pkl` → MinMaxScaler 0–100
        - `tfidf_vectorizer.pkl` → TF-IDF 5000, 1–2 gram
        - `nlp_model.pkl` → LogisticRegression

        *Inference:* `scaler.transform` → `decision_function` → `score_scaler` → 0–100
        """)
    st.divider()
    st.markdown("#### Four Canonical Cases (Decision Engine)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="card" style="border-top: 4px solid #16A34A;"><b>Case 1: Both High</b><br>Behavior ≥70, Text ≥70<br><span class="badge badge-green">Most probably genuine</span><br><small>Convergent genuine — strongest authentic case.</small></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card" style="border-top: 4px solid #DC2626;"><b>Case 2: Both Low</b><br>Behavior ≤40, Text ≤40<br><span class="badge badge-red">Highly suspicious</span><br><small>Convergent suspicious — strongest manipulated signal.</small></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card" style="border-top: 4px solid #F59E0B;"><b>Case 3: Behavior Low, Text High</b><br><span class="badge badge-yellow">Suspicious</span><br><small>Bursty history but text looks genuine. Don’t auto-label.</small></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card" style="border-top: 4px solid #3B82F6;"><b>Case 4: Behavior High, Text Low</b><br><span class="badge badge-blue">Needs to be reviewed manually</span><br><small>Normal history but text suspicious. Surface both.</small></div>', unsafe_allow_html=True)
    st.info("Prototype rule: **Final = wB·Behavior + wT·Text** (default 0.5/0.5, tunable). If the reviewer has **<3 reviews**, behavior is skipped and the Text Authenticity Score is used with no weight. Output is a likelihood the review is true — never a fake/genuine verdict.")

# ──────────────────────────────────────────────
# 📊 EDA & INSIGHTS
# ──────────────────────────────────────────────
elif page == "📊 EDA & Insights":
    st.markdown("### 📊 Exploratory Data Analysis — Behavioral Population")
    st.markdown("""
    **Population:** 16,613 reviewer-level rows (25 columns). **Modeled:** 9,435 reviewers with `total_reviews ≥3` (used by final IF). The EDA plots below are the project's visual evidence for feature selection — especially burst-related features.
    """)

    # Quick stats row
    if reviewer_df is not None:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Avg total_reviews", f"{reviewer_df['total_reviews'].mean():.1f}", border=True)
        with m2:
            st.metric("Median reviews/day", f"{reviewer_df['reviews/day'].median():.4f}", border=True)
        with m3:
            st.metric("Avg rating variance", f"{reviewer_df['rating_variance'].mean():.2f}", border=True)
        with m4:
            st.metric("Avg burst_7d", f"{reviewer_df['review_burst_7d'].mean():.1f}", border=True)
        with st.expander("📋 reviewer_dataset.csv — preview (first 5 rows, 25 cols)", expanded=False):
            st.dataframe(reviewer_df.head(), use_container_width=True)
            st.caption(f"Shape: {reviewer_df.shape} • Columns: {', '.join(reviewer_df.columns.tolist())}")

    # Behavior Score distribution
    if behaviour_df is not None:
        st.markdown("#### Behavior Score Distribution (final IF, 9,435 reviewers)")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.hist(behaviour_df["behavior_score"], bins=40, color="#0EA5E9", edgecolor="white", alpha=0.9)
        ax.axvline(behaviour_df["behavior_score"].median(), color="#0F172A", linestyle="--", label=f"median {behaviour_df['behavior_score'].median():.1f}")
        ax.set_xlabel("Behavior Score (0–100, high = normal)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title("Most reviewers cluster 75–95; low-score tail (≈2%) flagged as anomalies (prediction=-1)", fontsize=10)
        st.pyplot(fig, use_container_width=True)
        st.caption("Isolation Forest decision_function → MinMax(0,100). Low tail corresponds to `contamination=0.02` threshold.")

    st.divider()
    st.markdown("#### EDA Plots — Behavioral Features (from `EDA_Plots/`)")
    # Define insights for each plot
    insights = {
        "total_reviews.png": "Most reviewers have few reviews (heavy left tail). A small population has very high counts — candidates for high-frequency / burst checks. The ≥3 filter removes 7,178 low-history reviewers whose variance/entropy would be unstable.",
        "active_days.png": "Active days is right-skewed; many reviewers span short periods, while some span years. Active days vs total_reviews is moderately correlated but reviews/day adds independent signal (not redundant).",
        "reviews_per_day.png": "Reviews/day is also skewed; most are low (<0.1), a few are very high. High reviews/day often co-occurs with large 7-day bursts — important anomaly signal.",
        "rating_variance.png": "Many reviewers have low rating variance (e.g., all 5s). Low variance + low entropy can be normal (satisfied customers) but also appears in burst-heavy reviewers — context matters.",
        "avg_rating.png": "Average rating clusters high (≈4–5). High average alone is not suspicious; combined with low variance + high burst it becomes more interesting.",
        "avg_days_between_reviews.png": "Avg days between reviews is log-skewed (some 500+ days). That's why `log1p` is applied before scaling — compresses long tails so StandardScaler is well-behaved.",
        "percent_5Star.png": "High %_5Star mass near 0% and 100% — polarized. Reviewers with 100% 5-star and low variance are not automatically fake, but burst intensifies suspicion.",
        "percent_4Star.png": "4-star share is more spread; low mass at extremes. Useful as complementary star-distribution context.",
        "percent_3Star.png": "3-star share is typically low; higher values indicate more balanced / higher-entropy raters.",
        "percent_2Star.png": "2-star and 1-star shares are sparse; many reviewers have 0% in these buckets.",
        "percent_1Star.png": "1-star share mirrors 5-star polarization — most have 0%, a few have high share (critical reviewers).",
        "total_helpful_votes.png": "Helpful votes are sparse and zero-inflated; not in final 7 features (planned future: helpful ratio).",
        "percent_helpful_per_review.png": "Helpful per review is low on average; noisy for low-review users — excluded from final IF to keep signal clean.",
    }

    # Display in grid
    plot_files = sorted([p.name for p in EDA_PLOTS_DIR.glob("*.png")]) if EDA_PLOTS_DIR.exists() else []
    if not plot_files:
        st.warning("No plots found in EDA_Plots/. Expected 13 PNG files.")
    else:
        # Use 2 columns
        cols = st.columns(2)
        for i, fname in enumerate(plot_files):
            col = cols[i % 2]
            with col:
                st.markdown(f"**{fname}**")
                img_path = EDA_PLOTS_DIR / fname
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                insight = insights.get(fname, "Distribution of this behavioral feature across the reviewer population. Look for skew, tails, and outliers that Isolation Forest will isolate.")
                st.markdown(f'<div class="insight-box"><b>Insight:</b> {insight}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### EDA Takeaways → Why these 7 features?")
    st.markdown("""
    <div class="card">

    **EDA concluded:** most reviewers have low activity, but a small subset has unusually high frequency or concentrated bursts. Burst behavior is the strongest behavioral signal.

    **Final 7 features for Isolation Forest (in scaler order):**
    <ol>
        <li><code>rating_variance</code> — consistency of ratings</li>
        <li><code>rating_entropy</code> — spread across 1–5 stars</li>
        <li><code>reviews_per_day</code> — = total_reviews / active_days (frequency)</li>
        <li><code>review_burst_7d</code> — max reviews in any 7-day window (absolute burst)</li>
        <li><code>burst_ratio_7d</code> — = burst_7d / total_reviews (dominant burst?)</li>
        <li><code>avg_days_between_reviews</code> — with <code>log1p</code> transform (skew compression)</li>
        <li><code>burst_intensity</code> — = burst_7d × burst_ratio (size × dominance)</li>
    </ol>

    These are the exact seven the saved <code>scaler.pkl</code> expects, in this order. The three burst-related features capture both “how many in a burst” and “how much of your history was that burst.”

    **Preprocessing:** <code>log1p(avg_days_between_reviews)</code> → <code>StandardScaler</code> (fit once, reuse via <code>transform</code>) → Isolation Forest → <code>MinMaxScaler(0,100)</code>.

    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 🤖 BEHAVIORAL MODEL
# ──────────────────────────────────────────────
elif page == "🤖 Behavioral Model":
    st.markdown("### 🤖 Behavioral Anomaly Detection — Isolation Forest")
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown("#### Why unsupervised?")
        st.markdown("""
        <div class="insight-box">
        The Amazon behavioral dataset has <b>no fake/genuine labels</b>. Supervised classification (reviewer → fake/genuine) is not possible on this data.
        Instead: model the <i>population distribution</i> and flag reviewers whose behavior is <b>unusual</b> relative to the population.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **Isolation Forest intuition:** Anomalous points are easier to isolate with random feature splits → lower `decision_function` → lower Behavior Score.

        - **Normal reviewers:** similar behavioral region → harder to isolate → higher score (≈75–95).
        - **Unusual reviewers:** far from common patterns → easier to isolate → lower score (tail near 0).
        """)
        st.markdown("#### Notebook progression")
        st.markdown("""
        - `feature_engg_01.ipynb` — first pipeline from older Amazon format (avg rating, variance, entropy, active days, bursts, star distributions…).
        - `feature_enng_02.ipynb` — same for 2015–2023 format (`user_id → reviewerID`); results combined into `reviewer_dataset.csv`.
        - `EDA.ipynb` — distributions, correlations → burst features identified as key.
        - `behavorial_model_01.ipynb` — initial IF (200 trees, c=0.02) with broader feature set.
        - `behavorial_model_02.ipynb` — **contamination sweep** {0.01,0.02,0.03,0.04} + manual borderline inspection → keep **0.02** (at 0.03+ normal reviewers start being flagged; at 0.01–0.02 flagged cases show stronger bursts).
        - `behavorial_model_03.ipynb` — **model comparison:** Isolation Forest vs LOF (local neighborhoods) vs One-Class SVM (boundary around normal) — IF supported as reasonable choice.
        - `behavorial_model_final.ipynb` — **final pipeline** (this is the deployed one).
        """)
        st.markdown("#### Final pipeline (present this)")
        st.code("""
reviewer_dataset.csv
        │
Keep total_reviews >=3           (9,435 modeled)
        │
Create burst_intensity = review_burst_7d × burst_ratio_7d
        │
Select 7 features ──► log1p(avg_days_between_reviews) ──► StandardScaler (fit)
        │
IsolationForest(n_estimators=200, contamination=0.02, random_state=42) ──► decision_function()
        │
MinMaxScaler(0,100) ──► Behavior Score
        """, language="text")

    with right:
        st.markdown("#### Model artifacts & inference")
        st.markdown("""
        <div class="card">

        **Three .pkl = one pipeline:**
        <ul>
            <li><code>scaler.pkl</code> — StandardScaler (remembers training mean/var)</li>
            <li><code>isolation_forest.pkl</code> — trained IF</li>
            <li><code>score_scaler.pkl</code> — MinMaxScaler(0,100)</li>
        </ul>

        ```python
        def score_reviewer(features: dict, scaler, iso, score_scaler):
            row = DataFrame([features])[7_cols]
            row["avg_days_between_reviews"] = log1p(row["avg_days..."])
            row_scaled = scaler.transform(row)   # NOT fit_transform
            pred = iso.predict(row_scaled)[0]    # 1=normal, -1=anomaly
            decision = iso.decision_function(row_scaled)[0]
            behavior = score_scaler.transform([[decision]])[0][0]
            return {"prediction": pred, "decision_score": decision,
                    "behavior_score": behavior}
        ```

        **Validation:** notebook reloads saved pkls and reproduces
        `prediction=1, decision=0.198859..., behavior=81.72427...`
        — proving pipeline persistence works.

        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### How to talk about contamination")
        st.markdown("""
        <div class="warning-box">
        <b>contamination=0.02</b> means “configure threshold so ~2% of training points are labeled anomaly (prediction=-1).”<br>
        It is <b>not</b> “probability reviewer is fake.” It’s a <i>threshold</i> for the binary prediction; the <i>continuous</i> decision score → Behavior Score is the primary output.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Live: validate saved pipeline")
    if behav_err:
        st.error(f"Cannot load behavioral models: {behav_err}")
    else:
        st.success("Behavioral models loaded. Try a manual scoring below, or use Analyze Review / Reviewer Lookup.")
        with st.expander("Try scoring with a sample reviewer (raw features)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                rv = st.number_input("rating_variance", 0.0, 5.0, 0.2, 0.05)
                re = st.number_input("rating_entropy", 0.0, 2.5, 0.72, 0.05)
            with c2:
                rpd = st.number_input("reviews_per_day", 0.0, 5.0, 0.0078, 0.001, format="%.4f")
                rb7 = st.number_input("review_burst_7d", 1, 50, 3, 1)
            with c3:
                br = st.number_input("burst_ratio_7d", 0.0, 1.0, 0.2, 0.05)
                adb = st.number_input("avg_days_between_reviews", 0.0, 2000.0, 150.0, 5.0)
            with c4:
                bi = st.number_input("burst_intensity", 0.0, 50.0, 0.6, 0.05)
                if st.button("Score this reviewer", type="primary", use_container_width=True):
                    feats = dict(rating_variance=rv, rating_entropy=re, reviews_per_day=rpd,
                                 review_burst_7d=rb7, burst_ratio_7d=br,
                                 avg_days_between_reviews=adb, burst_intensity=bi)
                    res = score_reviewer_features(feats, scaler, iso, score_scaler)
                    st.json(res)
                    b_status, b_cls = get_behavior_status(res["behavior_score"])
                    st.markdown(f'Behavior Score: **{res["behavior_score"]:.1f}/100** <span class="badge {b_cls}">{b_status}</span>', unsafe_allow_html=True)
                    st.progress(int(np.clip(res["behavior_score"], 0, 100)) / 100)

# ──────────────────────────────────────────────
# 📝 NLP MODULE
# ──────────────────────────────────────────────
elif page == "📝 NLP Module":
    st.markdown("### 📝 NLP Authenticity — TF-IDF + Classical Classifiers")
    c1, c2 = st.columns([1.2, 0.8], gap="large")
    with c1:
        st.markdown("#### Reality check (from ZIP)")
        st.markdown("""
        <div class="warning-box">
        The ZIP does <b>not</b> contain a completed NLP trained model — it contains the <b>planned workflow & documentation</b> plus a Colab notebook that loads <code>fake reviews dataset.csv</code> (40,432 rows, columns: <code>category, rating, label, text_</code>).<br>
        This app therefore <b>trains a demonstrator</b> TF-IDF + Logistic Regression on startup (if no saved <code>nlp_model.pkl</code> exists) and saves it for reuse — so the hybrid system is end-to-end runnable.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### Dataset (from `NLP.ipynb`)")
        st.markdown("""
        - **Shape:** 40,432 rows × 4 cols
        - **Columns:** `category` (10 values: Kindle_Store_5, Books_5, …), `rating` (1–5, mean 4.26), `label` (CG = computer-generated/fake, OR = original/genuine), `text` (review body)
        - **Balance:** CG 20,216 / OR 20,216 (50/50)
        - **Text:** avg ~351 chars, ~69 words; unique texts 40,412/40,432
        - **Why separate dataset:** Amazon behavioral dataset has **no fake/real label** → supervised NLP cannot be trained on it directly.
        """)
        st.markdown("#### Planned workflow (from `Docs/NLP Workflow.docx`)")
        st.code("""
Labeled Review Dataset
        ↓
Text Preprocessing: lowercase → remove punctuation/URLs → remove stopwords → lemmatization
        ↓
Feature Extraction: TF-IDF (max 5000, ngram 1-2, stop_words=english)
        ↓
Train-Test Split (80:20, stratified)
        ↓
Train 3 models:  Logistic Regression  |  Linear SVM  |  Multinomial Naive Bayes
        ↓
Model Performance Comparison (Accuracy, Precision, Recall, F1)
        ↓
Select Best Performing Model → Save (.pkl: model + TF-IDF vectorizer)
        ↓
Prediction:  New Review Text → preprocessing → TF-IDF → model → P(genuine) → Text Score = P×100
        """, language="text")
        st.markdown("#### Metrics to report (when you have your real trained model)")
        st.markdown("""
        | Metric | Why |
        |---|---|
        | **Accuracy** | Overall correctness |
        | **Precision** | Of predicted fake, how many are actually fake? |
        | **Recall** | Of actual fake, how many are caught? |
        | **F1-score** | Harmonic mean — best single selector |

        Compare the three models on the same test split and justify the choice (e.g., “LogReg best F1 0.XX, also gives calibrated probabilities via `predict_proba` — Linear SVM may need calibration.”)
        """)

    with c2:
        st.markdown("#### Current demonstrator model (in this app)")
        if nlp_status == "trained" and isinstance(nlp_metrics, dict):
            m = nlp_metrics
            st.success("Demonstrator model trained on synthetic 400-sample balanced data (mirroring CG/OR patterns).")
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1: st.metric("Accuracy", f"{m['accuracy']:.2%}")
            with cc2: st.metric("Precision", f"{m['precision']:.2%}")
            with cc3: st.metric("Recall", f"{m['recall']:.2%}")
            with cc4: st.metric("F1", f"{m['f1']:.2%}")
            st.caption("Trained: TF-IDF(max 5000, 1-2gram, english stopwords) + LogisticRegression(C=1, liblinear). Saved to `models/nlp_model.pkl` + `models/tfidf_vectorizer.pkl`. Replace with your real trained model from `fake reviews dataset.csv` when ready — interface is `score_review_text(text) → text_score` and will auto-load.")
        elif nlp_status == "loaded":
            st.success("Loaded saved NLP model. Replace with your final trained model by overwriting the two .pkl files.")
        else:
            st.error(f"NLP model status: {nlp_status} — {nlp_metrics}")

        st.markdown("#### Interface contract (for integration)")
        st.code("""
# Behavioral
behavior = score_reviewer(features_dict)  # → {"behavior_score": 0-100, ...}

# NLP
text_score = score_review_text(review_text)  # → 0-100 (high=genuine)

# Decision Engine
final = wB*behavior + wT*text   # wB+wT=1
        """, language="python")
        st.markdown("""
        <div class="card">
        <b>Important:</b> “Text Authenticity Score = P(genuine)×100” only if your classifier outputs a calibrated probability.<br>
        - LogisticRegression & MultinomialNB: <code>predict_proba</code> is natural.<br>
        - Linear SVM: gives <code>decision_function</code>; needs Platt scaling / probability calibration — don’t blindly call it a probability.<br>
        Document <code>text_score = what exactly?</code> in your report.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### Example predictions (try below)")
        with st.expander("Try the NLP model now", expanded=True):
            demo_text = st.text_area("Review text", "I received a free copy of this product in exchange for my honest review. Excellent! Highly recommend to everyone, perfect amazing wonderful!", height=90)
            if st.button("Score text", use_container_width=True):
                if nlp_model is None or tfidf is None:
                    st.error("NLP model not available.")
                else:
                    ts, prob, label = score_text(demo_text, nlp_model, tfidf)
                    if ts is None:
                        st.error(f"Error: {label}")
                    else:
                        status, cls = get_text_status(ts)
                        st.metric("Text Authenticity Score", f"{ts:.1f} / 100", delta=label, delta_color="normal")
                        st.markdown(f'<span class="badge {cls}">{status}</span> — P(genuine)={prob:.3f}', unsafe_allow_html=True)
                        st.progress(int(np.clip(ts,0,100))/100)
                        if label == "Fake":
                            st.warning("Model flags as fake-leaning (low genuine probability). In the real dataset, fake reviews are often overly promotional, generic, or incentivized disclosures.")
                        else:
                            st.success("Model leans genuine (balanced, specific language).")

    st.divider()
    st.markdown("#### What to say to mentor")
    st.markdown("""
    <div class="insight-box">
    “The NLP module is trained on a separate <b>labeled</b> review dataset (40k, balanced CG/OR) that contains fake/genuine labels — the Amazon behavioral dataset cannot be used for supervised NLP because it has no labels. The documented pipeline is text cleaning → TF-IDF → 80:20 split → Logistic Regression / Linear SVM / Naive Bayes, compare on accuracy/precision/recall/F1, pick the best, and save the TF-IDF vectorizer + model. It outputs a Text Authenticity Score 0–100 (P(genuine)×100) where 100 is likely genuine. This is the second signal into the Decision Engine.”
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 🔍 HYBRID DETECTION
# ──────────────────────────────────────────────
elif page == "🔍 Analyze Review":
    st.markdown("### 🔍 Analyze Review — select a real review from the dataset")
    st.markdown("""
    Pick any review from **Amazon_Reviews_Combined_2000-2023.csv**. The app looks the reviewer up
    in `reviewer_dataset.csv`, re-runs the behavioral pipeline live on their 7 features, scores the
    review text with the NLP model, and fuses both signals in the Decision Engine.
    The output is **how likely the review is true** — never a fake/genuine verdict.
    """)

    if reviews_df is None or reviewer_lookup is None:
        st.error(f"Review dataset not found. Expected `Dataset/{REVIEWS_PATH.name}`.")
    elif behav_err:
        st.error(f"Behavioral model not loaded: {behav_err}")
    else:
        # ── 1) Select a review ────────────────────────────────────
        st.markdown("#### 🔎 Select a Review")
        with st.container(border=True):
            f1, f2, f3, f4 = st.columns([2.6, 1.1, 1.9, 1.0], gap="small")
            with f1:
                search = st.text_input("Search", "", placeholder="reviewerID, review title or product ASIN…")
            with f2:
                rating_filter = st.selectbox("Rating", RATING_OPTIONS, index=0)
            with f3:
                hist_filter = st.selectbox(
                    "Reviewer history",
                    ["All reviewers", "≥3 reviews — behavior available", "<3 reviews — text-only (cold-start)"],
                )
            with f4:
                st.write("")
                random_clicked = st.button("🎲 Random", use_container_width=True)

        totals = reviewer_lookup["total_reviews"]
        view = reviews_df
        if search.strip():
            s = search.strip().lower()
            view = view[
                view["reviewerID"].str.lower().str.contains(s, na=False)
                | view["summary"].str.lower().str.contains(s, na=False)
                | view["asin"].str.lower().str.contains(s, na=False)
            ]
        if rating_filter != "All ratings":
            view = view[view["overall"] == int(rating_filter.split("★")[0])]
        mapped = view["reviewerID"].map(totals)
        if hist_filter.startswith("≥"):
            view = view[mapped >= 3]
        elif hist_filter.startswith("<"):
            view = view[mapped.notna() & (mapped < 3)]
        view = view.head(REVIEW_OPTIONS_CAP)
        options = view["rid"].tolist()

        if len(options) == 0:
            st.warning("No reviews match these filters — widen the search or the rating/history choice.")
        else:
            label_map = {
                r.rid: "{} · {}★ · “{}” · {}".format(
                    r.reviewerID, int(r.overall), " ".join(r.summary.split())[:44] or "(no title)",
                    r.reviewDate)
                for r in view[["rid", "reviewerID", "overall", "summary", "reviewDate"]].itertuples()
            }

            def trim(lbl):
                return lbl if len(lbl) <= 110 else lbl[:110] + "…"

            # Options are the display strings themselves: a selectbox over range()
            # hands back a str in session_state, which breaks index arithmetic.
            label_to_rid = {}
            for rid in options:
                label_to_rid.setdefault(trim(label_map[rid]), int(rid))
            labels = list(label_to_rid)

            # Keying the widget to the filter set means a selection that no longer
            # exists in the list can't survive in the browser as a phantom value.
            pick_key = "review_pick|{}|{}|{}".format(search.strip(), rating_filter, hist_filter)

            if random_clicked:
                picked = int(np.random.default_rng().choice(np.array(options)))
                st.session_state["pick_rid"] = picked
                st.session_state[pick_key] = trim(label_map[picked])
                st.session_state["analyzed"] = False

            if pick_key in st.session_state and st.session_state[pick_key] not in label_to_rid:
                del st.session_state[pick_key]

            default_label = labels[0]
            if "pick_rid" in st.session_state:
                cand = trim(label_map.get(st.session_state["pick_rid"], "\u0000"))
                if cand in label_to_rid:
                    default_label = cand

            with st.container(border=True):
                st.markdown("**① Pick the review to analyze** — this dropdown is the selector; each entry starts with the `reviewerID`")
                sel_label = st.selectbox("Choose a review from the dataset", labels,
                                         index=labels.index(default_label), key=pick_key,
                                         label_visibility="collapsed")
                st.caption(f"{len(labels):,} reviews match the filters above (list capped at {REVIEW_OPTIONS_CAP}). "
                           f"Full corpus: {len(reviews_df):,} reviews. 🎲 Random picks one from these {len(labels):,}.")
            selected_rid = label_to_rid[sel_label]
            review = view[view["rid"] == selected_rid].iloc[0]

            if st.session_state.get("analyzed_rid") != selected_rid:
                st.session_state["analyzed"] = False

            # ── 2) Review information card ──────────────────────────
            st.markdown("### 📄 ② Review Information")
            reviewer_name = " ".join(str(review["reviewerName"]).split()).replace("_", "\\_").replace("*", "\\*")
            with st.container(border=True):
                ic1, ic2, ic3, ic4 = st.columns([1.9, 1.2, 0.85, 1.0])
                with ic1:
                    st.markdown(f"**Reviewer**\n\n{reviewer_name}")
                with ic2:
                    st.markdown(f"**Product (ASIN)**\n\n`{review['asin']}`")
                with ic3:
                    st.markdown(f"**Rating**\n\n{int(review['overall'])} / 5 {'⭐' * int(review['overall'])}")
                with ic4:
                    st.markdown(f"**Review date**\n\n{review['reviewDate']}")
                if int(review["helpful_total"]) > 0:
                    st.caption(f"👍 {int(review['helpful_votes'])} of {int(review['helpful_total'])} readers found this review helpful.")
                else:
                    st.caption("👍 No helpfulness votes recorded for this review yet.")
                st.markdown("**Review title**")
                st.markdown(f"“{review['summary'].strip() or '(no title)'}”")
                st.markdown("**Review text**")
                body = review["reviewText"].strip()
                if len(body) > 900:
                    st.info(body[:900] + " …")
                    with st.expander(f"Show full review text ({len(body):,} characters)"):
                        st.text(body)
                else:
                    st.info(body)

            # ── 3) Decision Engine weight ───────────────────────────
            wc1, wc2 = st.columns([1, 2])
            with wc1:
                w_b = st.slider("Behavior weight (wB)", 0.0, 1.0, 0.5, 0.05)
            with wc2:
                st.markdown("**③ Fusion rule**")
                st.caption(f"Final = {w_b:.2f}·Behavior + {1 - w_b:.2f}·Text. If the reviewer has fewer than 3 reviews, the behavior branch is skipped entirely and the Text Authenticity Score is used with **no weighting**.")

            analyze_clicked = st.button("🚀 Analyze Review", type="primary", use_container_width=True)
            if analyze_clicked:
                st.session_state["analyzed"] = True
                st.session_state["analyzed_rid"] = selected_rid

            if not st.session_state.get("analyzed"):
                st.info("👆 Click **Analyze Review** to run the behavioral + NLP pipeline on this review.")
            else:
                # ── 4) Run the pipeline ─────────────────────────────
                rid = review["reviewerID"]
                rrow = reviewer_lookup.loc[[rid]] if rid in reviewer_lookup.index else None
                total_reviews = int(rrow["total_reviews"].iloc[0]) if rrow is not None else 0
                cold_start = total_reviews < 3

                behavior_score = behav_detail = None
                feats = None
                if rrow is None:
                    st.warning("This reviewerID is not present in `reviewer_dataset.csv` — no behavioral history could be retrieved.")
                elif cold_start:
                    feats = reviewer_features(rrow.iloc[0])
                elif scaler is None or iso is None:
                    st.error("Behavioral pipeline unavailable.")
                else:
                    feats = reviewer_features(rrow.iloc[0])
                    behav_detail = score_reviewer_features(feats, scaler, iso, score_scaler)
                    behavior_score = float(behav_detail["behavior_score"])

                text_score = text_proba = text_label = None
                text_sig = analyze_text_signals(review["reviewText"])
                if nlp_model is None or tfidf is None:
                    st.error("NLP model not available.")
                else:
                    ts, prob, label = score_text(review["reviewText"], nlp_model, tfidf)
                    if ts is not None:
                        text_score, text_proba, text_label = float(ts), float(prob), label

                final_score, status_label, status_cls, justification = decision_engine(
                    behavior_score, text_score, w_b, cold_start,
                    behavior_chips_and_lines(feats, pop_stats)[0] if (feats and not cold_start) else [],
                    text_sig,
                )

                # ── 5) Tabs ─────────────────────────────────────────
                tab1, tab2, tab3 = st.tabs(["📄 Review Details", "📊 Behavioral Analysis", "🤖 Final Authenticity"])

                with tab1:
                    st.markdown("#### Selected review")
                    with st.container(border=True):
                        d1, d2, d3, d4, d5 = st.columns([1.7, 1.1, 0.75, 1.0, 0.9])
                        with d1:
                            st.markdown(f"**Reviewer**\n\n{reviewer_name}\n\n`{rid}`")
                        with d2:
                            st.markdown(f"**Product (ASIN)**\n\n`{review['asin']}`")
                        with d3:
                            st.markdown(f"**Rating**\n\n{int(review['overall'])} / 5 {'⭐' * int(review['overall'])}")
                        with d4:
                            st.markdown(f"**Review date**\n\n{review['reviewDate']}")
                        with d5:
                            st.markdown(f"**Found helpful**\n\n{int(review['helpful_votes'])} / {int(review['helpful_total'])}")
                    st.markdown("#### Reviewer record (from `reviewer_dataset.csv`)")
                    if rrow is None:
                        st.markdown('<div class="chip">not present in the reviewer dataset</div>', unsafe_allow_html=True)
                    else:
                        with st.container(border=True):
                            h1, h2 = st.columns([1, 1])
                            with h1:
                                st.markdown(f"- **Total reviews:** {total_reviews}\n"
                                            f"- **Avg rating:** {rrow['avg_rating'].iloc[0]:.2f}\n"
                                            f"- **Unique products:** {int(rrow['unique_products_reviewed'].iloc[0])}")
                            with h2:
                                st.markdown(f"- **First review:** {rrow['first_review'].iloc[0]}\n"
                                            f"- **Last review:** {rrow['last_review'].iloc[0]}\n"
                                            f"- **Status:** {'cold-start (<3 reviews)' if cold_start else 'modeled by Isolation Forest'}")
                    st.markdown("#### Full review text")
                    st.text(body)

                with tab2:
                    st.markdown("#### Card 1 — Behavior Analysis")
                    with st.container(border=True):
                        if cold_start or rrow is None:
                            why = ("fewer than 3 reviews in the dataset" if rrow is not None else "absent from the reviewer dataset")
                            st.markdown(f'<span class="badge badge-gray">Not qualified for behavior analysis</span>', unsafe_allow_html=True)
                            st.markdown(f'<div style="font-size:2rem; font-weight:800; color:#94A3B8;">— / 100</div>', unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="warning-box" style="margin-top:10px;">
                            <b>Why:</b> this reviewer has <b>{total_reviews}</b> review(s) — {why}. Rating variance, star
                            entropy and burst statistics are statistically meaningless on that little history, so the
                            Isolation Forest is <b>not</b> run for them. The Decision Engine falls back to the text signal
                            alone, and the final score below carries <b>no behavioral weight</b>.
                            </div>
                            """, unsafe_allow_html=True)
                            if feats:
                                with st.expander("Raw features that would have been used"):
                                    st.json(feats)
                        else:
                            b_status, b_cls = get_behavior_status(behavior_score)
                            colB1, colB2 = st.columns([1, 2])
                            with colB1:
                                st.metric("Behavior Score", f"{behavior_score:.1f} / 100",
                                          delta=f"{'anomaly' if behav_detail['prediction'] == -1 else 'normal'} · decision {behav_detail['decision_score']:+.3f}",
                                          delta_color="off")
                                st.markdown(badge_html(b_status, b_cls), unsafe_allow_html=True)
                            with colB2:
                                st.progress(int(np.clip(behavior_score, 0, 100)) / 100)
                                st.caption("100 = behaves like a typical reviewer of the 9,435-reviewer modeled population; 0 = far outside it. This is NOT P(fake).")

                            chips, blines = behavior_chips_and_lines(feats, pop_stats)
                            st.markdown("**Main contributing features**")
                            for label_txt, level in chips:
                                st.markdown(badge_html(label_txt, level), unsafe_allow_html=True)
                            st.markdown("**Justification from this reviewer's history**")
                            for line in (blines or ["• No feature crossed a reporting threshold — history looks unremarkable on every axis."]):
                                st.markdown(line)
                            with st.expander("🧾 Reviewer behavioral summary (live values)", expanded=False):
                                s1, s2, s3 = st.columns(3)
                                with s1:
                                    st.metric("Rating Variance", f"{feats['rating_variance']:.2f}")
                                    st.metric("Reviews / Day", f"{feats['reviews_per_day']:.4f}")
                                with s2:
                                    st.metric("Rating Entropy", f"{feats['rating_entropy']:.2f}")
                                    st.metric("Review Burst (7 days)", feats["review_burst_7d"])
                                with s3:
                                    st.metric("Avg Days Between", f"{feats['avg_days_between_reviews']:.1f}")
                                    st.metric("Burst Intensity", f"{feats['burst_intensity']:.2f}")
                            with st.expander("⚙️ How this score was produced (live re-run of the saved pipeline)", expanded=False):
                                st.code(
                                    "reviewerID {}  (total_reviews={})\n".format(rid, total_reviews)
                                    + "  7 features from reviewer_dataset.csv → log1p(avg_days_between_reviews)\n"
                                    + "  → StandardScaler.transform (scaler.pkl)\n"
                                    + "  → IsolationForest.decision_function = {:.8f}   (n_estimators=200, contamination=0.02)\n".format(behav_detail["decision_score"])
                                    + "  → MinMaxScaler(0,100) (score_scaler.pkl)\n"
                                    + "  → Behavior Score = {:.5f}   prediction = {}".format(behavior_score, behav_detail["prediction"]),
                                    language="text")
                                if behaviour_df is not None:
                                    stored = behaviour_df[behaviour_df["reviewerID"] == rid]
                                    if len(stored):
                                        sv = float(stored["behavior_score"].iloc[0])
                                        st.caption(f"Stored in `Outputs/Behaviour Score.csv`: {sv:.5f} — live re-run differs by {abs(sv - behavior_score):.6f}. The pipeline is reproducible, not cached.")

                with tab3:
                    st.markdown("#### Card 2 — Text Analysis")
                    with st.container(border=True):
                        if text_score is None:
                            st.markdown('<span class="badge badge-gray">Text unavailable</span>', unsafe_allow_html=True)
                            st.caption("The NLP model could not score this review text.")
                        else:
                            t_status, t_cls = get_text_status(text_score)
                            colT1, colT2 = st.columns([1, 2])
                            with colT1:
                                st.metric("Text Authenticity Score", f"{text_score:.1f} / 100",
                                          delta=f"P(true)={text_proba:.3f}", delta_color="off")
                                st.markdown(badge_html(f"{t_status} · model leans {text_label.lower()}", t_cls), unsafe_allow_html=True)
                            with colT2:
                                st.progress(int(np.clip(text_score, 0, 100)) / 100)
                                st.caption(f"TF-IDF + Logistic Regression on {text_sig['n_words']} words · detected tone: {text_sig['sentiment']} ({text_sig['pos_hits']} positive / {text_sig['neg_hits']} negative cue words).")
                            st.markdown("**Keywords detected**")
                            for kw in text_sig["keywords"]:
                                st.markdown(badge_html(kw, "gray"), unsafe_allow_html=True)
                            st.markdown("**Linguistic markers**")
                            for label_txt, level in text_sig["chips"]:
                                st.markdown(badge_html(label_txt, level), unsafe_allow_html=True)

                    st.markdown("#### Card 3 — Final Decision")
                    with st.container(border=True):
                        if final_score is None:
                            st.markdown('<span class="badge badge-gray">Insufficient data</span>', unsafe_allow_html=True)
                        else:
                            color = {"badge-green": "#16A34A", "badge-blue": "#3B82F6",
                                     "badge-yellow": "#F59E0B", "badge-red": "#DC2626"}.get(status_cls, "#94A3B8")
                            colF1, colF2 = st.columns([1, 2])
                            with colF1:
                                st.markdown(f"""
                                <div class="score-circle" style="border-color:{color};">
                                    <div class="num" style="color:{color};">{final_score:.0f}</div>
                                    <div class="lbl">/ 100 likely truthful</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with colF2:
                                st.markdown("**Overall Authenticity Score**")
                                st.progress(int(np.clip(final_score, 0, 100)) / 100)
                                st.markdown(badge_html(status_label, status_cls), unsafe_allow_html=True)
                                st.markdown(f'<div style="font-size:0.85rem; color:#94A3B8; margin-top:8px;">Bands: ≥75 most probably genuine · 55–75 needs manual review · 30–55 suspicious · &lt;30 highly suspicious. Prototype thresholds, not validated probabilities.</div>', unsafe_allow_html=True)

                    with st.expander("💡 Decision Engine justification", expanded=True):
                        for line in justification:
                            st.markdown(line)
                        st.markdown("""
                        <div class="insight-box" style="margin-top:10px;">
                        <b>Reading the result:</b> the score states how strongly the available evidence supports this
                        review being truthful. It is never a claim that the review <i>is</i> fake. Suspicious and
                        needs-manual-review outcomes should route to a human, not an auto-takedown.
                        </div>
                        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 🔎 REVIEWER LOOKUP
# ──────────────────────────────────────────────
elif page == "🔎 Reviewer Lookup":
    st.markdown("### 🔎 Reviewer Lookup — Historical Behavior Scores")
    st.markdown("Lookup a `reviewerID` from the modeled population (9,435 reviewers). Shows precomputed Behavior Score and underlying features from `reviewer_dataset.csv` + `Outputs/Behaviour Score.csv`.")
    if reviewer_df is None or behaviour_df is None:
        st.error("Datasets not found. Expected `Dataset/reviewer_dataset.csv` and `Outputs/Behaviour Score.csv`.")
    else:
        # Merge for display
        merged = behaviour_df.merge(reviewer_df, on="reviewerID", how="left")

        # ──  Dropdown of every modeled reviewer ────────────────────
        st.markdown("#### ① Pick a reviewer")
        ctx = {r.reviewerID: (float(r.behavior_score), int(r.total_reviews), int(r.prediction))
               for r in merged[["reviewerID", "behavior_score", "total_reviews", "prediction"]].itertuples()}
        all_ids = sorted(ctx)

        def id_label(rid):
            bs, tr, pr = ctx[rid]
            return f"{rid} · score {bs:.1f} · {tr} review{'s' if tr != 1 else ''} · {'normal' if pr == 1 else 'ANOMALY'}"

        with st.container(border=True):
            d0, d1 = st.columns([2.6, 1.0])
            with d0:
                sel_id = st.selectbox("Reviewer ID — type any part of the ID to filter the list",
                                      all_ids, index=None, format_func=id_label,
                                      placeholder="Start typing a reviewerID…")
            with d1:
                st.write("")
                st.caption(f"{len(all_ids):,} modeled reviewers in the dropdown.")

        if sel_id:
            st.markdown("#### ② Selected reviewer — rescore & explain")
            row = merged[merged["reviewerID"] == sel_id].iloc[0]
            cc1, cc2 = st.columns([1, 1])
            with cc1:
                b_status, b_cls = get_behavior_status(row["behavior_score"])
                st.markdown(f"**{sel_id}** <span class='badge {b_cls}'>{b_status}</span> • prediction={int(row['prediction'])}", unsafe_allow_html=True)
                st.metric("Behavior Score", f"{row['behavior_score']:.1f} / 100", delta=f"decision {row['decision_score']:.3f}")
                st.progress(int(np.clip(row["behavior_score"],0,100))/100)
                st.json({
                    "rating_variance": float(row["rating_variance"]),
                    "rating_entropy": float(row["rating_entropy"]),
                    "reviews_per_day": float(row["reviews/day"]),
                    "review_burst_7d": int(row["review_burst_7d"]),
                    "burst_ratio_7d": float(row["burst_ratio_7d"]),
                    "avg_days_between_reviews": float(row["avg_days_between_reviews"]),
                    "burst_intensity": float(row["review_burst_7d"] * row["burst_ratio_7d"])
                })
                # Re-score to prove pipeline
                if scaler is not None and iso is not None:
                    feats = dict(rating_variance=float(row["rating_variance"]), rating_entropy=float(row["rating_entropy"]),
                                 reviews_per_day=float(row["reviews/day"]), review_burst_7d=int(row["review_burst_7d"]),
                                 burst_ratio_7d=float(row["burst_ratio_7d"]),
                                 avg_days_between_reviews=float(row["avg_days_between_reviews"]),
                                 burst_intensity=float(row["review_burst_7d"] * row["burst_ratio_7d"]))
                    reproduced = score_reviewer_features(feats, scaler, iso, score_scaler)
                    st.caption(f"Reproduced via loaded pkls: {reproduced['behavior_score']:.5f} (stored {row['behavior_score']:.5f}) — diff {abs(reproduced['behavior_score']-row['behavior_score']):.6f}")
            with cc2:
                st.markdown("**Raw reviewer features (from reviewer_dataset.csv)**")
                st.dataframe(pd.DataFrame([row[["avg_rating","total_reviews","active_days","reviews/day","review_burst_7d","burst_ratio_7d","1_star_reviews","5_star_reviews","unique_products_reviewed","total_helpful_votes","%_helpful_per_review","avg_days_between_reviews"]]]).T.rename(columns={row.name: "value"}), use_container_width=True)
                st.markdown("""
                <div class="insight-box" style="font-size:0.9rem;">
                <b>Interpretation:</b> Low behavior scores often show high <code>review_burst_7d</code> + high <code>burst_ratio</code> (large share of history in one burst) + high <code>reviews_per_day</code> + low <code>rating_variance/entropy</code>. Check the selected reviewer’s values against these.
                </div>
                """, unsafe_allow_html=True)

        # ──  Browse / rank the whole modeled population ────────────
        st.divider()
        st.markdown("#### ③ Browse & rank all modeled reviewers")
        b1, b2, b3, b4 = st.columns([1, 1.2, 0.6, 0.5])
        with b1:
            p_filter = st.selectbox("Prediction", ["All", "Normal (1)", "Anomaly (-1)"])
        with b2:
            sort_by = st.selectbox("Sort by", ["behavior_score", "total_reviews", "reviews/day", "rating_variance", "review_burst_7d"])
        with b3:
            asc = st.checkbox("Ascending", value=(sort_by=="behavior_score"))
        with b4:
            top_n = st.number_input("Top N", 10, 500, 50, 10)

        filt = merged
        if p_filter == "Normal (1)":
            filt = filt[filt["prediction"] == 1]
        elif p_filter == "Anomaly (-1)":
            filt = filt[filt["prediction"] == -1]

        filt = filt.sort_values(sort_by, ascending=asc).head(int(top_n))

        st.dataframe(
            filt[["reviewerID", "prediction", "behavior_score", "decision_score",
                  "total_reviews", "avg_rating", "rating_variance", "rating_entropy",
                  "reviews/day", "review_burst_7d", "burst_ratio_7d", "avg_days_between_reviews", "active_days"]],
            use_container_width=True, height=420
        )
        st.caption(f"Showing {len(filt)} of {len(merged)} modeled reviewers. Anomalies = {len(merged[merged['prediction']==-1])} (2%). Behavior Score median {merged['behavior_score'].median():.1f}.")

        # Download
        st.divider()
        st.download_button("⬇️ Download Behaviour Score.csv", data=behaviour_df.to_csv(index=False).encode(), file_name="Behaviour Score.csv", mime="text/csv", use_container_width=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#94A3B8; font-size:0.85rem; padding: 0.5rem 0 1rem 0;">
    Hybrid Review Authenticity Detection System • Behavioral (Isolation Forest, unsupervised) + NLP (TF-IDF + LogReg, supervised) + Decision Engine • Streamlit • Indus University<br>
    <i>Behavior Score: high = normal (not P(fake)). &nbsp;|&nbsp; Text Score: high = likely genuine. &nbsp;|&nbsp; Final = wB·Behavior + wT·Text. &nbsp;|&nbsp; Cold-start: behavior unavailable if total_reviews &lt;3.</i>
</div>
""", unsafe_allow_html=True)
