import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="🎥 AI Movie Rating Predictor", layout="centered")

# ------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------
@st.cache_resource
def load_models():
    models = joblib.load("ensemble_models.pkl")
    return models["word_model"], models["char_model"]

word_model, char_model = load_models()

# ------------------------------------------------------
# MOVIES
# ------------------------------------------------------
movies = [
    {"title": "Deadpool & Wolverine", "poster": "https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4t6EDMubfO.jpg"},
    {"title": "Gladiator II", "poster": "https://image.tmdb.org/t/p/w500/2cxhvwyEwRlysAmRH4iodkvo0z5.jpg"},
    {"title": "Moana 2", "poster": "https://image.tmdb.org/t/p/w500/aLVkiINlIeCkcZIzb7XHzPYgO6L.jpg"},
    {"title": "Mufasa: The Lion King", "poster": "https://image.tmdb.org/t/p/w500/lurEK87kukWNaHd0zYnsi3yzJrs.jpg"}
]

# ------------------------------------------------------
# STYLE (CSS)
# ------------------------------------------------------
st.markdown("""
    <style>
    /* General Page */
    body, .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #f5f5f5;
        font-family: "Segoe UI", Roboto, Arial, sans-serif;
    }
    h1 {
        font-size: 3rem !important;
        text-align: center;
        margin-top: 50px !important;
        margin-bottom: 10px !important;
        text-shadow: 0 0 15px rgba(255,255,255,0.3);
    }
    h3 {
        text-align: center;
        font-weight: 500;
        color: #f1f1f1;
        margin-top: 20px !important;
    }
    /* Movie grid */
    .movie-grid {
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
        margin-top: 40px;
    }
    .movie-card {
        cursor: pointer;
        transition: all 0.3s ease;
        border-radius: 12px;
        overflow: hidden;
        text-align: center;
        padding: 10px;
        background-color: rgba(255,255,255,0.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .movie-card:hover {
        transform: translateY(-5px) scale(1.03);
        box-shadow: 0 8px 20px rgba(255,255,255,0.2);
    }
    .movie-card img {
        width: 180px;
        height: 270px;
        object-fit: cover;
        border-radius: 8px;
    }
    /* Large Poster */
    .poster-large {
        width: 280px;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(255,255,255,0.3);
        margin: 30px auto;
        display: block;
    }
    /* Rating Display */
    .rating {
        font-size: 4em;
        color: #facc15;
        text-align: center;
        text-shadow: 0 0 15px rgba(250,204,21,0.7);
    }
    textarea {
        border-radius: 8px;
        padding: 12px;
        width: 100%;
        font-size: 1rem;
        resize: none;
        background-color: rgba(255,255,255,0.1);
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
    }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------------
# PREDICT FUNCTION
# ------------------------------------------------------
def predict_ensemble(texts, clip_range=(1, 10)):
    p1 = word_model.predict(texts)
    p2 = char_model.predict(texts)
    preds = (p1 + p2) / 2.0
    return np.clip(preds, *clip_range)

# ------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "predicted_rating" not in st.session_state:
    st.session_state.predicted_rating = None

# ------------------------------------------------------
# UI LOGIC
# ------------------------------------------------------
st.markdown("<h1>🎥 AI Movie Rating Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# Step 1: Movie selection
if not st.session_state.selected_movie:
    st.markdown("<h3>Step 1: Choose the movie you want to review</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    cols = st.columns(len(movies))
    for i, movie in enumerate(movies):
        with cols[i]:
            st.image(movie["poster"], caption=movie["title"], use_container_width=True)
            if st.button(movie["title"]):
                st.session_state.selected_movie = movie
                st.session_state.predicted_rating = None
                st.rerun()  # ✅ Forces rerun immediately, no second click needed

# Step 2: Review and prediction
else:
    movie = st.session_state.selected_movie
    st.markdown(f"<h3>Step 2: Write a review for <b>{movie['title']}</b></h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    st.image(movie["poster"], width=280)

    review = st.text_area("✏️ Write your review here:", height=150, key="review_input")

    if st.button("Predict My Rating ⭐"):
        cleaned_review = review.strip()
        if len(cleaned_review.split()) < 3:
            st.warning("⚠️ Please write a bit more text for an accurate prediction.")
        else:
            vectorizer = None
            for name, step in word_model.named_steps.items():
                if isinstance(step, (TfidfVectorizer, CountVectorizer)):
                    vectorizer = step
                    break
            if vectorizer is not None:
                vec = vectorizer.transform([cleaned_review])
                if vec.nnz == 0:
                    st.warning("No known words found — please try a longer or clearer review.")
                else:
                    pred = predict_ensemble([cleaned_review])[0]
                    st.session_state.predicted_rating = int(np.clip(round(pred), 1, 10))
                    st.rerun()  # ✅ Makes the result appear immediately

# Step 3: Show result
if st.session_state.predicted_rating:
    movie = st.session_state.selected_movie
    st.markdown("---")
    st.markdown(f"<h3>Your predicted rating for <b>{movie['title']}</b> is...</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='rating'>⭐ {st.session_state.predicted_rating}/10 ⭐</div>", unsafe_allow_html=True)
    if st.button("🔄 Try another movie"):
        st.session_state.selected_movie = None
        st.session_state.predicted_rating = None
        st.rerun()

