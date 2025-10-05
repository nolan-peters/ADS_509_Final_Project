from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

models = joblib.load("ensemble_models.pkl")
word_model = models["word_model"]
char_model = models["char_model"]

def predict_ensemble(texts, clip_range=(1, 10)):
    p1 = word_model.predict(texts)
    p2 = char_model.predict(texts)
    preds = (p1 + p2) / 2.0
    print("RAW PRED word/char:", p1, p2, "AVG:", (p1+p2)/2)
    return np.clip(preds, *clip_range)

@app.route('/', methods=['GET'])
def home():
    movies = [
        {"title": "Deadpool & Wolverine", "poster": "https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4t6EDMubfO.jpg"},
        {"title": "Gladiator II", "poster": "https://image.tmdb.org/t/p/w500/2cxhvwyEwRlysAmRH4iodkvo0z5.jpg"},
        {"title": "Moana 2", "poster": "https://image.tmdb.org/t/p/w500/aLVkiINlIeCkcZIzb7XHzPYgO6L.jpg"},
        {"title": "Mufasa: The Lion King", "poster": "https://image.tmdb.org/t/p/w500/lurEK87kukWNaHd0zYnsi3yzJrs.jpg"}
    ]
    return render_template('home.html', movies=movies)

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    movie = request.form['movie']

    # --- 🧹 Step 1: basic cleanup and minimum text length check ---
    cleaned_review = review.strip()
    if len(cleaned_review.split()) < 3:
        return render_template(
            'home.html',
            rating="Please write a bit more text for an accurate prediction.",
            movie=movie,
            movies=[
                {"title": "Deadpool & Wolverine", "poster": "https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4t6EDMubfO.jpg"},
                {"title": "Gladiator II", "poster": "https://image.tmdb.org/t/p/w500/2cxhvwyEwRlysAmRH4iodkvo0z5.jpg"},
                {"title": "Moana 2", "poster": "https://image.tmdb.org/t/p/w500/aLVkiINlIeCkcZIzb7XHzPYgO6L.jpg"},
                {"title": "Mufasa: The Lion King", "poster": "https://image.tmdb.org/t/p/w500/lurEK87kukWNaHd0zYnsi3yzJrs.jpg"}
            ]
        )

    # --- 🧩 Step 2: check if model recognizes any words ---
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    vectorizer = None
    for name, step in word_model.named_steps.items():
        if isinstance(step, (TfidfVectorizer, CountVectorizer)):
            vectorizer = step
            break

    if vectorizer is not None:
        vec = vectorizer.transform([cleaned_review])
        if vec.nnz == 0:
            return render_template(
                'home.html',
                rating="No known words found — please try a longer or clearer review.",
                movie=movie,
                movies=[
                    {"title": "Deadpool & Wolverine", "poster": "https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4t6EDMubfO.jpg"},
                    {"title": "Gladiator II", "poster": "https://image.tmdb.org/t/p/w500/2cxhvwyEwRlysAmRH4iodkvo0z5.jpg"},
                    {"title": "Moana 2", "poster": "https://image.tmdb.org/t/p/w500/aLVkiINlIeCkcZIzb7XHzPYgO6L.jpg"},
                    {"title": "Mufasa: The Lion King", "poster": "https://image.tmdb.org/t/p/w500/lurEK87kukWNaHd0zYnsi3yzJrs.jpg"}
                ]
            )

    # --- ✅ Normal prediction flow ---
    pred_rating = predict_ensemble([cleaned_review])[0]
    pred_rating_rounded = int(np.clip(round(pred_rating), 1, 10))

    print("INPUT LEN:", len(cleaned_review), "SAMPLE:", cleaned_review[:120])
    p1 = word_model.predict([cleaned_review])[0]
    p2 = char_model.predict([cleaned_review])[0]
    print("RAW PRED word/char:", p1, p2, "AVG:", (p1 + p2) / 2)

    return render_template(
        'home.html',
        movies=[
            {"title": "Deadpool & Wolverine", "poster": "https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4t6EDMubfO.jpg"},
            {"title": "Gladiator II", "poster": "https://image.tmdb.org/t/p/w500/2cxhvwyEwRlysAmRH4iodkvo0z5.jpg"},
            {"title": "Moana 2", "poster": "https://image.tmdb.org/t/p/w500/aLVkiINlIeCkcZIzb7XHzPYgO6L.jpg"},
            {"title": "Mufasa: The Lion King", "poster": "https://image.tmdb.org/t/p/w500/lurEK87kukWNaHd0zYnsi3yzJrs.jpg"}
        ],
        rating=pred_rating_rounded,
        movie=movie
    )
if __name__ == '__main__':
    app.run(debug=True)
