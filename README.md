# 509-Prediction-Review-Final_Project

# 🎬 AI Movie Review Rating Predictor  

[![Streamlit App](https://img.shields.io/badge/Live%20App-Visit%20Now-brightgreen?logo=streamlit)](https://509-final-project.streamlit.app/)  
Predict IMDb-style movie ratings (1–10⭐) from written reviews using Natural Language Processing (NLP) and Ensemble Machine Learning models.  

---

## 🚀 Overview  

This project analyzes thousands of real movie reviews from **The Movie Database (TMDb)** and predicts a numerical movie rating from 1 to 10 based solely on the written text.  

It uses **two powerful text-based models** (word-level and character-level regressors) and combines them into an **ensemble predictor** to capture both vocabulary semantics and emotional tone.  

You can try it live here 👉 **[509-Final-Project Streamlit App](https://509-final-project.streamlit.app/)**  

---

## 🧠 Model Architecture  

| Component | Description |
|------------|--------------|
| **Word-level Model** | Uses TF-IDF features to capture word importance and context. |
| **Character-level Model** | Focuses on subword patterns, spelling, and punctuation sentiment. |
| **Ensemble** | Averages both models’ predictions for smoother, more robust ratings. |

The model outputs a **continuous score** between 1 and 10, then rounds to the nearest whole rating.  

---

## 📊 Example Predictions  

| Review | Predicted Rating |
|--------|------------------|
| “This was the worst movie I've ever seen. Terrible acting.” | ⭐ 4 / 10 |
| “Absolutely amazing! The visuals and story kept me hooked.” | ⭐ 9 / 10 |
| “It was okay, not great but not terrible either.” | ⭐ 3 / 10 |
| “Masterpiece. Brilliant acting, stunning direction.” | ⭐ 8 / 10 |

These results show the model successfully distinguishing between **negative, neutral, and positive sentiment** — mapping tone to numerical scores realistically.

---

## 🧰 Tech Stack  

- **Python 3.12**  
- **Scikit-Learn** – TF-IDF vectorization & SVR regression  
- **NumPy / Pandas** – data cleaning & feature prep  
- **Flask** – local web prototype  
- **Streamlit** – final interactive web deployment  
- **Bootstrap CSS** – styling for Flask version  

---

## 🧩 Files Overview  

| File | Purpose |
|------|----------|
| `Final Project.ipynb` | Main notebook for training and evaluation |
| `ensemble_models.pkl` | Serialized trained models |
| `tmdb_reviews.csv` | Raw movie review dataset |
| `cleaned_tmdb_reviews.csv` | Preprocessed dataset |
| `app.py` | Flask web app version |
| `streamlit_app.py` | Streamlit web app version |
| `requirements.txt` | Dependencies list for environment setup |

---

## ⚙️ Run Locally (Flask Version)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/gerardcf1/509-Prediction-Review-Final_Project.git
cd 509-Prediction-Review-Final_Project
```

2️⃣ Install Dependencies:
```bash
pip install -r requirements.txt
```
3️⃣ Run the Flask App:
```bash
python app.py
```

Then open your browser and go to http://127.0.0.1:5000/

🌐 Run via Streamlit (Recommended)
1️⃣ Launch Locally
```bash
streamlit run streamlit_app.py
```
2️⃣ Or Try It Online
```bash
🚀 https://509-final-project.streamlit.app/
```
🧩 How It Works

The user selects a movie and writes a short review.

The app cleans and vectorizes the text (TF-IDF / char-ngrams).

Both models predict a numeric rating.

Their results are averaged → clipped to a 1-10 range → displayed with stars.
