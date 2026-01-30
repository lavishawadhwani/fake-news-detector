from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# -----------------------
# App setup
# -----------------------

app = Flask(__name__)

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# -----------------------
# Load trained model
# -----------------------

vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
svd = pickle.load(open("model/svd.pkl", "rb"))
model = pickle.load(open("model/model.pkl", "rb"))

# -----------------------
# Text cleaning
# -----------------------

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------
# Routes
# -----------------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)

        vec = vectorizer.transform([cleaned])
        vec = svd.transform(vec)

        pred = model.predict(vec)[0]

        try:
            proba = model.predict_proba(vec)[0]
            confidence = round(np.max(proba) * 100, 2)
        except:
            confidence = None

        prediction = "REAL" if pred == "real" else "FAKE"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )
 
if __name__ == "__main__":
    app.run()

