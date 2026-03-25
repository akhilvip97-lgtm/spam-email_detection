from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import re

app = Flask(__name__)

# =========================
# LOAD DATA (FIXED)
# =========================
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

data['message'] = data['message'].apply(clean_text)

# =========================
# PREPARE DATA
# =========================
emails = data["message"]
labels = data["label"].map({"ham": 0, "spam": 1})

# =========================
# VECTORIZE + TRAIN
# =========================
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    result = model.predict(vec)[0]
    
    return jsonify({"prediction": int(result)})

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)