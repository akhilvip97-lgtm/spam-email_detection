from flask import Flask, request, jsonify,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

app = Flask(__name__)

data = pd.read_csv("spam.csv")

emails = data["message"]
labels = data["label"].map({"ham":0, "spam":1})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)
@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return jsonify({"prediction": int(result)})

if __name__ == "__main__":
    app.run(debug=True)