import pandas as pd
from flask import Flask, request, render_template
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('sentiment_analysis.csv')

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment_label = None
    sentiment_score = None
    text = None

    if request.method == "POST":
        text = request.form.get("text")
        if text:
            prediction = sentiment_pipeline(text)
            sentiment_label = prediction[0]["label"]
            sentiment_score = round(prediction[0]["score"], 2)

    return render_template("index.html", text=text, sentiment=sentiment_label, score=sentiment_score)

if __name__ == "__main__":
    app.run(debug=True)