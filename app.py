import pandas as pd

df = pd.read_csv('sentiment_analysis.csv')
print(df.head())
print(df.columns)

#Step-3:
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess_text)

#Step:4
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']  # Target labels

#Step:5
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Step:6
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
print(sentiment_pipeline("I love this project!"))

#Step:7
from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

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

