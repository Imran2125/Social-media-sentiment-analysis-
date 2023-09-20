from flask import Flask, render_template, request
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import joblib

# Download stopwords if not already downloaded
nltk.download('stopwords')

app = Flask(__name__)


# Define the getCleanedText function here if it's not defined in a separate module
def getCleanedText(text):
    tokenizer = RegexpTokenizer(r"\w+")
    en_stopwords = set(stopwords.words('english'))
    ps = PorterStemmer()
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    clean_text = " ".join(stemmed_tokens)
    return clean_text


# Load your trained model (mn) and vectorizer (cv) here if not already loaded
cv = joblib.load('count_vectorizer.pkl')
mn = joblib.load('multinomial_nb.pkl')


# cv = CountVectorizer(ngram_range=(1, 2))
# mn = MultinomialNB()
# Assuming you have already trained and saved the model and vectorizer
# cv.fit(X_clean)
# mn.fit(X_vec, y_train)

@app.route('/')
def index():
    return render_template('index.html')  # Display the form page


@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['text']  # Get user's input from the form
    cleaned_input = getCleanedText(user_input)  # Preprocess the input
    vectorized_input = cv.transform([cleaned_input]).toarray()  # Vectorize the input
    sentiment_prediction = mn.predict(vectorized_input)[0]
    sentiment_score = 1 if sentiment_prediction == "positive" else 0
    sentiment_category = "Positive" if sentiment_prediction == "positive" else "Negative"

    return render_template('results.html', sentiment_score=sentiment_score, sentiment_category=sentiment_category)


if __name__ == '__main__':
    app.run(debug=True)
