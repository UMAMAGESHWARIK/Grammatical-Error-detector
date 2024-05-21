from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from joblib import load

app = Flask(__name__)

# Load the trained model and vectorizer
loaded_model = load('grammatical_error_detector.joblib')

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']
    # Preprocess the input sentence (vectorize)
    vectorizer = loaded_model['vectorizer']
    sentence_vectorized = vectorizer.transform([sentence])
    # Predict the label
    prediction = loaded_model['model'].predict(sentence_vectorized)[0]
    # Convert label to human-readable form
    prediction_label = "Grammatically correct" if prediction == 0 else "Grammatically incorrect"
    return render_template('result.html', sentence=sentence, prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
