import re
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK resources (first time only)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define preprocessing steps
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    cleaning_pattern = r'[^a-zA-Z\s]'
    clean = re.sub(cleaning_pattern, '', text)
    clean = clean.replace('\n', ' ')
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

def preprocess_text(text):
    # Step 1: Clean and lowercase
    cleaned = clean_text(text.lower())

    # Step 2: Tokenize
    tokens = word_tokenize(cleaned)

    # Step 3: Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Step 4: Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Step 5: Join tokens back to string
    return ' '.join(lemmatized_tokens)

# Load the saved pipeline
pipeline_path = r"sentences_passive_aggressive_pipeline.pkl"
model_pipeline = joblib.load(pipeline_path)

def predict_input(sentence):
    preprocessed = preprocess_text(sentence)
    prediction = model_pipeline.predict([preprocessed])[0]
    probability = model_pipeline.decision_function([preprocessed])  # raw scores
    return prediction, probability

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a sentence to classify: ")
    label, score = predict_input(user_input)
    print(f"\nPrediction: {label}")
    print(f"Confidence Score: {score}")
