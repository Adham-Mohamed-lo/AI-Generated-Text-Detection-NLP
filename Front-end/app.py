from flask import Flask, request, jsonify, render_template
import re
import joblib
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
import langid
from nltk.stem import WordNetLemmatizer, ISRIStemmer

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

models = {
    'en': joblib.load("essay_passive_aggressive_pipeline.pkl"),
    'ar': joblib.load("arabic_passive_aggressive_pipeline.pkl")
}

english_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

arabic_tokenizer = RegexpTokenizer(r'[\u0600-\u06FF]+')
arabic_stopwords = set(stopwords.words('arabic'))
arabic_stemmer = ISRIStemmer()

@app.route('/')
def home():
    return render_template('index.html') 

def clean_arabic_text(text):
    """Clean Arabic text following training pipeline"""
    text = re.sub(r'[^\u0621-\u064A\s]', '', text) 
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_arabic(text):
    """Complete Arabic preprocessing: clean, tokenize, remove stopwords, stem"""
    text = clean_arabic_text(text)
    tokens = arabic_tokenizer.tokenize(text)
    filtered_tokens = [arabic_stemmer.stem(word) for word in tokens if word not in arabic_stopwords]
    return ' '.join(filtered_tokens)

def clean_english_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text).replace('\n', ' ').strip()

def preprocess_english(text):
    text = clean_english_text(text.lower())
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in english_stopwords]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return ' '.join(lemmatized)

def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

def chunk_text(original_text, chunk_size=20):
    words = original_text.split()
    chunks = []
    positions = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)
            positions.append((i, i + len(chunk)))
    return chunks, positions

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    language = data.get('language', 'en')

    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    detected_lang = detect_language(text)
    if language not in ['en', 'ar']:
        return jsonify({'error': 'Invalid language selection'}), 400
    if language != detected_lang:
        return jsonify({'error': f'The text appears to be in {detected_lang}, not {language}.'}), 400
    if len(text.split()) < 80:
        return jsonify({'error': 'Text must be at least 80 words.'}), 400

    model = models.get(language)
    chunks, positions = chunk_text(text, chunk_size=20)

    if language == 'en':
        processed_chunks = [preprocess_english(chunk) for chunk in chunks]
    else:
        processed_chunks = [preprocess_arabic(chunk) for chunk in chunks]

    predictions = model.predict(processed_chunks)

    words = text.split()
    highlighted_words = words[:]
    ai_count = 0

    for pred, (start, end) in zip(predictions, positions):
        if pred == 1:
            ai_count += 1
            for i in range(start, end):
                highlighted_words[i] = f"<span class='bg-yellow-300 px-1 rounded'>{highlighted_words[i]}</span>"

    highlighted_text = ' '.join(highlighted_words)
    ai_percentage = round((ai_count / len(chunks)) * 100, 2)

    return jsonify({
        'ai_percentage': ai_percentage,
        'highlighted_text': highlighted_text,
        'detected_language': detected_lang
    })

if __name__ == '__main__':
    app.run(debug=True)
