from flask import Flask, request, render_template, jsonify

from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, pipeline
import torch
import os
import re
import nltk
from PyPDF2 import PdfReader
import string
from werkzeug.utils import secure_filename
import sqlite3
import spacy

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load SpaCy's English model
spacy_nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_directory = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Classifier Model\\macroecon_classifier'
tokenizer_directory = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Classifier Model\\tokenizer_model'
tokenizer = BertTokenizer.from_pretrained(tokenizer_directory)
model = BertForSequenceClassification.from_pretrained(model_directory)

# Load sentiment analysis model
sentiment_model_directory = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Sentiment Model\\sentiment_v1'
sentiment_tokenizer_directory = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Sentiment Model\\sentiment_v1_tokenizer'
sentiment_tokenizer = RobertaTokenizer.from_pretrained(sentiment_tokenizer_directory)
sentiment_model = RobertaForSequenceClassification.from_pretrained(sentiment_model_directory)

# Load finBERT model
finbert_tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
finbert_model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
finbert_nlp = pipeline('sentiment-analysis', model=finbert_model, tokenizer=finbert_tokenizer)

# Load BART summarizer model and tokenizer
summarizer_model_name = "facebook/bart-large-cnn"
summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_model_name)
summarizer_tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)


def get_db_connection():
    conn = sqlite3.connect('predictions.db')
    return conn


def extract_text_from_pdf_file(pdf_file: str) -> str:
    with open(pdf_file, 'rb') as pdf:
        reader = PdfReader(pdf_file)
        pdf_text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                pdf_text.append(content)
        return pdf_text


def lemma_and_stopword_text(text):
    doc = spacy_nlp(text)
    # Remove stopwords and perform lemmatization
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(processed_tokens)


def tokenize_sentences(text):
    doc = spacy_nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def merge_short_sentences(sentences, min_length=4):
    merged_sentences = []
    buffer = ''
    for sentence in sentences:
        if len(sentence.split()) < min_length:
            buffer += ' ' + sentence
        else:
            if buffer:
                merged_sentences.append(buffer.strip())
                buffer = ''
            merged_sentences.append(sentence.strip())
    if buffer:
        merged_sentences.append(buffer.strip())
    return merged_sentences


def remove_headers_footers(text: str) -> str:
    patterns = [
        r'Page \d+',
        r'\bChapter \d+\b',
        r'\bTable of Contents\b',
        r'\n+'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    text = remove_headers_footers(text)
    # text = remove_tables(text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.lower()
    return text


def fix_common_issues(text):
    text = re.sub(r'(\w)\.([A-Z])', r'\1. \2', text)
    return text


def preprocess_text(pdf_file_path: str):
    raw_texts = extract_text_from_pdf_file(pdf_file_path)
    clean_sentences = []
    for text in raw_texts:
        dirt_sentences = tokenize_sentences(text)
        final_sentences = merge_short_sentences(dirt_sentences)
        for sentence in final_sentences:
            if len(sentence.split()) < 4:  # Skip sentences with less than 3 words
                continue
            cleaned_text = clean_text(sentence)
            fixed_text = fix_common_issues(cleaned_text)
            # Sentiment analysis before removing stopwords and lemmatization
            no_lemma_txt = fixed_text
            preprocessed_txt = lemma_and_stopword_text(fixed_text)
            if len(preprocessed_txt.split()) < 4:  # Ensure the cleaned sentence is also meaningful
                continue
            clean_sentences.append((preprocessed_txt, no_lemma_txt))
    return clean_sentences


def convert_sentiment_scores(positive, neutral, negative):
    # Weights
    w_p = 1
    w_n = 0.5
    w_ng = -1

    # Compute weighted sum
    score = (w_p * positive) + (w_n * neutral) + (w_ng * negative)

    # Map to [0, 1] range
    normalized_score = (score + 1) / 2

    return normalized_score


def predict_finbert_sentiment(text):
    # Tokenize the texts
    inputs = finbert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Get the raw scores (logits) from the model
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        logits = outputs.logits

    # Convert logits to sentiment scores
    scores = torch.softmax(logits, dim=1)
    fair_score = convert_sentiment_scores(scores[0][0].item(), scores[0][2].item(), scores[0][1].item())
    return fair_score


def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_score = scores[:, 2].item() - scores[:, 0].item()  # Positive score minus Negative score
    return sentiment_score


def predict_macroecon(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()
    labels = ['Inflation', 'International Trade', 'GDP Growth', 'Exchange Rates', 'Monetary Policy', 'Fiscal Policy',
              'Unemployment']
    macroecon_label = labels[predicted_class]
    return macroecon_label


def store_prediction(sentence, label, sentiment_score, year, document_name):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('INSERT INTO predictions (sentence, label, sentiment_score, year, document_name) VALUES (?, ?, ?, ?, ?)',
              (sentence, label, sentiment_score, year, document_name))
    conn.commit()
    conn.close()


def get_label_counts():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT label, AVG(sentiment_score) as avg_sentiment, COUNT(*) as count FROM predictions GROUP BY label')
    label_counts = c.fetchall()
    conn.close()
    return {label: {'count': count, 'avg_sentiment': avg_sentiment} for label, avg_sentiment, count in label_counts}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/store_data', methods=['POST'])
def store_data():
    if 'file' not in request.files or 'year' not in request.form:
        return jsonify({'error': 'File or year not provided'})
    file = request.files['file']
    year = request.form['year']
    if file.filename == '' or year == '':
        return jsonify({'error': 'File or year not selected'})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        sentences = preprocess_text(filepath)
        total_sentences = len(sentences)
        for i, (preprocessed_txt, no_lemma_txt) in enumerate(sentences):
            macroecon_label = predict_macroecon(preprocessed_txt)
            sentiment_score = predict_finbert_sentiment(no_lemma_txt)
            store_prediction(no_lemma_txt, macroecon_label, sentiment_score, year, filename)
            progress = (i + 1) / total_sentences * 100
            print(f'Processing: {progress:.2f}%')
        return jsonify({'status': 'Data stored successfully'})


@app.route('/get_data', methods=['GET'])
def get_data():
    label_counts = get_label_counts()
    return jsonify({'results': label_counts})


@app.route('/get_summary', methods=['POST'])
def get_summary():
    data = request.get_json()
    year = data['year']
    factor = data['factor']

    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT sentence FROM predictions WHERE year=? AND label=?', (year, factor))
    sentences = c.fetchall()
    conn.close()

    combined_text = '. '.join([sentence[0] for sentence in sentences])
    inputs = summarizer_tokenizer.encode("summarize: " + combined_text, return_tensors="pt", max_length=1024,
                                         truncation=True)
    summary_ids = summarizer_model.generate(inputs, num_beams=4, min_length=50, max_length=100, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({'summary': summary})


if __name__ == '__main__':
    app.run(debug=True)
