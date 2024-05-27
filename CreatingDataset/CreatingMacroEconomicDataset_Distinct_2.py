#THIS FILE CONTAINS MORE IMPROVED PREPROCESSING FOR TEXTS

from PyPDF2 import PdfReader
import os
import csv
import re
import nltk

import spacy

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

spacy_nlp = spacy.load('en_core_web_sm')


# Function to extract text from a PDF file
def extract_text_from_pdf_file(pdf_file: str) -> str:
    with open(pdf_file, 'rb') as pdf:
        reader = PdfReader(pdf_file)
        pdf_text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                pdf_text.append(content)
        return pdf_text

def tokenize_sentences(text):
    doc = spacy_nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def lemma_and_stopword_text(text):
    doc = spacy_nlp(text)
    # Remove stopwords and perform lemmatization
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(processed_tokens)


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
            preprocessed_txt = lemma_and_stopword_text(fixed_text)
            if len(fixed_text.split()) < 4:  # Ensure the cleaned sentence is also meaningful
                continue
            clean_sentences.append(preprocessed_txt)


    return clean_sentences


def add_to_csv(pdf_file_path: str, sentences):
    # Create a new CSV file with the same name as the PDF
    base_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
    csv_file_path = os.path.join(f'{base_name}.csv')

    # Check if the CSV file exists to decide whether to write headers
    write_header = not os.path.exists(csv_file_path)

    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['sentence', 'macroeconomic factor'])  # Write column headings

        for sentence in sentences:
            writer.writerow([sentence, ''])  # Leave macroeconomic factor column empty for manual classification


# Define the path to the PDF file
pdf_file = '../Textextraction/PDFs/Annual Report/6_Chapter_02.pdf'
sentences = preprocess_text(pdf_file)
add_to_csv(pdf_file,sentences)

# Call the function
