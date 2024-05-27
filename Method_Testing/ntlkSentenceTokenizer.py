import nltk
import spacy
import re
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')


def extract_text_from_pdf_file(pdf_file: str) -> str:
    with open(pdf_file, 'rb') as pdf:
        reader = PdfReader(pdf_file)
        pdf_text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                pdf_text.append(content)
        return pdf_text


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.!?]', '', text)
    return text


def fix_common_issues(text):
    text = re.sub(r'(\w)\.([A-Z])', r'\1. \2', text)
    return text


def tokenize_sentences(text):
    sentences = []

    current_sentences = sent_tokenize(text)
    sentences.extend(current_sentences)
    return sentences


def merge_short_sentences(sentences, min_length=5):
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


# Example usage
pdf_file = 'E:/studies/USJ FOT/lecture/Research/CodeBase/Textextraction/PDFs/Annual Report/6_Chapter_02.pdf'
texts = extract_text_from_pdf_file(pdf_file)
for text in texts:
    cleaned_text = clean_text(text)
    fixed_text = fix_common_issues(cleaned_text)
    sentences = tokenize_sentences(fixed_text)
    final_sentences = merge_short_sentences(sentences)

    print(final_sentences)

