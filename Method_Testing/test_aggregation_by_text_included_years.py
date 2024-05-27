from PyPDF2 import PdfReader
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import datefinder
from collections import defaultdict

# Download necessary NLTK data (if not already done)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def extract_text_from_pdf_file(pdf_file: str) -> [str]:
    with open(pdf_file, 'rb') as pdf:
        reader = PdfReader(pdf_file)
        pdf_text = []

        for page in reader.pages:
            content = page.extract_text()
            pdf_text.append(content)

        return pdf_text

def clean_text(text):
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)
    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Rejoin tokens into a single string
    text = ' '.join(tokens)

    return text

def extract_dates(text):
    matches = datefinder.find_dates(text)
    dates = [match for match in matches]
    return dates

def associate_text_with_dates(text, dates):
    sentences = sent_tokenize(text)
    date_text_mapping = []
    current_date = None
    for sentence in sentences:
        found_dates = extract_dates(sentence)
        if found_dates:
            current_date = found_dates[0]
        if current_date:
            date_text_mapping.append((current_date.year, sentence))
    return date_text_mapping

def aggregate_by_date(date_text_mapping):
    aggregated_data = defaultdict(list)
    for date, text in date_text_mapping:
        cleaned_text = clean_text(text)
        aggregated_data[date].append(cleaned_text)
    return aggregated_data

def main():
    # Path to the PDF file
    pdf_file_path = '/Textextraction/PDFs/Annual Report/6_Chapter_02.pdf'  # Replace with your PDF file path

    # Step 1: Extract text from PDF
    pdf_text = extract_text_from_pdf_file(pdf_file_path)
    combined_text = " ".join(pdf_text)

    # Step 2: Extract dates from the text
    dates = extract_dates(combined_text)

    # Step 3: Associate text with dates
    date_text_mapping = associate_text_with_dates(combined_text, dates)

    # Step 4: Aggregate text by date
    aggregated_data = aggregate_by_date(date_text_mapping)

    # Step 5: Print aggregated sentences by year (only for years between 2018 and 2023)
    for year, texts in aggregated_data.items():
        if 2018 <= year <= 2023:
            print(f"Year: {year}")
            for text in texts:
                print(f" - {text}")

if __name__ == '__main__':
    main()
