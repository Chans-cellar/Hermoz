import nltk
nltk.download('punkt')

from PyPDF2 import PdfReader
import os
import csv
from nltk.tokenize import sent_tokenize


# Function to extract text from a PDF file
def extract_text_from_pdf_file(Pdf_file: str) -> [str]:
    with open(Pdf_file, 'rb') as pdf:
        reader = PdfReader(pdf)
        pdf_text = []

        for page in reader.pages:
            content = page.extract_text()
            if content:  # Check if content is not None
                pdf_text.append(content)

        return pdf_text


# Function to process PDF and append sentences to a CSV file
def pdf_to_sentences_csv(Pdf_file: str, csv_file: str):
    # Extract text from PDF
    extracted_txt = extract_text_from_pdf_file(Pdf_file)

    # Initialize a list to hold all sentences
    sentences = []

    # Convert extracted text to sentences
    for text in extracted_txt:
        # Use sent_tokenize to split the text into sentences
        current_sentences = sent_tokenize(text)
        sentences.extend(current_sentences)

    # Append sentences to the existing CSV file
    with open(csv_file, 'a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        for sentence in sentences:
            writer.writerow([sentence, ''])  # Leave macroeconomic factor column empty for manual classification


# Define the paths
pdf_file = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Textextraction\\PDFs\\press_20240123_Monetary_Policy_Review_No_1_2024_e_Ge643v.pdf'
csv_file = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\CreatingDataset\\text classification-macroeconomic.csv'

# Call the function
pdf_to_sentences_csv(pdf_file, csv_file)
