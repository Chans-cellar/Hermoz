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


# Function to process PDF and create a new CSV file with sentences
def pdf_to_sentences_csv(Pdf_file: str):
    # Extract text from PDF
    extracted_txt = extract_text_from_pdf_file(Pdf_file)

    # Initialize a list to hold all sentences
    sentences = []

    # Convert extracted text to sentences
    for text in extracted_txt:
        # Use sent_tokenize to split the text into sentences
        current_sentences = sent_tokenize(text)
        sentences.extend(current_sentences)

    # Create a new CSV file with the same name as the PDF
    base_name = os.path.splitext(os.path.basename(Pdf_file))[0]
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

# Call the function
pdf_to_sentences_csv(pdf_file)
