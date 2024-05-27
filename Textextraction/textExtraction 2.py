# IN THIS FILE IT AUTOMATICALLY CREATE THE OUTPUT TEXT FILE USING INPUT TEXT FILE NAME


from PyPDF2 import PdfReader
import re
import os

def extract_text_from_pdf_file(Pdf_file: str) -> [str]:
    with open(Pdf_file, 'rb') as pdf:
        reader = PdfReader(Pdf_file)
        pdf_text = []

        for page in reader.pages:
            content = page.extract_text()
            pdf_text.append(content)
        
        return pdf_text

def pdf_to_text(Pdf_file: str):
    # Extract text from PDF
    extracted_txt = extract_text_from_pdf_file(Pdf_file)
    
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(Pdf_file))[0]
    output_txt_file = f'Textextraction\\txt\\{filename}.txt'
    

    # Write extracted text to a text file
    with open(output_txt_file, 'w', encoding="utf-8") as txt_file:
        for text in extracted_txt:
            txt_file.write(text + '\n')

file = 'Textextraction\PDFs\press_20231124_Monetary_Policy_Review_No_8_2023_e_Jw52vr.pdf'
pdf_to_text(file)
