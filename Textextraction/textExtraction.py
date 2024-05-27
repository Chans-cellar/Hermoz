# IN THIS FILE WE ARE MANUALLY PROVIDING THE OUTPUT TEXT FILE

from PyPDF2 import PdfReader
import re

def extract_text_from_pdf_file(Pdf_file:str) -> [str]:
    with open(Pdf_file, 'rb') as pdf:
        reader = PdfReader(Pdf_file)
        pdf_text = []

        for page in reader.pages:
            content = page.extract_text()
            pdf_text.append(content)
        
        return pdf_text



file = 'PDFs\monetary_policy_report_2023_july_e.pdf'
txt_file = open('txt\monetary_policy_report_2023_july_e.txt', 'w', encoding="utf-8")

extracted_txt = extract_text_from_pdf_file(file)
# print(extracted_txt)
for text in extracted_txt:
    split_message = re.split(r'\s+|[,;?!.-]\s*',text.lower())
    txt_file.write(text+'\n')
    # print(split_message)
txt_file.close()