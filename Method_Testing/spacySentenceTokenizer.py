import spacy
import re
from PyPDF2 import PdfReader

# Load SpaCy's English model
nlp = spacy.load('en_core_web_sm')
def extract_text_from_pdf_file(pdf_file: str) -> str:
    with open(pdf_file, 'rb') as pdf:
        reader = PdfReader(pdf_file)
        pdf_text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                pdf_text.append(content)
        return pdf_text

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


def fix_common_issues(text):
    text = re.sub(r'(\w)\.([A-Z])', r'\1. \2', text)
    return text

def lemma_and_stopword_text(text):
    doc = nlp(text)
    # Remove stopwords and perform lemmatization
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(processed_tokens)
def tokenize_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences



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
# texts = extract_text_from_pdf_file(pdf_file)
texts = ['Inflation reached a historic high in September 2022 but commenced a descending path thereafter due to tight monetary conditions.',
'The annual unemployment rate declined to 4.7 percent in 2022, compared to 5.1 percent in 2021.',
'The real GDP contracted by 7.8 percent in 2022, marking the deepest economic contraction since independence.',
'The Government has implemented strong revenue enhancement measures, such as increasing the VAT and personal income tax rates.',
'Monetary policy tightening was implemented through significant increases in policy interest rates to contain rising inflationary pressures.',
'Export sector policies need to focus on increasing domestic value addition to strengthen export competitiveness.',
'The Sri Lanka rupee depreciated substantially, necessitating a measured adjustment in the exchange rate by the Monetary Board.']
# print(texts)
for text in texts:
    cleaned_text = clean_text(text)
    fixed_text = fix_common_issues(cleaned_text)
    preprocessed_txt = lemma_and_stopword_text(fixed_text)
    print(preprocessed_txt + ',')
# for text in texts:
#     sentences = tokenize_sentences(text)
#     final_sentences = merge_short_sentences(sentences)
#     for sentence in final_sentences:
#         cleaned_text = clean_text(sentence)
#         fixed_text = fix_common_issues(cleaned_text)
#         preprocessed_txt = lemma_and_stopword_text(fixed_text)
#         print(sentence + ',')






