import pandas as pd
import re
import spacy
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

spacy_nlp = spacy.load('en_core_web_sm')

def tokenize_sentences(text):
    doc = spacy_nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def lemma_and_stopword_text(text):
    doc = spacy_nlp(text)
    # Remove stopwords and perform lemmatization
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(processed_tokens)

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

def preprocess_text(sentence):
    if not isinstance(sentence, str):
        return ""
    cleaned_text = clean_text(sentence)
    fixed_text = fix_common_issues(cleaned_text)
    preprocessed_txt = lemma_and_stopword_text(fixed_text)
    return preprocessed_txt

# Read the dataset from the CSV file
input_csv = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\CreatingDataset\\macroeconomic_classifier_dataset_4_cleaned.csv'  # Replace with your input file path
output_csv = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\CreatingDataset\\cleaned\\acroeconomic_classifier_dataset_4_hypercleaned.csv'  # Replace with your desired output file path

df = pd.read_csv(input_csv)

# Apply preprocessing to each sentence in the 'sentence' column
df['sentence'] = df['sentence'].apply(preprocess_text)

# Remove rows where the sentence has less than 4 words
df = df[df['sentence'].apply(lambda x: len(x.split()) >= 4)]

# Save the modified dataset to a new CSV file
df.to_csv(output_csv, index=False)

print(f"Preprocessed dataset saved to {output_csv}")
