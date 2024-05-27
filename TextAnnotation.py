import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load English language model for spaCy
nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file"""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    return text

def extract_text_from_txt(file_path):
    """Read text from a text file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def tokenize_text(text):
    """Tokenize text and remove stopwords"""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens

def get_top_keywords(text, n=10):
    """Extract top keywords using TF-IDF or RAKE"""
    # Use TF-IDF for keyword extraction
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_keywords_idx = tfidf_matrix[0].toarray().argsort()[0][-n:][::-1]
    top_keywords = [feature_names[idx] for idx in top_keywords_idx]
    return top_keywords

def get_named_entities(text):
    """Extract named entities using spaCy"""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE']]
    return entities

def get_phrases(text):
    """Extract noun phrases using spaCy"""
    doc = nlp(text)
    matcher = PhraseMatcher(nlp.vocab)
    noun_phrases = [doc[start:end].text for _, start, end in matcher(nlp(text))]
    return noun_phrases

def main(file_path):
    # Extract text from PDF
    text = extract_text_from_txt(file_path)
    
    # Tokenize text
    tokens = tokenize_text(text)
    
    # Keyword extraction using TF-IDF
    top_keywords_tfidf = get_top_keywords(' '.join(tokens))
    
    # Keyword extraction using RAKE
    rake = Rake()
    rake.extract_keywords_from_text(text)
    top_keywords_rake = rake.get_ranked_phrases()[:10]
    
    # Named entity extraction using spaCy
    named_entities = get_named_entities(text)
    
    # Noun phrase extraction using spaCy
    noun_phrases = get_phrases(text)
    
    # Display results
    print('-------------------')
    print("Top Keywords (TF-IDF):", top_keywords_tfidf)
    print('-------------------')
    print("Top Keywords (RAKE):", top_keywords_rake)
    print('-------------------')
    print("Named Entities:", named_entities)
    print('-------------------')
    print("Noun Phrases:", noun_phrases)

if __name__ == "__main__":
    pdf_file_path = "Textextraction\\txt\\press_20240123_Monetary_Policy_Review_No_1_2024_e_Ge643v.txt"  # Change this to the path of your PDF file
    main(pdf_file_path)
