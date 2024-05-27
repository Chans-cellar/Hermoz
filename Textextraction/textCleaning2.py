import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data (if not already done)
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean text data
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

# Directory where your .txt files are stored
input_directory = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Textextraction\\txt'
# Directory where cleaned files will be saved
output_directory ='E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Textextraction\\cleanedTxt'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over each file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_directory, filename)
        with open(input_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            cleaned_content = clean_text(content)
            # Construct the output file path
            output_file_path = os.path.join(output_directory, f'cleaned_{filename}')
            # Write the cleaned content to the output file
            with open(output_file_path, 'w', encoding='utf-8') as cleaned_file:
                cleaned_file.write(cleaned_content)
                print(f"Cleaned content of {filename} saved to {output_file_path}")
