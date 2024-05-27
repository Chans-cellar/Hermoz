import requests
from bs4 import BeautifulSoup
import nltk
import csv
import os

nltk.download('punkt')  # Download the Punkt tokenizer models

def is_economic_related(text):
    keywords = ['economy', 'financial', 'market', 'inflation', 'GDP', 'unemployment', 'fiscal', 'monetary', 'trade', 'budget']
    return any(keyword in text.lower() for keyword in keywords)

# URL of the blog post you want to scrape
url = 'https://www.imf.org/en/Blogs/Articles/2024/04/23/latin-americas-shifting-demographics-could-undercut-growth'

# Send a GET request to the URL
response = requests.get(url)

# Initialize a BeautifulSoup object with the content of the response
soup = BeautifulSoup(response.content, 'html.parser')

# Assuming the main text of the blog is within <article> tags
article = soup.find('article')
if article:
    article_text = article.get_text(strip=True)
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(article_text)
    # Filter sentences that are related to economics
    economic_sentences = [sentence for sentence in sentences if is_economic_related(sentence)]
    
    # Define the full path where the CSV will be saved
    file_path = os.path.join('e:', os.sep, 'd', 'economic_sentences.csv')
    
    # Write economic-related sentences to a CSV file
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Economic Sentences'])
        for sentence in economic_sentences:
            writer.writerow([sentence])
        
        print(f"CSV file has been written to {file_path}")
else:
    print("No article found")
