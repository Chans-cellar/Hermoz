from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Load the FinBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

# Create a sentiment analysis pipeline
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Sample economic texts
texts = [
    "The GDP growth rate has significantly improved this quarter.",
    # "Unemployment rates have been steadily increasing over the past few months.",
    # "The new fiscal policy is expected to boost economic growth.",
    # "Inflation rates are at an all-time high, causing concern among economists."
]

# Generate sentiment scores
results = nlp(texts)

# Print the results
for text, result in zip(texts, results):
    print(f"Text: {text}\nSentiment: {result['label']}, Score: {result['score']:.4f}\n")
