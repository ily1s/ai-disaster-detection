import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import spacy

# Load Pre-trained BERT Model for Disaster Classification
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME) # 2 classes (Disaster/Non-Disaster)

# Load spaCy NER Model for Location & Disaster Type Extraction
nlp = spacy.load("en_core_web_sm")

def preprocess_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    return text.strip()

def classify_tweet(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Disaster" if prediction == 1 else "Non-Disaster"

def extract_info(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]  # GPE = Geopolitical Entity
    disaster_types = [ent.text for ent in doc.ents if ent.label_ in ["EVENT", "ORG"]]  # Extract disaster events
    return {
        "locations": locations if locations else "No location detected",
        "disaster_type": disaster_types if disaster_types else "No disaster type detected"
    }

def analyze_tweet(tweet):
    cleaned_tweet = preprocess_tweet(tweet)
    classification = classify_tweet(cleaned_tweet)
    extracted_info = extract_info(cleaned_tweet) if classification == "Disaster" else {"locations": "N/A", "disaster_type": "N/A"}
    return {"classification": classification, **extracted_info}

# Example Tweet
test_tweet = "Breaking: A 6.5 earthquake hit Los Angeles! Stay safe!"
print(analyze_tweet(test_tweet))
