import tensorflow as tf
import numpy as np
import sqlite3
import pickle
from keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.models import model_from_json # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Load model architecture from JSON (since it's actually a JSON file)
with open("/Users/ilyas/myenv/YARBI/AI4SDG/models/model_architecture.json", "r") as json_file:
    model_json = json_file.read()

# Reconstruct model
model = model_from_json(model_json)

# Load tokenizer
with open("/Users/ilyas/myenv/YARBI/AI4SDG/models/tokenizer.joblib", "rb") as handle:
    tokenizer = pickle.load(handle)

# Disaster categories
disaster_labels = ["earthquake", "explosion", "bombing", "floods", "hurricane", "tornado", "unrelated"]

def preprocess_text(tweet):
    """Tokenize and pad tweet text."""
    sequence = tokenizer.texts_to_sequences([tweet])
    padded = pad_sequences(sequence, maxlen=100)  # Ensure same length as training
    return padded

def classify_disaster(tweet):
    """Predict disaster type."""
    processed_tweet = preprocess_text(tweet)
    prediction = model.predict(processed_tweet)
    predicted_label = np.argmax(prediction)  # Get category index
    return disaster_labels[predicted_label]

def classify_and_update_tweets():
    """Classifies tweets and updates the database."""
    conn = sqlite3.connect("tweets.db")
    cursor = conn.cursor()

    # Fetch tweets that have not been classified
    cursor.execute("SELECT tweet_id, tweet_text FROM disaster_tweets WHERE disaster_type IS NULL")
    tweets = cursor.fetchall()

    for tweet_id, tweet_text in tweets:
        disaster_type = classify_disaster(tweet_text)
        
        # Update database with prediction
        cursor.execute("UPDATE disaster_tweets SET disaster_type = ? WHERE tweet_id = ?", (disaster_type, tweet_id))

    conn.commit()
    conn.close()
    print("✅ Tweets classified and updated in the database!")

# Run classification
classify_and_update_tweets()
