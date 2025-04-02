from dotenv import load_dotenv
import tweepy
import os

load_dotenv()

BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
if BEARER_TOKEN is None:
    raise ValueError("TWITTER_BEARER_TOKEN is not set. Check your .env file.")

client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets():
    query = "earthquake OR flood OR fire -is:retweet lang:en"
    tweets = client.search_recent_tweets(query=query, max_results=10)

    if tweets.data is None: 
        return []

    return [tweet.text for tweet in tweets.data] 
