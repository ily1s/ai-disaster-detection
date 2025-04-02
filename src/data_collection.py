from dotenv import load_dotenv

import tweepy
import os

BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets():
    query = "earthquake OR flood OR fire -is:retweet lang:en"
    tweets = client.search_recent_tweets(query=query, max_results=10)

    for tweet in tweets.data:
        print(tweet.text)

