import streamlit as st
from data_collection import fetch_tweets

st.title("AI-Powered Disaster Detection")

# Show real-time tweet updates
if st.button("Fetch Latest Tweets"):
    tweets = fetch_tweets()
    for tweet in tweets:
        st.write(tweet)