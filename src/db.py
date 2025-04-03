import sqlite3

# Connect to SQLite
conn = sqlite3.connect("tweets.db")
cursor = conn.cursor()

# Create table for tweets (if not exists)
cursor.execute("""
CREATE TABLE IF NOT EXISTS disaster_tweets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id TEXT UNIQUE,
    tweet_text TEXT,
    disaster_type TEXT DEFAULT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
