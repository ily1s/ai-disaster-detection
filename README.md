#  AI-Powered Disaster Detection from Social Media

##  Introduction
Disasters strike unexpectedly, and early detection is crucial to saving lives. Our **AI-driven disaster detection system** leverages **social media data** (Twitter) to identify crisis events in real time. By using **NLP and machine learning**, we provide rapid alerts to emergency responders, governments, and humanitarian organizations.

##  Problem Statement
Traditional disaster detection relies on **news media and official reports**, often leading to delays. Meanwhile, social media platforms **capture events as they happen**. However, distinguishing real disasters from fake news and irrelevant posts is a challenge. We solve this by using **AI-powered filtering and classification**.

##  The Data
- **Source:** Twitter API (real-time tweets)  
- **Dataset:** Labeled tweets from past disasters (earthquakes, floods, fires,bombing,explosion,huricane)  
- **Features:** Text content, geolocation, timestamps  

##  Solution Approach
1. **Data Collection** – Stream tweets using **Twitter API**, filtering disaster-related keywords.  
2. **Text Preprocessing** – Remove noise, extract relevant terms, and detect locations.  
3. **AI Model** – Fine-tuned **BERT model** to classify tweets as **disaster or non-disaster**.  
4. **Event Detection** – Analyze tweet frequency and patterns for real-time alerts.  
5. **Visualization Dashboard** – Display detected disasters on an interactive map.  

##  Results & Impact
 <li>**High accuracy disaster classification using fine-tuned BERT**  </li>
<li> **Fast response time** – Alerts generated within minutes of an event  </li>
<li>**Interactive map** – Visual representation of disaster locations </li>

##  Tech Stack
- **Python, Flask** – Backend for API & data processing  
- **Tweepy** – Twitter API for real-time streaming  
- **Hugging Face Transformers** – BERT for disaster classification  
- **Scikit-Learn, Pandas** – Data processing & machine learning  
- **Streamlit / Dash** – Interactive UI for alerts  

##  Installation & Usage
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/AI-Disaster-Detection.git
   cd AI-Disaster-Detection
