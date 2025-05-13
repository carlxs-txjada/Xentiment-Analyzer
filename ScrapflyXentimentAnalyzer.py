import os
import re
import json
import jmespath
import asyncio
import pickle
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
from scrapfly import ScrapeConfig, ScrapflyClient

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

with open("XentimentAnalyzer.pkl", "rb") as f:
    model, label_encoder = pickle.load(f)

SCRAPFLY = ScrapflyClient(key="SCRAPFLY-CLIENT-KEY")

def preprocess(text):
    text = re.sub(r"http\S+|@\w+|#\w+|[^A-Za-z\s]", '', text.lower())
    tokens = word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word not in stopWords])

def extractInfoFromUrl(url):
    path = urlparse(url).path.strip("/").split("/")
    user = path[0]
    tweetId = path[-1]
    timestamp = (int(tweetId) >> 22) + 1288834974657
    date = datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y%m%d')
    return user, tweetId, date

def parseTweet(data):
    result = jmespath.search(
        """{
            created_at: legacy.created_at,
            text: legacy.full_text,
            id: legacy.id_str,
            user: core.user_results.result.legacy.screen_name
        }""", data
    )
    return result

async def scrapeTweet(url: str):
    result = await SCRAPFLY.async_scrape(ScrapeConfig(
        url,
        render_js=True,
        wait_for_selector="[data-testid='tweet']"
    ))
    _XHR_Calls = result.scrape_result["browser_data"]["xhr_call"]
    tweetCall = [f for f in _XHR_Calls if "TweetResultByRestId" in f["url"]]

    tweets = []
    for xhr in tweetCall:
        if xhr.get("response") and "body" in xhr["response"]:
            try:
                data = json.loads(xhr["response"]["body"])
                tweet_data = data['data']['tweetResult']['result']
                tweet_parsed = parseTweet(tweet_data)
                if tweet_parsed:
                    tweets.append(tweet_parsed)
            except Exception as e:
                print(f"Error parsing tweet JSON: {e}")
    return tweets

async def main():
    url = input("x.com url post: ").strip()
    user, tweetId, date = extractInfoFromUrl(url)
    tweets = await scrapeTweet(url)

    dataset = []
    for tweet in tweets:
        content = tweet["text"]
        cleaned = preprocess(content)
        entities = [ent.text for ent in nlp(content).ents if ent.label_ in ("PERSON", "ORG", "GPE")]
        if cleaned:
            prediction = model.predict([cleaned])[0]
            sentiment = label_encoder.inverse_transform([prediction])[0]
            for ent in entities or ["(no_entity)"]:
                dataset.append({
                    "Tweet_ID": tweet["id"],
                    "Entity": ent,
                    "Sentiment": sentiment,
                    "Tweet_Content": content,
                    "Cleaned_Text": cleaned
                })

    if not dataset:
        print("No valid posts were found.")
        return

    df = pd.DataFrame(dataset)
    os.makedirs("scrapfly", exist_ok=True)
    filename = f"scrapfly/{user}{date}.csv"
    df.to_csv(filename, index=False)
    print(f"File saved: {filename}")

if __name__ == "__main__":
    asyncio.run(main())
