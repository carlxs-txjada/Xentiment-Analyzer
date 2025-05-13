import pandas as pd
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|@\w+|#\w+|[^A-Za-z\s]", '', text.lower())
    tokens = word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word not in stopWords])

path = "data/twitter_training.csv"
df = pd.read_csv(path, names=["Tweet_ID", "Entity", "Sentiment", "Tweet_Content"])

df = df.dropna(subset=['Tweet_Content'])

df['Cleaned_Text'] = df['Tweet_Content'].apply(preprocess)

label_encoder = LabelEncoder()
df['Encoded_Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

X_train, X_val, y_train, y_val = train_test_split(
    df['Cleaned_Text'],
    df['Encoded_Sentiment'],
    test_size=0.2,
    random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

params = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__min_df': [3, 5],
    'clf__C': [0.1, 1, 10]
}

grid = GridSearchCV(pipeline, param_grid=params, cv=3, scoring='f1_macro', verbose=1)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_val)
print("Validation Report:\n", classification_report(
    y_val, y_pred, target_names=label_encoder.classes_))

with open("XentimentAnalyzer.pkl", "wb") as f:
    pickle.dump((grid.best_estimator_, label_encoder), f)

print("Model trained, saved, and tested. 'XentimentAnalyzer.pkl' was saved.")
