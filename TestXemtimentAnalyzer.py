import pandas as pd
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|@\w+|#\w+|[^A-Za-z\s]", '', text.lower())
    tokens = word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word not in stopWords])

testPath = "data/twitter_validation.csv"
test_df = pd.read_csv(testPath, names=["Tweet_ID", "Entity", "Sentiment", "Tweet_Content"])

test_df = test_df.dropna(subset=['Tweet_Content'])

test_df['Cleaned_Text'] = test_df['Tweet_Content'].apply(preprocess)

with open("XentimentAnalyzer.pkl", "rb") as f:
    model, label_encoder = pickle.load(f)

y_true = label_encoder.transform(test_df['Sentiment'])
X_test = test_df['Cleaned_Text']

y_pred = model.predict(X_test)

print("Test Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Current")
plt.title("Confusion Matrix on Test Data")
plt.tight_layout()
plt.show()
