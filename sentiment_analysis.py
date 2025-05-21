import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv('data/Reviews.csv')
data = data[['Text', 'Score']].dropna()

def get_sentiment(score):
    if score <= 2:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else:
        return 'positive'

data['Sentiment'] = data['Score'].apply(get_sentiment)
data = data.rename(columns={'Text': 'Review'})[['Review', 'Sentiment']]

samples = data.groupby('Sentiment').apply(lambda x: x.sample(n=10000, random_state=42)).reset_index(drop=True)

reviews = samples['Review']
labels = samples['Sentiment']

reviews_train, reviews_test, labels_train, labels_test = train_test_split(
    reviews, labels, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
reviews_train_vec = tfidf.fit_transform(reviews_train)
reviews_test_vec = tfidf.transform(reviews_test)

model = LogisticRegression(max_iter=1000)
model.fit(reviews_train_vec, labels_train)

predictions = model.predict(reviews_test_vec)

print("Accuracy:", accuracy_score(labels_test, predictions))
print("\nClassification Report:\n", classification_report(labels_test, predictions))

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
