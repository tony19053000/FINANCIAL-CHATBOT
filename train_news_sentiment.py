import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load and combine datasets
file1 = "data/financial_news_sentiment(AutoRecovered).csv"
file2 = "data/google_financial_news_sentiment.csv"

# Check if files exist
if not os.path.exists(file1) or not os.path.exists(file2):
    raise FileNotFoundError("One or both dataset files not found!")

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df = pd.concat([df1, df2], ignore_index=True)

# Clean and prepare data
df = df[['title', 'Sentiment']].dropna()
df['Sentiment'] = df['Sentiment'].str.lower()  # normalize case
df = df[df['Sentiment'].isin(["positive", "neutral", "negative"])]

# Map sentiment to numeric labels
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df['label'] = df['Sentiment'].map(label_map)

# Show dataset info
print("Dataset size:", df.shape)
print("Sentiment counts:\n", df['Sentiment'].value_counts())
print(df.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['title'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved to 'models/' folder.")
