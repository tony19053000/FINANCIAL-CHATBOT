import joblib

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_sentiment(text):
    tfidf_input = vectorizer.transform([text])
    prediction = model.predict(tfidf_input)[0]
    
    # Reverse label map
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map[prediction]

# Test it
headline = "Stock markets surge as inflation cools down"
predicted_sentiment = predict_sentiment(headline)
print(f"Predicted Sentiment: {predicted_sentiment}")
