from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df.drop("business_model", axis=1)
y = df["business_model"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print("\nDecision Tree Report:\n", classification_report(y_test, dt_predictions))

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print("\nLogistic Regression Report:\n", classification_report(y_test, lr_predictions))

if accuracy_score(y_test, dt_predictions) > accuracy_score(y_test, lr_predictions):
    best_model = dt_model
else:
    best_model = lr_model

print("Best Model Selected:", best_model)

import joblib
joblib.dump(best_model, "best_model.pkl")

# Task 3: Sentiment Analysis using NLTK (VADER)

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required lexicon
nltk.download('vader_lexicon')

# Sample text data
texts = [
    "I really love this product, it’s amazing!",
    "This is the worst experience I’ve ever had.",
    "The movie was okay, not too good and not too bad.",
    "I am extremely happy with the service!",
    "I feel nothing about this.",
]

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

print("----- Sentiment Analysis Results -----\n")

# Loop through each text and analyze sentiment
for text in texts:
    scores = sia.polarity_scores(text)

    # Print original text
    print(f"Text: {text}")
    
    # Print sentiment scores
    print("Scores:", scores)

    # Determine final sentiment label
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "Positive 😀"
    elif compound <= -0.05:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    print("Overall Sentiment:", sentiment)
    print("-------------------------------------\n")
