"""
train_model.py
----------------
Trains a text classification model that reads a customer's message and
predicts the *intent* behind it (e.g. warranty_status, damage_report,
repair_request, coverage_info, greeting, out_of_scope).

Pipeline:
  raw text -> TF-IDF vectorizer -> Logistic Regression classifier

This is the "machine learning" piece of the project: instead of hand-written
if/keyword rules, the model learns from labeled examples (data/intents.csv)
which words are associated with which intent, and generalizes to new,
unseen phrasings.

Run:
    python src/train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_classifier.pkl")


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = df["text"], df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)

    print("=== Evaluation on held-out test split ===")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
