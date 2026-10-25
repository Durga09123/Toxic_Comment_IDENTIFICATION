# ============================================================
# TOXIC COMMENT DETECTION USING NLP
# Model Training Program
# ============================================================

import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")


data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")

print("Dataset loaded successfully!")
print("Number of comments:", len(data))


# ============================================================
# 2. CREATE TOXIC / NON-TOXIC LABEL
# ============================================================

toxic_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# If any toxic category is 1, comment is considered toxic
data["is_toxic"] = data[toxic_columns].max(axis=1)

# Remove comments with missing values
data = data.dropna(subset=["comment_text"])


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning comments...")

data["clean_comment"] = data["comment_text"].apply(clean_text)

print("Text preprocessing completed!")


# ============================================================
# 4. PREPARE INPUT AND OUTPUT
# ============================================================

X = data["clean_comment"]
y = data["is_toxic"]


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training comments:", len(X_train))
print("Testing comments:", len(X_test))


# ============================================================
# 6. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\nApplying TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF completed!")


# ============================================================
# 7. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ============================================================
# 8. EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Toxic", "Toxic"]
    )
)


# ============================================================
# 9. SAVE MODEL
# ============================================================

print("\nSaving model...")

joblib.dump(model, "toxic_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model saved as: toxic_model.pkl")
print("TF-IDF saved as: tfidf_vectorizer.pkl")

print("\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("========================================")