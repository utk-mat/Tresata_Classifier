import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ------------------------------
# Load data
# ------------------------------
def load_training_data():
    data = []

    # Phone numbers
    phone_df = pd.read_csv("phoneNumber.csv")
    for val in phone_df.iloc[:, 0]:
        data.append((str(val), "PhoneNumber"))

    # Companies
    comp_df = pd.read_csv("Company.csv")
    for val in comp_df.iloc[:, 0]:
        data.append((str(val), "CompanyName"))

    # Countries
    with open("countries.txt", "r", encoding="utf-8") as f:
        for line in f:
            data.append((line.strip(), "Country"))

    # Dates
    date_df = pd.read_csv("Dates.csv")
    for val in date_df.iloc[:, 0]:
        data.append((str(val), "Date"))

    # Add some random text for "Other"
    other_examples = ["abc123", "random text", "foo bar", "sample data"]
    for val in other_examples:
        data.append((val, "Other"))

    df = pd.DataFrame(data, columns=["text", "label"])
    return df

# ------------------------------
# Train model
# ------------------------------
def train_model():
    df = load_training_data()
    X = df["text"].tolist()
    y = df["label"].tolist()

    # Use SBERT for embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X_embeddings = model.encode(X, convert_to_numpy=True, normalize_embeddings=True)

    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model + SBERT
    joblib.dump(clf, "semantic_classifier.pkl")
    model.save("sbert_model")
    print("✅ Model saved as semantic_classifier.pkl and sbert_model")

if __name__ == "__main__":
    train_model()