#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import phonenumbers
from phonenumbers import geocoder
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import string
import json

# -----------------------------
# ML Classifier persistence
# -----------------------------
MODEL_FILE = "ml_classifier.joblib"
LABEL_ENCODER_FILE = "label_encoder.joblib"

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(value: str) -> dict:
    value = str(value).strip()
    return {
        "length": len(value),
        "digits": sum(c.isdigit() for c in value),
        "letters": sum(c.isalpha() for c in value),
        "spaces": sum(c.isspace() for c in value),
        "specials": sum(not c.isalnum() and not c.isspace() for c in value),
        "upper": sum(c.isupper() for c in value),
        "lower": sum(c.islower() for c in value),
        "plus": int("+" in value),
        "dot": int("." in value),
        "dash": int("-" in value),
        "comma": int("," in value)
    }

# -----------------------------
# Train ML model
# -----------------------------
# -----------------------------
# Train ML model
# -----------------------------
def train_model(csv_file: str):
    try:
        df = pd.read_csv(csv_file, quotechar='"', dtype=str, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"⚠️ Warning: Encountered a ParserError while reading the CSV. Some malformed lines might have been skipped. Error: {e}")
        df = pd.read_csv(csv_file, quotechar='"', dtype=str, error_bad_lines=False)
    
    if "data" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'data' and 'label' columns")

    # Drop any rows where 'data' or 'label' is missing after the skip
    df.dropna(subset=['data', 'label'], inplace=True)

    if df.empty:
        raise ValueError("No valid data found after cleaning the CSV. Please check your file.")

    X = pd.DataFrame([extract_features(v) for v in df["data"]])
    le = LabelEncoder()
    y_encoded = le.fit_transform(df["label"])

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y_encoded)

    joblib.dump(clf, MODEL_FILE)
    joblib.dump(le, LABEL_ENCODER_FILE)
    print(f"✅ Model and label encoder saved.")

# -----------------------------
# Load model
# -----------------------------
def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_ENCODER_FILE):
        raise FileNotFoundError("Model or label encoder not found. Train first.")
    clf = joblib.load(MODEL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)
    return clf, le

# -----------------------------
# ML Classification functions
# -----------------------------
def classify_value_ml(value: str, clf=None, le=None) -> str:
    if clf is None or le is None:
        clf, le = load_model()
    features = pd.DataFrame([extract_features(value)])
    label_encoded = clf.predict(features)[0]
    return le.inverse_transform([label_encoded])[0]

def classify_value_columns(value: str, clf=None, le=None) -> dict:
    label = classify_value_ml(value, clf, le)
    return {
        "phone": int(label == "phone"),
        "company": int(label == "company"),
        "country": int(label == "country"),
        "date": int(label == "date"),
        "other": int(label == "other")
    }

# -----------------------------
# Phone Parsing
# -----------------------------
def parse_phone(number):
    try:
        number_str = str(number).strip()
        parsed = phonenumbers.parse(number_str, None)
        if not phonenumbers.is_valid_number(parsed):
            return " ", " "
        country = geocoder.region_code_for_number(parsed)
        national = str(parsed.national_number)
        return country, national
    except:
        return " ", " "

# -----------------------------
# Company Parsing
# -----------------------------
# Load legal suffixes
try:
    with open("legal_suffixes.json", "r", encoding="utf-8") as f:
        LEGAL_SUFFIXES = json.load(f)["all_suffixes"]
except:
    LEGAL_SUFFIXES = []

# SBERT model
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def parse_company(name):
    if not isinstance(name, str) or not name.strip():
        return " ", " "
    name_clean = name.strip()
    if not LEGAL_SUFFIXES:
        return name_clean, " "
    suffix_embeddings = sbert.encode(LEGAL_SUFFIXES, convert_to_tensor=True, normalize_embeddings=True)
    name_embedding = sbert.encode(name_clean, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(name_embedding, suffix_embeddings)[0]
    best_idx = cosine_scores.argmax().item()
    best_suffix = LEGAL_SUFFIXES[best_idx]
    best_score = cosine_scores[best_idx].item()
    if best_score > 0.6 and name_clean.lower().endswith(best_suffix.lower()):
        base = name_clean[: -len(best_suffix)].strip(",. ").strip()
        return base, best_suffix
    return name_clean, " "

# -----------------------------
# Process file
# -----------------------------
def process_file(input_path, output_path="parsed_output.csv"):
    clf, le = load_model()
    df = pd.read_csv(input_path, header=None, names=["data"])
    classifications = df["data"].astype(str).apply(lambda x: classify_value_columns(x, clf, le))
    df = pd.concat([df, classifications.apply(pd.Series)], axis=1)

    # Parse phones
    parsed_countries, parsed_numbers = [], []
    for idx, row in df.iterrows():
        if row["phone"] == 1:
            country, national = parse_phone(row["data"])
        else:
            country, national = " ", " "
        parsed_countries.append(country)
        parsed_numbers.append(national)
    df["ParsedCountry"] = parsed_countries
    df["ParsedNumber"] = parsed_numbers

    # Parse companies
    parsed_bases, parsed_legals = [], []
    for idx, row in df.iterrows():
        if row["company"] == 1:
            base, legal = parse_company(row["data"])
        else:
            base, legal = " ", " "
        parsed_bases.append(base)
        parsed_legals.append(legal)
    df["ParsedCompanyName"] = parsed_bases
    df["ParsedLegalSuffix"] = parsed_legals

    df.to_csv(output_path, index=False)
    print(f"✅ Parsed file saved at {output_path}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ML Classifier + Phone/Company Parser")
    parser.add_argument("--train", type=str, help="CSV file with columns: data,label")
    parser.add_argument("--input", type=str, help="CSV to classify and parse")
    args = parser.parse_args()

    if args.train:
        train_model(args.train)
    elif args.input:
        process_file(args.input)
    else:
        print("❌ Provide --train <csv> to train or --input <csv> to classify")
