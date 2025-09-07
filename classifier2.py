#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import phonenumbers
from phonenumbers import geocoder
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import string
import json
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Calculate training performance
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y_encoded, y_pred)
    print(f"✅ Model and label encoder saved.")
    print(f"📊 Training Accuracy: {accuracy:.4f}")
    
    return clf, le, df["label"], le.inverse_transform(y_pred)

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
# ML Classification functions with prediction scores
# -----------------------------
def classify_value_ml(value: str, clf=None, le=None, return_proba=False):
    if clf is None or le is None:
        clf, le = load_model()
    features = pd.DataFrame([extract_features(value)])
    
    if return_proba:
        # Return probabilities for all classes
        probabilities = clf.predict_proba(features)[0]
        label_encoded = clf.predict(features)[0]
        label = le.inverse_transform([label_encoded])[0]
        return label, probabilities
    else:
        # Return just the predicted label (original behavior)
        label_encoded = clf.predict(features)[0]
        return le.inverse_transform([label_encoded])[0]

def classify_value_columns(value: str, clf=None, le=None, return_proba=False) -> dict:
    if return_proba:
        label, probabilities = classify_value_ml(value, clf, le, return_proba=True)
        class_names = le.classes_
        proba_dict = {f"{class_name}_score": float(prob) 
                     for class_name, prob in zip(class_names, probabilities)}
        
        result = {
            "phone": int(label == "phone"),
            "company": int(label == "company"),
            "country": int(label == "country"),
            "date": int(label == "date"),
            "other": int(label == "other"),
            "predicted_label": label
        }
        result.update(proba_dict)
        return result
    else:
        label = classify_value_ml(value, clf, le)
        return {
            "phone": int(label == "phone"),
            "company": int(label == "company"),
            "country": int(label == "country"),
            "date": int(label == "date"),
            "other": int(label == "other"),
            "predicted_label": label
        }

def get_prediction_scores(value: str):
    """Get detailed prediction scores for a single value"""
    clf, le = load_model()
    features = pd.DataFrame([extract_features(value)])
    
    # Get probabilities
    probabilities = clf.predict_proba(features)[0]
    class_names = le.classes_
    
    # Get prediction
    prediction = clf.predict(features)[0]
    predicted_label = le.inverse_transform([prediction])[0]
    
    # Create detailed results
    results = {
        "input": value,
        "predicted_label": predicted_label,
        "scores": {class_name: float(score) 
                  for class_name, score in zip(class_names, probabilities)}
    }
    
    return results

# -----------------------------
# Confidence Analysis Functions (No true labels needed)
# -----------------------------
def analyze_confidence_scores(df):
    """Analyze confidence scores without requiring true labels"""
    confidence_scores = []
    predicted_labels = []
    
    for _, row in df.iterrows():
        max_confidence = max([
            row.get('phone_score', 0),
            row.get('company_score', 0),
            row.get('country_score', 0),
            row.get('date_score', 0),
            row.get('other_score', 0)
        ])
        confidence_scores.append(max_confidence)
        predicted_labels.append(row['predicted_label'])
    
    confidence_scores = np.array(confidence_scores)
    
    # Calculate confidence statistics
    results = {
        "total_predictions": len(df),
        "avg_confidence": float(np.mean(confidence_scores)),
        "median_confidence": float(np.median(confidence_scores)),
        "min_confidence": float(np.min(confidence_scores)),
        "max_confidence": float(np.max(confidence_scores)),
        "confidence_std": float(np.std(confidence_scores)),
        "high_confidence_predictions": int(np.sum(confidence_scores >= 0.8)),
        "medium_confidence_predictions": int(np.sum((confidence_scores >= 0.6) & (confidence_scores < 0.8))),
        "low_confidence_predictions": int(np.sum(confidence_scores < 0.6)),
        "class_distribution": pd.Series(predicted_labels).value_counts().to_dict()
    }
    
    # Add confidence thresholds
    results["high_confidence_percentage"] = round(results["high_confidence_predictions"] / results["total_predictions"] * 100, 2)
    results["medium_confidence_percentage"] = round(results["medium_confidence_predictions"] / results["total_predictions"] * 100, 2)
    results["low_confidence_percentage"] = round(results["low_confidence_predictions"] / results["total_predictions"] * 100, 2)
    
    return results

def plot_confidence_distribution(confidence_scores, output_path="confidence_distribution.png"):
    """Plot confidence score distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.8, color='red', linestyle='--', label='High Confidence Threshold (0.8)')
    plt.axvline(x=0.6, color='orange', linestyle='--', label='Medium Confidence Threshold (0.6)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidence Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Confidence distribution plot saved as {output_path}")

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
# Process file with confidence analysis
# -----------------------------
def process_file(input_path, output_path="parsed_output.csv", include_scores=False, analyze=False):
    clf, le = load_model()
    
    # Read the input file
    try:
        df = pd.read_csv(input_path)
        input_data = df.iloc[:, 0].ast(str)  # Use first column as data
    except:
        # If reading with headers fails, try without headers
        df = pd.read_csv(input_path, header=None, names=["data"])
        input_data = df["data"].astype(str)
    
    print(f"📊 Processing {len(input_data)} records...")
    
    # Get classifications with scores (required for analysis)
    print("🔍 Classifying with prediction scores...")
    classifications = input_data.apply(
        lambda x: classify_value_columns(x, clf, le, return_proba=True)
    )
    
    result_df = pd.concat([input_data.rename("data"), classifications.apply(pd.Series)], axis=1)
    
    # Parse phones
    print("📞 Parsing phone numbers...")
    parsed_countries, parsed_numbers = [], []
    for idx, row in result_df.iterrows():
        if row["phone"] == 1:
            country, national = parse_phone(row["data"])
        else:
            country, national = " ", " "
        parsed_countries.append(country)
        parsed_numbers.append(national)
    result_df["ParsedCountry"] = parsed_countries
    result_df["ParsedNumber"] = parsed_numbers

    # Parse companies
    print("🏢 Parsing company names...")
    parsed_bases, parsed_legals = [], []
    for idx, row in result_df.iterrows():
        if row["company"] == 1:
            base, legal = parse_company(row["data"])
        else:
            base, legal = " ", " "
        parsed_bases.append(base)
        parsed_legals.append(legal)
    result_df["ParsedCompanyName"] = parsed_bases
    result_df["ParsedLegalSuffix"] = parsed_legals

    result_df.to_csv(output_path, index=False)
    print(f"✅ Parsed file saved at {output_path}")
    
    # Analyze confidence scores if requested
    if analyze:
        print("📈 Analyzing prediction confidence...")
        analysis_results = analyze_confidence_scores(result_df)
        
        # Print analysis results
        print("\n" + "="*60)
        print("PREDICTION CONFIDENCE ANALYSIS")
        print("="*60)
        print(f"Total predictions: {analysis_results['total_predictions']}")
        print(f"Average confidence: {analysis_results['avg_confidence']:.4f}")
        print(f"Median confidence: {analysis_results['median_confidence']:.4f}")
        print(f"Confidence range: [{analysis_results['min_confidence']:.4f}, {analysis_results['max_confidence']:.4f}]")
        
        print(f"\nConfidence Levels:")
        print(f"  High (≥0.8): {analysis_results['high_confidence_predictions']} ({analysis_results['high_confidence_percentage']}%)")
        print(f"  Medium (0.6-0.8): {analysis_results['medium_confidence_predictions']} ({analysis_results['medium_confidence_percentage']}%)")
        print(f"  Low (<0.6): {analysis_results['low_confidence_predictions']} ({analysis_results['low_confidence_percentage']}%)")
        
        print(f"\nClass Distribution:")
        for class_name, count in analysis_results['class_distribution'].items():
            percentage = (count / analysis_results['total_predictions']) * 100
            print(f"  {class_name.upper():<10}: {count} ({percentage:.1f}%)")
        
        # Save analysis results
        analysis_path = output_path.replace('.csv', '_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Plot confidence distribution
        confidence_scores = []
        for _, row in result_df.iterrows():
            confidence_scores.append(max([
                row.get('phone_score', 0),
                row.get('company_score', 0),
                row.get('country_score', 0),
                row.get('date_score', 0),
                row.get('other_score', 0)
            ]))
        
        plot_confidence_distribution(confidence_scores, output_path.replace('.csv', '_confidence_plot.png'))
        
        print(f"📊 Analysis results saved as {analysis_path}")
        
        return analysis_results
    
    return None

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ML Classifier + Phone/Company Parser")
    parser.add_argument("--train", type=str, help="CSV file with columns: data,label")
    parser.add_argument("--input", type=str, help="CSV to classify and parse")
    parser.add_argument("--scores", action="store_true", help="Include prediction scores in output")
    parser.add_argument("--test", type=str, help="Test a single value and show prediction scores")
    parser.add_argument("--analyze", action="store_true", help="Analyze prediction confidence (no true labels needed)")
    args = parser.parse_args()

    if args.train:
        train_model(args.train)
    elif args.input:
        process_file(args.input, include_scores=True, analyze=args.analyze)
    elif args.test:
        result = get_prediction_scores(args.test)
        print(json.dumps(result, indent=2))
    else:
        print("❌ Provide --train <csv> to train, --input <csv> to classify, or --test <value> to test a single value")