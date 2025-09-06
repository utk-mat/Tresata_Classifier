import pandas as pd
import phonenumbers
from phonenumbers import geocoder
from sentence_transformers import SentenceTransformer, util
import json

# ------------------------------
# Load legal suffixes
# ------------------------------
try:
    with open("legal_suffixes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    LEGAL_SUFFIXES = data["all_suffixes"]
except FileNotFoundError:
    print("Warning: 'legal_suffixes.json' not found. Company parsing will be limited.")
    LEGAL_SUFFIXES = []

# SBERT model for company suffix matching
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------
# Phone Parsing
# ------------------------------
def parse_phone(number):
    """Parse phone number into country code and national number"""
    try:
        number_str = str(number).strip()
        # Try parsing with no region first
        parsed = phonenumbers.parse(number_str, None)
        if not phonenumbers.is_valid_number(parsed):
            return " ", " ", 0
        country = geocoder.region_code_for_number(parsed)
        national = str(parsed.national_number)
        return country, national, 1
    except:
        return " ", " ", 0

# ------------------------------
# Company Parsing
# ------------------------------
def parse_company(name):
    """Split company name into base and legal suffix using embeddings"""
    if not isinstance(name, str) or not name.strip():
        return " ", " ", 0

    name_clean = name.strip()

    if not LEGAL_SUFFIXES:
        return name_clean, " ", 0

    suffix_embeddings = sbert.encode(LEGAL_SUFFIXES, convert_to_tensor=True, normalize_embeddings=True)
    name_embedding = sbert.encode(name_clean, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(name_embedding, suffix_embeddings)[0]

    best_idx = cosine_scores.argmax().item()
    best_suffix = LEGAL_SUFFIXES[best_idx]
    best_score = cosine_scores[best_idx].item()

    # Accept suffix if similarity > threshold and text ends with it (case-insensitive)
    if best_score > 0.6 and name_clean.lower().endswith(best_suffix.lower()):
        base = name_clean[: -len(best_suffix)].strip(",. ").strip()
        return base, best_suffix, 1
    else:
        return name_clean, " ", 0

# ------------------------------
# Process Classified CSV
# ------------------------------
def process_file(input_path, output_path="parsed_output.csv"):
    df = pd.read_csv(input_path)

    # ------------------------------
    # Parse Phone Numbers
    # Only process rows flagged as PhoneNumber==1
    # ------------------------------
    if "PhoneNumber" in df.columns:
        countries, numbers = [], []
        for idx, row in df.iterrows():
            if row["PhoneNumber"] == 1:
                country, number, _ = parse_phone(row["data"])
            else:
                country, number = " ", " "
            countries.append(country)
            numbers.append(number)
        df["ParsedCountry"] = countries
        df["ParsedNumber"] = numbers
    else:
        print("Warning: 'PhoneNumber' column not found.")

    # ------------------------------
    # Parse Company Names
    # Only process rows flagged as CompanyName==1
    # ------------------------------
    if "CompanyName" in df.columns:
        bases, legals = [], []
        for idx, row in df.iterrows():
            if row["CompanyName"] == 1:
                base, legal, _ = parse_company(row["data"])
            else:
                base, legal = " ", " "
            bases.append(base)
            legals.append(legal)
        df["ParsedName"] = bases
        df["ParsedLegal"] = legals
    else:
        print("Warning: 'CompanyName' column not found.")

    # ------------------------------
    # Save parsed CSV
    # ------------------------------
    df.to_csv(output_path, index=False)
    print(f"✅ Parsed file saved at {output_path}")

if __name__ == "__main__":
    process_file("classification_output_columns.csv")
