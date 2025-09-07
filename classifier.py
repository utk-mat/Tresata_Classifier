#!/usr/bin/env python3
import re
import pandas as pd
import google.generativeai as genai

# -----------------------------
# Configure Gemini (hardcoded for testing)
# -----------------------------
GENIE_API_KEY = ""
genai.configure(api_key=GENIE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# Regex patterns
# -----------------------------
PHONE_REGEX = re.compile(
    r"^(\+?\d{1,3}[\s\-\.]?)?(\(?\d{2,5}\)?[\s\-\.]?\d+)+((\s+ext|\s+x|\s+extension)[\s\.]?\d+)?$",
    re.IGNORECASE,
)
DATE_REGEX = re.compile(
    r"^(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})$"
)
COMPANY_SUFFIXES = ["inc", "ltd", "llc", "plc", "corp", "company", "gmbh", "sa", "pvt"]
COUNTRIES = [
    "brunei darussalam", "argentina", "sao tome and principe", "zambia",
    "martinique", "united states virgin islands", "germany", "india",
    "bangladesh", "puerto rico"
]

# -----------------------------
# Check functions
# -----------------------------
def is_phone(value: str) -> bool:
    v = value.strip()
    if not PHONE_REGEX.match(v):
        return False
    digits_only = re.sub(r"[^\d]", "", v)
    if v.startswith("+1") or (v.startswith("1") and len(digits_only) == 11):
        return len(digits_only) == 11
    elif v.startswith("+44"):
        return 12 <= len(digits_only) <= 13
    elif v.startswith("+91"):
        return len(digits_only) == 12
    elif v.startswith("+49"):
        return 12 <= len(digits_only) <= 14
    elif v.startswith("+61"):
        return len(digits_only) == 11
    elif v.startswith("+"):
        country_code = v[1:4] if len(v) > 3 else v[1:]
        if country_code.isdigit() and int(country_code) > 999:
            return False
        return 7 <= len(digits_only) <= 15
    else:
        return 7 <= len(digits_only) <= 10

def is_company(value: str) -> bool:
    v = value.lower().strip().strip(",")
    return any(v.endswith(suffix) for suffix in COMPANY_SUFFIXES)

def is_country(value: str) -> bool:
    v = value.lower().strip().strip(",")
    return v in COUNTRIES

def is_date(value: str) -> bool:
    return bool(DATE_REGEX.match(value.strip()))

# -----------------------------
# Gemini fallback
# -----------------------------
def gemini_fallback(value: str) -> str:
    prompt = f"""
You are a data classifier. Classify the following value into one of:
phone, company, country, date, other

Value: "{value}"

Respond with only one word (phone/company/country/date/other).
"""
    try:
        response = gemini_model.generate_content(prompt)
        label = response.text.strip().lower()
        if label in {"phone", "company", "country", "date"}:
            return label
        return "other"
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "other"

# -----------------------------
# Main classifier returning all columns
# -----------------------------
def classify_value_columns(value: str) -> dict:
    """Returns a dictionary with column-wise classification (1 or 0)."""
    if not value or not isinstance(value, str):
        value = ""
    
    if is_phone(value):
        label = "phone"
    elif is_company(value):
        label = "company"
    elif is_country(value):
        label = "country"
    elif is_date(value):
        label = "date"
    else:
        label = gemini_fallback(value)
    
    return {
        "phone": 1 if label == "phone" else 0,
        "company": 1 if label == "company" else 0,
        "country": 1 if label == "country" else 0,
        "date": 1 if label == "date" else 0,
        "other": 1 if label == "other" else 0
    }

# -----------------------------
# CLI entry point
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify values in a CSV file (column-wise).")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--column", type=str, default="data", help="Column name (default: data)")
    args = parser.parse_args()

    df = pd.read_csv(args.input, header=None, names=[args.column])
    classifications = df[args.column].astype(str).apply(classify_value_columns)
    # Expand dictionary into separate columns
    df = pd.concat([df, classifications.apply(pd.Series)], axis=1)

    output_file = "classification_output_columns.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Column-wise classification complete! Results saved to {output_file}")
