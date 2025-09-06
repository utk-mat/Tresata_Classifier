# Tresata Classifier with Gemini Fallback

A data classification tool that uses regex patterns for initial classification and falls back to Google's Gemini AI for ambiguous cases.

## Features

- **Regex-based classification** for phone numbers, companies, countries, and dates
- **Gemini AI fallback** for values that don't match regex patterns
- **CSV processing** with configurable column names
- **Error handling** with graceful fallbacks

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or run the setup script:

```bash
python setup.py
```

### 2. Configure Gemini API Key

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and set it as an environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Or add it to your shell profile:

```bash
echo 'export GEMINI_API_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

## Usage

### Basic Usage

```bash
python classifier.py --input sample.csv --column data
```

### Parameters

- `--input`: Path to input CSV file (required)
- `--column`: Column name to classify (default: "data")

### Output

The classifier creates a new column `{column_name}_class` with the classification results and saves to `classification_output.csv`.

## Classification Categories

- **phone**: Valid phone numbers with proper country codes and lengths
  - US/Canada (+1): 10 digits after country code
  - UK (+44): 10-11 digits after country code  
  - India (+91): 10 digits after country code
  - Germany (+49): 10-12 digits after country code
  - Australia (+61): 9 digits after country code
  - Other countries: 7-15 digits total
- **company**: Company names with common suffixes (Inc, Ltd, LLC, etc.)
- **country**: Country names from the predefined list
- **date**: Dates in YYYY-MM-DD or MM/DD/YYYY format
- **other**: Everything else (including invalid phone numbers and Gemini fallback results)

## How It Works

1. **Regex Matching**: First attempts to classify using regex patterns
2. **Gemini Fallback**: If no regex matches, sends the value to Gemini AI for classification
3. **Error Handling**: If Gemini fails, defaults to "other"

## Example

Input CSV:
```csv
data
+1-555-123-4567
Apple Inc
Germany
2023-12-25
Some random text
```

Output CSV:
```csv
data,data_class
+1-555-123-4567,phone
Apple Inc,company
Germany,country
2023-12-25,date
Some random text,other
```

## Error Handling

- Invalid API keys: Falls back to "other"
- Network issues: Falls back to "other"
- Invalid input: Returns "other"
- Missing files: Shows appropriate error messages
