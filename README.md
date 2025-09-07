# 🚀 VIT-Tresata Hackathon 2025 - Team 8

## 🆔 Team Information  
- **Team Name:** Hackathon Team - 8
- **Hackathon Theme:** Data Classification and Processing
- **Selected Topic / Problem Statement:** Intelligent Data Classification and Parsing Pipeline

## 👥 Team Members  

| S.No | Name            | Reg No.   | VIT Email ID                      | Personal Email ID                | Mobile No.  | 
|------|--------------   |---------- |----------------------             |--------------------              |-------------|
| 1    | Pratyush Dubey  | 22BBS0064 |pratyush.dubey2022@vitstudent.ac.in|pratyushdubey2021@gmail.com       |8777494857   |
| 2    | Anushika Verma  | 22BCE2301 |anushika.verma2022@vitstudent.ac.in|anu11421verma@gmail.com           |6351940851   |
| 3    | Utkarsh Mathur  | 22BCT0129 |utkarsh.mathur2022@vitstudent.ac.in|utkarshbmathur04@gmail.com        |6205404964   |

---

# Tresata Classifier - Two-Stage Data Processing Pipeline

A comprehensive data classification and parsing system that uses machine learning models to classify data into categories (phone, company, country, date, other) and then parses the classified data for structured extraction.

## 🏗️ Architecture Overview

This project implements a **two-stage pipeline**:

1. **Stage 1: Classification** - Uses machine learning models to categorize input data
2. **Stage 2: Parsing** - Extracts structured information from classified data

## 🤖 Classification Models

### Primary Model: Random Forest Classifier (`classifier2.py`)

The main classification engine uses a **Random Forest model** trained on feature-engineered data:

- **Features**: Length, digit count, letter count, special characters, case analysis, punctuation patterns
- **Categories**: phone, company, country, date, other
- **Confidence Scoring**: Provides prediction probabilities for each class
- **Performance Analysis**: Built-in confidence analysis and visualization
- **Model Persistence**: Saves trained models for reuse

#### Key Features:
- ✅ **High Accuracy**: Trained Random Forest with 200 estimators
- ✅ **Confidence Analysis**: Detailed prediction confidence metrics
- ✅ **Feature Engineering**: 10+ engineered features for robust classification
- ✅ **Model Persistence**: Saves/loads trained models automatically
- ✅ **Performance Metrics**: Training accuracy and confidence distribution analysis

### Secondary Model: Gemini API Fallback (`classifier.py`)

A lightweight fallback classifier using Google's Gemini AI:

- **Purpose**: Alternative classification method using AI
- **Use Case**: When ML model is not available or for comparison
- **Categories**: Same 5 categories as Random Forest model
- **Integration**: Can be used alongside or instead of the ML model

## 📊 Two-Stage Pipeline

### Stage 1: Classification
```bash
# Train the Random Forest model
python classifier2.py --train training_data.csv

# Classify new data with confidence analysis
python classifier2.py --input sample.csv --analyze
```

**Output**: CSV with classification results and confidence scores

### Stage 2: Parsing (`parseB.py`)
```bash
# Parse classified data for structured extraction
python parseB.py
```

**Input**: Classification output from Stage 1
**Output**: Parsed data with extracted components

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (First Time)

```bash
python classifier2.py --train training_data.csv
```

### 3. Run Complete Pipeline

```bash
# Stage 1: Classify your data
python classifier2.py --input your_data.csv --analyze

# Stage 2: Parse the classified results
python parseB.py
```

## 📋 Detailed Usage

### Random Forest Classifier (`classifier2.py`)

#### Training
```bash
python classifier2.py --train training_data.csv
```
- Requires CSV with `data` and `label` columns
- Saves model as `ml_classifier.joblib`
- Saves label encoder as `label_encoder.joblib`

#### Classification
```bash
# Basic classification
python classifier2.py --input sample.csv

# With confidence analysis and visualization
python classifier2.py --input sample.csv --analyze

# Test single value with detailed scores
python classifier2.py --test "+1-555-123-4567"
```

#### Output Columns
- `phone`, `company`, `country`, `date`, `other`: Binary classification (0/1)
- `predicted_label`: The predicted category
- `{category}_score`: Confidence scores for each category
- `ParsedCountry`, `ParsedNumber`: Extracted phone components
- `ParsedCompanyName`, `ParsedLegalSuffix`: Extracted company components

### Gemini Classifier (`classifier.py`)

```bash
# Set API key
export GEMINI_API_KEY="your_api_key_here"

# Run classification
python classifier.py --input sample.csv --column data
```

### Parser (`parseB.py`)

```bash
# Parse classification results
python parseB.py
```

**Parsing Capabilities:**
- **Phone Numbers**: Extracts country code and national number
- **Company Names**: Separates base name from legal suffix using semantic similarity

## 📈 Performance Analysis

The Random Forest classifier includes comprehensive performance analysis:

- **Confidence Distribution**: Histogram of prediction confidence scores
- **Class Distribution**: Breakdown of predicted categories
- **Confidence Levels**: High (≥0.8), Medium (0.6-0.8), Low (<0.6)
- **Statistical Metrics**: Mean, median, standard deviation of confidence

## 📁 Project Structure

```
Tresata_Classifier/
├── classifier2.py          # Main Random Forest classifier
├── classifier.py           # Gemini API classifier (fallback)
├── parseB.py              # Second-stage parser
├── TrainB.py              # Training utilities
├── ml_classifier.joblib   # Trained Random Forest model
├── label_encoder.joblib   # Label encoder for model
├── legal_suffixes.json    # Company legal suffixes database
├── data/
│   ├── company_suffixes.txt
│   └── countries.txt
├── requirements.txt       # Python dependencies
└── setup.py              # Setup script
```

## 🔧 Dependencies

### Core Dependencies
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning (Random Forest)
- `numpy` - Numerical operations
- `joblib` - Model persistence

### ML & NLP
- `sentence-transformers` - Semantic similarity for company parsing
- `phonenumbers` - Phone number validation and parsing

### Visualization
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization

### Optional
- `google-generativeai` - Gemini API integration

## 📊 Classification Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **phone** | Valid phone numbers | `+1-555-123-4567`, `+44 20 7946 0958` |
| **company** | Company names with legal suffixes | `Apple Inc`, `Microsoft Corporation` |
| **country** | Country names | `United States`, `Germany`, `India` |
| **date** | Date formats | `2023-12-25`, `12/25/2023` |
| **other** | Everything else | Invalid data, random text |

## 🎯 Use Cases

- **Data Cleaning**: Automatically categorize and structure messy data
- **Contact Information Extraction**: Parse phone numbers and company details
- **Data Validation**: Identify and flag invalid or malformed entries
- **ETL Pipelines**: Integrate into data processing workflows
- **Data Quality Assessment**: Analyze confidence scores for data quality metrics

## 🔍 Advanced Features

### Confidence Analysis
- Real-time confidence scoring for each prediction
- Confidence distribution visualization
- Quality metrics for data assessment

### Semantic Company Parsing
- Uses sentence transformers for intelligent suffix matching
- Handles variations in legal entity naming
- Extracts base company names and legal suffixes

### Phone Number Validation
- International phone number validation
- Country code extraction
- National number formatting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample data
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: The Random Forest classifier (`classifier2.py`) is the primary and recommended model for production use, offering superior performance, confidence analysis, and structured output compared to the Gemini API fallback.

## 📁 Project Organization

This project follows the hackathon folder structure:

- `Code/` → All source code files (classifier2.py, classifier.py, parseB.py, etc.)
- `Docs/` → Documentation, sample data, and output files
- `Demo/` → Demo materials and presentations
