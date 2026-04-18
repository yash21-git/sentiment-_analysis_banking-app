# Customer Sentiment Analysis in Digital Banking (India)

## Overview

This project analyzes customer experience in digital banking applications using large-scale user review data from the Google Play Store. It combines machine learning and econometric techniques to extract sentiment, identify key drivers of user satisfaction, and uncover recurring issues in digital banking services.

The pipeline is designed to be fully reproducible and follows a structured workflow from data extraction to topic modelling.

---

## Data

- Source: Google Play Store (via `google-play-scraper`)
- Coverage: Banking applications across India
- Raw Data Size: ~14 million reviews
- Final Dataset: ~4 million filtered reviews (post April 2023)

---

## Methodology

### 1. Data Extraction
- Scraped app-level metadata and user reviews using Google Play Scraper API
- Batch-wise extraction across multiple banking applications

### 2. Data Cleaning
- Filtered reviews based on time (FY 2023–24 onwards)
- Removed missing and inconsistent observations
- Structured dataset for downstream analysis

### 3. Sentiment Analysis
- Implemented a fine-tuned **DistilBERT** model for sentiment classification
- Classified reviews into positive, neutral, and negative categories
- Generated confidence scores for predictions

### 4. Aggregation
- Computed app-level sentiment indicators
- Constructed key variable: **share of positive reviews**
- Generated comparative metrics across applications

### 5. Topic Modelling
- Extracted negative reviews (length > 30 characters)
- Applied **BERTopic** for topic extraction
- Identified recurring issues such as transaction failures, app crashes, and UI concerns

---

## Key Outputs

- App-level sentiment metrics (positive/negative shares)
- Distribution of customer sentiment across banking applications
- Topic clusters highlighting major user concerns
- Clean datasets ready for econometric analysis (e.g., fractional probit models)

---

## Tools & Technologies

- Python (Pandas, NumPy)
- NLP: Transformers (DistilBERT), BERTopic
- Data Collection: google-play-scraper
- Visualization: Matplotlib, Seaborn

---

## Reproducibility

The project is structured as a single pipeline (`main.py`) with modular steps:
1. Review extraction  
2. Data cleaning  
3. Sentiment analysis  
4. Aggregation  
5. Topic modelling  

All steps can be executed sequentially.

---

## Notes

- This study was independently implemented, including data collection, model execution, and analysis.
- The output dataset can be directly used for econometric modelling of customer experience in digital banking.

---
