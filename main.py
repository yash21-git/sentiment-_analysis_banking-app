# ============================================================
# CUSTOMER SENTIMENT ANALYSIS IN DIGITAL BANKING
# Full Pipeline: Extraction → Cleaning → Sentiment → Topics
# Author: Yash
# ============================================================

# ============================
# 1. IMPORTS & SETUP
# ============================
import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from google_play_scraper import reviews_all
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ============================
# 2. CONFIGURATION
# ============================
APP_IDS = ["com.axis.mobile", "com.sbi.lotusintouch"]  # sample apps
RAW_DIR = "data/raw_reviews"
CLEAN_DIR = "data/clean_reviews"
MODEL_PATH = "models/distilbert"
START_DATE = "2023-04-01"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# ============================
# 3. REVIEW EXTRACTION
# ============================
def fetch_reviews():
    """
    Extracts reviews from Google Play Store.
    """
    print("📥 Fetching reviews...")

    for app_id in APP_IDS:
        try:
            df = pd.DataFrame(
                reviews_all(app_id, lang='en', country='in')
            )

            path = os.path.join(RAW_DIR, f"{app_id}.csv")
            df.to_csv(path, index=False)

            print(f"✅ {app_id}: {len(df)} reviews saved")

        except Exception as e:
            print(f"❌ Error fetching {app_id}: {e}")

# ============================
# 4. DATA CLEANING
# ============================
def clean_reviews():
    """
    Filters reviews by date and keeps required columns.
    """
    print("🧹 Cleaning data...")

    for file in os.listdir(RAW_DIR):
        df = pd.read_csv(os.path.join(RAW_DIR, file))

        if 'at' not in df.columns:
            continue

        df['at'] = pd.to_datetime(df['at'], errors='coerce')
        df = df[df['at'] >= START_DATE]

        df = df[['content', 'score', 'at']].dropna()

        df.to_csv(os.path.join(CLEAN_DIR, file), index=False)

# ============================
# 5. SENTIMENT ANALYSIS (DISTILBERT)
# ============================
def run_sentiment():
    """
    Runs DistilBERT sentiment classification.
    """
    print("🤖 Running sentiment analysis...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    for file in os.listdir(CLEAN_DIR):
        path = os.path.join(CLEAN_DIR, file)
        df = pd.read_csv(path)

        sentiments = []

        for text in tqdm(df['content'], desc=f"Processing {file}"):
            inputs = tokenizer(
                str(text),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs).item()

            sentiments.append(label)

        df['sentiment'] = sentiments
        df.to_csv(path, index=False)

# ============================
# 6. SENTIMENT AGGREGATION
# ============================
def aggregate_results():
    """
    Computes share of positive reviews.
    """
    print("📊 Aggregating results...")

    summary = []

    for file in os.listdir(CLEAN_DIR):
        df = pd.read_csv(os.path.join(CLEAN_DIR, file))

        total = len(df)
        positive = (df['sentiment'] == 2).sum()  # assuming label 2 = positive

        summary.append({
            "app": file,
            "total_reviews": total,
            "positive_share": positive / total if total > 0 else 0
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("app_summary.csv", index=False)

    return summary_df

# ============================
# 7. TOPIC MODELLING (BERTOPIC)
# ============================
def run_topic_model():
    """
    Extracts topics from negative reviews.
    """
    print("🧠 Running topic modelling...")

    all_reviews = []

    for file in os.listdir(CLEAN_DIR):
        df = pd.read_csv(os.path.join(CLEAN_DIR, file))

        neg_reviews = df[df['sentiment'] == 0]['content'].dropna().tolist()
        all_reviews.extend(neg_reviews)

    print(f"Total negative reviews: {len(all_reviews)}")

    # Sample for training
    sample = all_reviews[:30000]

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=50
    )

    topics, _ = topic_model.fit_transform(sample)

    print("Top Topics:")
    print(topic_model.get_topic_info().head())

# ============================
# 8. MAIN PIPELINE
# ============================
if __name__ == "__main__":

    print("🚀 STARTING PIPELINE\n")

    fetch_reviews()
    clean_reviews()
    run_sentiment()
    aggregate_results()
    run_topic_model()

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
