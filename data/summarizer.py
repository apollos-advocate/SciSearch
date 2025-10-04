import pandas as pd
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize(text):
    if pd.isna(text) or len(text.strip()) == 0:
        return ""
    return summarizer(text[:1024], max_length=60, min_length=20, do_sample=False)[0]["summary_text"]

def main():
    df = pd.read_csv("raw_studies.csv").drop_duplicates(subset=["Title", "Abstract"])
    print(df.head())
    df["Summary"] = df["Abstract"].apply(summarize)
    df.to_csv("structured_output.csv", index=False)
    print("✅ Saved summarized studies to structured_output.csv")

if __name__ == "__main__":
    main()
