import pandas as pd
from transformers import pipeline

print("🔍 Loading model...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
print("✅ Model ready.")

def summarize(text):
    if pd.isna(text) or len(text.strip()) == 0:
        return ""
    text = str(text)[:1024]  # Limit to max model input
    try:
        summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        print("⚠️ Summarization error:", e)
        return "Summary unavailable."

def main():
    try:
        df = pd.read_csv("raw_studies.csv").drop_duplicates(subset=["Title", "Abstract"])
        print(f"📄 Loaded {len(df)} studies.")

        print("📝 Summarizing...")
        df["Summary"] = df["Abstract"].apply(summarize)

        df.to_csv("structured_output.csv", index=False)
        print("✅ Saved summarized studies to structured_output.csv")
    except Exception as e:
        print("❌ Error in summarizer.py:", e)

if __name__ == "__main__":
    main()
