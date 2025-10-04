import pandas as pd
from transformers import pipeline

print("Script is running...")

csv_files = ['micromuscle.csv', 'ntrs-export.csv', 'spaceradiation.csv']  

# Load and combine CSVs
dataframes = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dataframes, ignore_index=True)

print("Loaded CSVs. Shape:", df.shape)
print("Columns:", df.columns)

# Set up the summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Function to summarize abstracts
def summarize(text):
    if pd.isna(text) or len(str(text).strip()) == 0:
        return ""
    text = str(text)[:1024]  # truncate if too long
    summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return summary[0]['summary_text']

# Apply summarization to first 3 rows (for testing)
try:
    df['Summary'] = df['Abstract'].head(3).apply(summarize)
    print("✅ Summarization complete for first 3 rows.")
except Exception as e:
    print("❌ Error during summarization:", e)

# Save output
df.to_csv("summarized_output.csv", index=False)
print("✅ Done! Saved as summarized_output.csv")
