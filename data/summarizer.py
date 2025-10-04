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

# ---------- NEW: Structured Tag Extraction Section ----------

# Define simple keyword lists for tagging
organism_keywords = ["mouse", "mice", "rat", "human", "plant", "yeast", "cell"]
condition_keywords = ["microgravity", "simulated microgravity", "spaceflight", "zero gravity"]
topic_keywords = ["muscle atrophy", "skeletal muscle", "renal function", "immune response", 
                  "bone loss", "fatigue", "neuromuscular", "fluid balance"]

def extract_tags(text):
    text = str(text).lower()
    
    organism = next((word for word in organism_keywords if word in text), "Unknown")
    condition = next((word for word in condition_keywords if word in text), "Unknown")
    topic = next((word for word in topic_keywords if word in text), "Unknown")
    return pd.Series([organism, condition, topic])

# Apply tag extraction to the rows with summaries
df[['Organism', 'Space Condition', 'Topic']] = df['Summary'].apply(extract_tags)

# Save output
df.to_csv("structured_output.csv", index=False)
print("✅ Done! Saved as structured_output.csv with summary + tags.")
