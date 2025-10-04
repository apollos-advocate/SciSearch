import pandas as pd

nasa_df = pd.read_csv("structured_output.csv")
pubmed_df = pd.read_csv("pubmed_summaries.csv")

# Add source tag
nasa_df['Source'] = 'NASA'
pubmed_df['Source'] = 'PubMed'

# Make sure both have the same columns
for col in ['Organism', 'Space Condition', 'Topic', 'Outcome', 'Title', 'Record URL']:
    if col not in pubmed_df.columns:
        pubmed_df[col] = 'N/A'

# Merge
merged_df = pd.concat([nasa_df, pubmed_df], ignore_index=True)
merged_df.to_csv("merged_output.csv", index=False)
print("✅ Combined and saved as merged_output.csv")
