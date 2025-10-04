import requests
from transformers import pipeline

def pubmed_search(query, max_results=10):
    # Step 1: Use ESearch to get PMIDs
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    response = requests.get(esearch_url, params=params)
    pmids = response.json()['esearchresult']['idlist']
    return pmids

def pubmed_fetch_abstracts(pmids):
    # Step 2: Use EFetch to get abstracts for those PMIDs
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }
    response = requests.get(efetch_url, params=params)
    return response.text  # XML with abstracts

def extract_abstracts(xml_text):
    # Simple XML parsing to extract abstracts
    # You can use 'xml.etree.ElementTree' or 'BeautifulSoup'
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    abstracts = []
    for article in root.findall(".//AbstractText"):
        abstracts.append(article.text)
    return abstracts

# Summarizer setup
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    if not text:
        return ""
    # Truncate to 1024 tokens (if needed)
    text = text[:1024]
    summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    query = "life science microgravity spaceflight"
    pmids = pubmed_search(query, max_results=5)
    xml_data = pubmed_fetch_abstracts(pmids)
    abstracts = extract_abstracts(xml_data)
    
    for i, abstract in enumerate(abstracts):
        print(f"\n--- Study {i+1} Abstract ---\n{abstract}\n")
        print("Summary:")
        print(summarize_text(abstract))
        print("\n" + "="*40)
import pandas as pd

# At the end of pubmed.py
data = []

for i, abstract in enumerate(abstracts):
    summary = summarize_text(abstract)
    data.append({"PMID": pmids[i], "Abstract": abstract, "Summary": summary})

df = pd.DataFrame(data)
df.to_csv("pubmed_summaries.csv", index=False)
print("✅ Saved summarized data to pubmed_summaries.csv")
