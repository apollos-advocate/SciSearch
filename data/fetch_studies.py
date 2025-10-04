import requests
import pandas as pd
from xml.etree import ElementTree as ET

def fetch_pubmed(query, max_results=20):
    print(f"🔍 Searching PubMed for: '{query}' (max {max_results} results)")
    
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    # Step 1: Get PMIDs
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    resp = requests.get(esearch_url, params=params)
    pmids = resp.json().get('esearchresult', {}).get('idlist', [])
    print(f"Found {len(pmids)} PMIDs")

    if not pmids:
        print("No studies found.")
        return []

    # Step 2: Fetch Details
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }
    data = requests.get(efetch_url, params=fetch_params).text
    root = ET.fromstring(data)

    records = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="N/A")
        abstract = article.findtext(".//AbstractText", default="")
        authors = ", ".join([a.findtext("LastName", "") for a in article.findall(".//Author") if a.find("LastName") is not None])
        records.append({
            "Title": title,
            "Abstract": abstract,
            "Authors": authors,
            "Source": "PubMed"
        })
    print(f"Fetched {len(records)} records from PubMed")
    return records

def main():
    query = input("Enter search term (e.g., life science microgravity): ")
    results = fetch_pubmed(query, max_results=30)
    if results:
        df = pd.DataFrame(results)
        df.to_csv("raw_studies.csv", index=False)
        print(f"✅ Saved {len(df)} studies to raw_studies.csv")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
