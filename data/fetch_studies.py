import requests
import pandas as pd
from xml.etree import ElementTree as ET
import time

def fetch_pubmed(query, max_results=200):
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

    if not pmids:
        return []

    records = []
    batch_size = 100  # PubMed recommends ≤200
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml"
        }
        response = requests.get(efetch_url, params=fetch_params)
        root = ET.fromstring(response.text)

        for article in root.findall(".//PubmedArticle"):
            title = article.findtext(".//ArticleTitle", default="N/A")
            abstract = article.findtext(".//AbstractText", default="")
            authors = ", ".join([
                a.findtext("LastName", "") for a in article.findall(".//Author")
                if a.find("LastName") is not None
            ])
            records.append({
                "Title": title,
                "Abstract": abstract,
                "Authors": authors,
                "Source": "PubMed"
            })

        time.sleep(0.3)  # Respect API rate limits

    return records

def main():
    query = input("Enter search term (e.g., life science microgravity): ")
    results = fetch_pubmed(query, max_results=300)  # bump this as needed
    df = pd.DataFrame(results)
    df.to_csv("raw_studies.csv", index=False)
    print(f"✅ Saved {len(df)} studies to raw_studies.csv")

if __name__ == "__main__":
    main()
