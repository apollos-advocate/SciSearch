import requests
import pandas as pd

# NASA ADS API wrapper function
def fetch_nasa_ads(query, max_results=20, token=None):
    """
    Fetch publications from NASA ADS API with a query string.
    Returns a list of dicts with metadata.
    """
    if token is None:
        raise ValueError("NASA ADS API token is required.")
    
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {token}"}
    
    # Define the fields we want and max results
    params = {
        "q": query,
        "fl": "title,author,abstract,pubdate,bibcode,doctype",
        "rows": max_results,
        "format": "json"
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        raise Exception(f"NASA ADS API error: {response.status_code} - {response.text}")
    
    data = response.json()
    
    records = []
    for doc in data.get("response", {}).get("docs", []):
        records.append({
            "Title": doc.get("title", [""])[0],
            "Authors": ", ".join(doc.get("author", [])),
            "Abstract": doc.get("abstract", ""),
            "PubDate": doc.get("pubdate", ""),
            "Bibcode": doc.get("bibcode", ""),
            "Source": "NASA ADS"
        })
    
    return records
