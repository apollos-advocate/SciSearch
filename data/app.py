import streamlit as st
import pandas as pd
import requests
from xml.etree import ElementTree as ET

st.set_page_config(page_title="SciSearch", layout="wide")
st.title("SciSearch: PubMed Explorer")
st.markdown("Search life sciences studies related to space and get AI-generated summaries.")

# --- Summarizer ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  

# Initialize summarizer once
summarizer = LsaSummarizer()

def summarize(text, sentences_count=3):
    if not text or text.strip() == "":
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

# --- Fetch from PubMed ---
def fetch_pubmed(query, max_results=10):
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    resp = requests.get(esearch_url, params=params)
    pmids = resp.json().get('esearchresult', {}).get('idlist', [])

    if not pmids:
        return []

    fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    data = requests.get(efetch_url, params=fetch_params).text
    root = ET.fromstring(data)

    records = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="N/A")
        abstract = article.findtext(".//AbstractText", default="")
        authors = ", ".join([
            a.findtext("LastName", "") for a in article.findall(".//Author") if a.find("LastName") is not None
        ])
        records.append({
            "Title": title,
            "Abstract": abstract,
            "Authors": authors,
            "Source": "PubMed"
        })
    return records

# --- User Search Input ---
search_term = st.text_input("Enter your search term (e.g., 'microgravity muscle atrophy')")

if st.button("Search PubMed"):
    with st.spinner("Searching PubMed and summarizing results..."):
        results = fetch_pubmed(search_term, max_results=15)
        if results:
            df = pd.DataFrame(results)
            df["Summary"] = df["Abstract"].apply(summarize)

            st.success(f"Found {len(df)} studies.")
            
            for idx, row in df.iterrows():
                st.subheader(row["Title"])
                st.markdown(f"**Authors:** {row['Authors']}")
                st.markdown(f"**Summary:** {row['Summary']}")
                with st.expander("Show Abstract"):
                    st.markdown(row["Abstract"])
                st.markdown("---")

            # Option to download results
            st.download_button(
                label="Download as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="summarized_pubmed_studies.csv",
                mime="text/csv"
            )
        else:
            st.warning("No studies found for that query.")
