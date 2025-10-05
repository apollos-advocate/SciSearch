import streamlit as st
import pandas as pd
import requests
from xml.etree import ElementTree as ET
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os


@st.cache_resource
def load_flan_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_flan_model()

def summarize_text(text, max_length=150):
    if not text.strip():
        return ""
    inputs = tokenizer(f"summarize: {text}", return_tensors="pt", truncation=True)
    summary_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_ai_answer(question, docs, chunk_size=3):
    abstracts = [doc.get("Abstract", "") for doc in docs if doc.get("Abstract")]
    chunks = [abstracts[i:i+chunk_size] for i in range(0, len(abstracts), chunk_size)]
    chunk_summaries = []
    for chunk in chunks:
        combined = "\n\n".join(chunk)
        chunk_summaries.append(summarize_text(combined))
    context = "\n\n".join(chunk_summaries)
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_ids = model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


@st.cache_resource
def load_distilbart_summarizer():
    print("Loading distilbart model...")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    print("Distilbart ready.")
    return summarizer

distilbart_summarizer = load_distilbart_summarizer()

def summarize_abstract(text):
    if pd.isna(text) or len(text.strip()) == 0:
        return "No abstract available"
    text = str(text)[:1024]
    try:
        summary = distilbart_summarizer(text, max_length=60, min_length=20, do_sample=False)
        return summary[0]["summary_text"]
    except:
        return "Summary unavailable"


def fetch_pubmed(query, max_results=10):
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    resp = requests.get(esearch_url, params=params)
    pmids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []
    fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    data = requests.get(efetch_url, params=fetch_params).text
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []
    records = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="N/A")
        abstract = article.findtext(".//AbstractText", default="")
        authors = ", ".join([a.findtext("LastName", "") for a in article.findall(".//Author") if a.find("LastName") is not None])
        url = f"https://pubmed.ncbi.nlm.nih.gov/{article.findtext('.//PMID')}"
        records.append({"Title": title, "Abstract": abstract, "Authors": authors, "Source": "PubMed", "URL": url})
    return records


def fetch_nasa_ads(query, max_results=10, token=None):
    if not token:
        return []
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "fl": "title,author,abstract,pubdate,bibcode,doctype", "rows": max_results, "format": "json"}
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        st.error(f"NASA ADS API error: {resp.status_code} - {resp.text}")
        return []
    data = resp.json()
    records = []
    for doc in data.get("response", {}).get("docs", []):
        bibcode = doc.get("bibcode", "")
        url = f"https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract"
        records.append({
            "Title": doc.get("title", [""])[0],
            "Authors": ", ".join(doc.get("author", [])),
            "Abstract": doc.get("abstract", ""),
            "PubDate": doc.get("pubdate", ""),
            "Source": "NASA ADS",
            "URL": url
        })
    return records


st.set_page_config(page_title="SciSearch", layout="wide")
st.title("SciSearch: PubMed & NASA ADS Explorer")

tab1, tab2 = st.tabs(["Studies Explorer", "AI Overview"])

with tab1:
    st.header("Search & Explore Studies")
    query = st.text_input("Enter search term:", value="microgravity muscle atrophy")
    sources = st.multiselect("Filter by source", ["PubMed", "NASA ADS"], default=["PubMed"])
    max_results = st.slider("Max results per source", 5, 20, 10)

    if st.button("Search Studies"):
        all_results = []
        if "PubMed" in sources:
            all_results.extend(fetch_pubmed(query, max_results))
        if "NASA ADS" in sources:
            token = st.secrets.get("NASA_ADS_API_TOKEN")
            all_results.extend(fetch_nasa_ads(query, max_results, token))

        for doc in all_results:
            doc["Summary"] = summarize_abstract(doc.get("Abstract", ""))

        if all_results:
            df = pd.DataFrame(all_results)
            st.success(f"Found {len(all_results)} studies.")
            for _, row in df.iterrows():
                st.subheader(row["Title"])
                st.markdown(f"**Authors:** {row['Authors']}")
                st.markdown(f"**Source:** {row['Source']}")
                if row.get("PubDate"):
                    st.markdown(f"**Publication Date:** {row['PubDate']}")
                st.markdown(f"**Summary:** {row['Summary']}")
                st.markdown(f"[Open Study]({row['URL']})", unsafe_allow_html=True)
                with st.expander("Show Abstract"):
                    st.write(row["Abstract"])
                st.markdown("---")
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "studies_results.csv")
        else:
            st.warning("No studies found.")

with tab2:
    st.header("AI Overview")
    question = st.text_area("Ask a question related to your search:")
    if st.button("Generate AI Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            combined_docs = []
            if "PubMed" in sources:
                combined_docs.extend(fetch_pubmed(query, max_results))
            if "NASA ADS" in sources:
                token = st.secrets.get("NASA_ADS_API_TOKEN")
                combined_docs.extend(fetch_nasa_ads(query, max_results, token))

            if combined_docs:
                answer = generate_ai_answer(question, combined_docs)
                st.markdown("### AI-generated Answer")
                st.write(answer)
            else:
                st.warning("No studies found to generate AI answer.")
