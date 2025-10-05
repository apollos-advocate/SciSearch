import streamlit as st
import pandas as pd
import requests
from xml.etree import ElementTree as ET
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- NLTK setup ---
nltk.download('punkt')

# --- Sumy summarizer ---
stemmer = Stemmer("english")
lsa_summarizer = LsaSummarizer(stemmer)

def sumy_summarize(text, sentences_count=3):
    if not text or text.strip() == "":
        return "No abstract available"
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary_sentences = lsa_summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary_sentences)

# --- PubMed fetch ---
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
    root = ET.fromstring(data)
    records = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="N/A")
        abstract = article.findtext(".//AbstractText", default="")
        authors = ", ".join([a.findtext("LastName", "") for a in article.findall(".//Author") if a.find("LastName") is not None])
        url = f"https://pubmed.ncbi.nlm.nih.gov/{article.findtext('.//PMID')}"
        records.append({"Title": title, "Abstract": abstract, "Authors": authors, "Source": "PubMed", "URL": url})
    return records

# --- NASA ADS fetch ---
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

# --- Load FLAN-T5 once ---
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

# --- Streamlit UI ---
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
            doc["Summary"] = sumy_summarize(doc.get("Abstract", ""))
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
