import streamlit as st
import pandas as pd
import requests
from xml.etree import ElementTree as ET
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from huggingface_hub import InferenceApi

# -------------------
# NLTK Setup
# -------------------
def ensure_nltk_resource(resource_name):
    try:
        nltk.data.find(f'tokenizers/{resource_name}')
    except LookupError:
        nltk.download(resource_name)

ensure_nltk_resource('punkt')

# -------------------
# Summarizer
# -------------------
summarizer = LsaSummarizer()

def summarize(text, sentences_count=3):
    if not text or text.strip() == "":
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

# -------------------
# Fetch PubMed
# -------------------
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
        authors = ", ".join([a.findtext("LastName", "") for a in article.findall(".//Author") if a.find("LastName") is not None])
        url = f"https://pubmed.ncbi.nlm.nih.gov/{article.findtext('.//PMID')}"
        records.append({
            "Title": title,
            "Abstract": abstract,
            "Authors": authors,
            "Source": "PubMed",
            "URL": url
        })
    return records

# -------------------
# Fetch NASA ADS
# -------------------
def fetch_nasa_ads(query, max_results=20, token=None):
    if token is None:
        raise ValueError("NASA ADS API token is required.")
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": query,
        "fl": "title,author,abstract,pubdate,bibcode,doctype",
        "rows": max_results,
        "format": "json"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        st.error(f"NASA ADS API error: {response.status_code} - {response.text}")
        return []
    data = response.json()
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

# -------------------
# Hugging Face Inference API Setup
# -------------------
hf_token = st.secrets.get("HF_API_TOKEN")
hf_model_id = "google/flan-t5-large"
hf_inference = InferenceApi(repo_id=hf_model_id, token=hf_token, task="text2text-generation")

def generate_ai_answer(question, docs, max_docs=5):
    context = "\n\n".join([doc["Abstract"] or "" for doc in docs[:max_docs]])
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    
    # Use raw_response to handle plain text output
    response = hf_inference(inputs=prompt, raw_response=True)
    
    if hasattr(response, "text"):
        return response.text.strip()
    return "No answer generated."

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="SciSearch", layout="wide")
st.title("SciSearch: PubMed & NASA ADS Explorer")

# -------------------
# Sidebar: Query & Filters
# -------------------
query = st.sidebar.text_input("Enter search term:", value="microgravity muscle atrophy")
source_filter = st.sidebar.multiselect("Filter by source", options=["PubMed", "NASA ADS"], default=["PubMed", "NASA ADS"])
max_results = st.sidebar.slider("Max results per source", 5, 30, 15)
fetch_button = st.sidebar.button("Fetch Studies")

# -------------------
# Session state caching
# -------------------
if "all_results" not in st.session_state:
    st.session_state["all_results"] = []

if fetch_button:
    all_results = []
    if "PubMed" in source_filter:
        with st.spinner("Fetching PubMed studies..."):
            all_results.extend(fetch_pubmed(query, max_results))
    if "NASA ADS" in source_filter:
        with st.spinner("Fetching NASA ADS studies..."):
            all_results.extend(fetch_nasa_ads(query, max_results, st.secrets["NASA_ADS_API_TOKEN"]))
    # add summaries
    for doc in all_results:
        doc["Summary"] = summarize(doc["Abstract"])
    st.session_state["all_results"] = all_results

# -------------------
# Tabs: Explorer & AI
# -------------------
tab1, tab2 = st.tabs(["Studies Explorer", "AI Q&A"])

with tab1:
    st.header("Search & Explore Studies")
    if not st.session_state["all_results"]:
        st.info("Use the sidebar to fetch studies first.")
    else:
        df = pd.DataFrame(st.session_state["all_results"])
        for idx, row in df.iterrows():
            st.subheader(row["Title"])
            st.markdown(f"**Authors:** {row['Authors']}")
            st.markdown(f"**Source:** {row['Source']}")
            if "PubDate" in row and row["PubDate"]:
                st.markdown(f"**Publication Date:** {row['PubDate']}")
            st.markdown(f"**Summary:** {row['Summary']}")
            st.markdown(f"[Open Study]({row['URL']})", unsafe_allow_html=True)
            with st.expander("Show Abstract"):
                st.write(row["Abstract"])
            st.markdown("---")
        st.download_button(
            label="Download results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="studies_results.csv",
            mime="text/csv"
        )

with tab2:
    st.header("AI Q&A")
    question = st.text_area("Ask a question related to your search:")
    if st.button("Generate AI Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not st.session_state["all_results"]:
            st.warning("No studies fetched yet. Fetch studies first using the sidebar.")
        else:
            with st.spinner("Generating AI answer..."):
                answer = generate_ai_answer(question, st.session_state["all_results"])
                st.markdown("### AI-generated Answer")
                st.write(answer)
                st.markdown("---")
                st.markdown("### Top Relevant Studies Used:")
                for doc in st.session_state["all_results"][:5]:
                    st.markdown(f"- [{doc['Title']}]({doc['URL']}) ({doc['Source']})")
