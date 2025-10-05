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
# NLTK setup
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
    try:
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
    except Exception as e:
        st.error(f"PubMed fetch failed: {e}")
        return []

# -------------------
# Fetch NASA ADS
# -------------------
def fetch_nasa_ads(query, max_results=20, token=None):
    if not token:
        st.warning("NASA ADS API token not found! Add it to secrets.")
        return []
    try:
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
    except Exception as e:
        st.error(f"NASA ADS fetch failed: {e}")
        return []

# -------------------
# Hugging Face AI
# -------------------
def generate_ai_answer(question, docs, max_docs=5):
    context = "\n\n".join([doc["Abstract"] or "" for doc in docs[:max_docs]])
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    
    hf_token = st.secrets.get("HF_API_TOKEN")
    if not hf_token:
        return "HF API token not found! Add it to secrets."
    
    hf_model_id = "google/flan-t5-large"
    hf_inference = InferenceApi(repo_id=hf_model_id, token=hf_token, task="text2text-generation")
    
    try:
        response = hf_inference(inputs=prompt, raw_response=True)
        return response.text.strip() if hasattr(response, "text") else "No answer generated."
    except Exception as e:
        return f"AI generation failed: {e}"

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="SciSearch", layout="wide")

# Dark/light toggle
mode = st.sidebar.radio("Theme Mode:", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("<style>body{background-color:#222;color:#eee}</style>", unsafe_allow_html=True)

# Signup form
with st.sidebar.form("signup_form"):
    st.header("Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Sign Up")
    if submitted:
        st.success(f"Welcome, {username}!")
        st.session_state["user"] = username

# Search inputs
query = st.sidebar.text_input("Search term:", value="microgravity muscle atrophy")
source_filter = st.sidebar.multiselect("Sources:", ["PubMed", "NASA ADS"], default=["PubMed", "NASA ADS"])
max_results = st.sidebar.slider("Max results per source:", 5, 30, 15)
fetch_button = st.sidebar.button("Fetch Studies")

# session_state caching
if "all_results" not in st.session_state:
    st.session_state["all_results"] = []

if fetch_button:
    all_results = []
    if "PubMed" in source_filter:
        with st.spinner("Fetching PubMed studies..."):
            all_results.extend(fetch_pubmed(query, max_results))
    if "NASA ADS" in source_filter:
        nasa_token = st.secrets.get("NASA_ADS_API_TOKEN")
        with st.spinner("Fetching NASA ADS studies..."):
            all_results.extend(fetch_nasa_ads(query, max_results, nasa_token))
    for doc in all_results:
        doc["Summary"] = summarize(doc["Abstract"])
    st.session_state["all_results"] = all_results

# Tabs
tab1, tab2 = st.tabs(["Studies Explorer", "AI Q&A"])

with tab1:
    st.header("Explore Studies")
    if not st.session_state["all_results"]:
        st.info("Fetch studies using the sidebar first.")
    else:
        df = pd.DataFrame(st.session_state["all_results"])
        for idx, row in df.iterrows():
            st.subheader(row["Title"])
            st.markdown(f"**Authors:** {row['Authors']}  |  **Source:** {row['Source']}")
            if row.get("PubDate"):
                st.markdown(f"**Publication Date:** {row['PubDate']}")
            st.markdown(f"**Summary:** {row['Summary']}")
            st.markdown(f"[Open Study]({row['URL']})")
            with st.expander("Show Abstract"):
                st.write(row["Abstract"])
            st.markdown("---")
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="studies.csv")

with tab2:
    st.header("AI Q&A")
    question = st.text_area("Ask a question:")
    if st.button("Generate AI Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not st.session_state["all_results"]:
            st.warning("Fetch studies first using the sidebar.")
        else:
            with st.spinner("Generating AI answer..."):
                answer = generate_ai_answer(question, st.session_state["all_results"])
                st.markdown("### AI-generated Answer")
                st.write(answer)
                st.markdown("---")
                st.markdown("### Top Studies Used:")
                for doc in st.session_state["all_results"][:5]:
                    st.markdown(f"- [{doc['Title']}]({doc['URL']}) ({doc['Source']})")
