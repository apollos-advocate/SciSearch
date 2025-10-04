import streamlit as st
import pandas as pd
import requests
from xml.etree import ElementTree as ET
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from huggingface_hub import InferenceApi

# --- Setup nltk ---
def ensure_nltk_resource(resource_name):
    try:
        nltk.data.find(f'tokenizers/{resource_name}')
    except LookupError:
        nltk.download(resource_name)

ensure_nltk_resource('punkt')
ensure_nltk_resource('punkt_tab')

# --- Summarizer ---
summarizer = LsaSummarizer()

def summarize(text, sentences_count=3):
    if not text or text.strip() == "":
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

# --- Fetch PubMed ---
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
        url = f"https://pubmed.ncbi.nlm.nih.gov/{article.findtext('.//PMID')}"
        records.append({
            "Title": title,
            "Abstract": abstract,
            "Authors": authors,
            "Source": "PubMed",
            "URL": url
        })
    return records

# --- Fetch NASA ADS ---
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

# --- Setup HuggingFace Inference API ---
hf_token = st.secrets.get("HF_API_TOKEN")
hf_model_id = "google/flan-t5-large"  # Swap for your preferred model
hf_inference = InferenceApi(repo_id=hf_model_id, token=hf_token)

def generate_ai_answer(question, docs):
    context = "\n\n".join([doc["Abstract"] or "" for doc in docs[:5]])
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    response = hf_inference(inputs=prompt)
    if isinstance(response, str):
        return response
    return response.get("generated_text", "No answer generated.")

# --- Streamlit UI ---

st.set_page_config(page_title="SciSearch", layout="wide")
st.title("SciSearch: PubMed & NASA ADS Explorer")

# Tabs for UI
tab1, tab2 = st.tabs(["Studies Explorer", "AI Overview"])

with tab1:
    st.header("Search & Explore Studies")
    query = st.text_input("Enter search term:", value="microgravity muscle atrophy")

    source_filter = st.multiselect(
        "Filter by source",
        options=["PubMed", "NASA ADS"],
        default=["PubMed", "NASA ADS"]
    )

    max_results = st.slider("Max results per source", 5, 30, 15)

    if st.button("Search Studies"):
        all_results = []
        if "PubMed" in source_filter:
            with st.spinner("Fetching PubMed studies..."):
                pubmed_results = fetch_pubmed(query, max_results)
                all_results.extend(pubmed_results)
        if "NASA ADS" in source_filter:
            with st.spinner("Fetching NASA ADS studies..."):
                nasa_token = st.secrets["NASA_ADS_API_TOKEN"]
                nasa_results = fetch_nasa_ads(query, max_results, nasa_token)
                all_results.extend(nasa_results)

        if all_results:
            df = pd.DataFrame(all_results)
            df["Summary"] = df["Abstract"].apply(summarize)

            st.success(f"Found {len(df)} studies.")
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
        else:
            st.warning("No studies found for the given query and sources.")

with tab2:
    st.header("AI Overview")
    question = st.text_area("Ask a question related to your search:")
    if st.button("Generate AI Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Fetching studies for AI context..."):
                combined_docs = []
                # Use same filters as before or default to both
                # Could optionally store filters in session state for sync
                # For now, fetch from both sources again
                if "PubMed" in source_filter:
                    combined_docs.extend(fetch_pubmed(query, max_results))
                if "NASA ADS" in source_filter:
                    combined_docs.extend(fetch_nasa_ads(query, max_results, st.secrets["NASA_ADS_API_TOKEN"]))

            if combined_docs:
                with st.spinner("Generating AI answer..."):
                    answer = generate_ai_answer(question, combined_docs)
                    st.markdown("### AI-generated Answer")
                    st.write(answer)

                    st.markdown("---")
                    st.markdown("### Top Relevant Studies Used:")
                    for doc in combined_docs[:5]:
                        st.markdown(f"- [{doc['Title']}]({doc['URL']}) ({doc['Source']})")
            else:
                st.warning("No studies found to generate AI answer.")
