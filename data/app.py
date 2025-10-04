import streamlit as st
import pandas as pd

st.set_page_config(page_title="SciSearch", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("merged_output.csv")



df = load_data()

st.title("🧬 SciSearch: NASA Bioscience Explorer")
st.markdown("Explore the effects of space conditions on biological systems in NASA studies. Filter, search, and download curated data on life sciences experiments conducted in space.")

# --- Sidebar filters ---
st.sidebar.header("🔍 Filter Studies")

filtered_df = df.copy()

# Limit filtering to relevant tag columns
filter_cols = ['Organism', 'Space Condition', 'Topic', 'Document Type']
for col in filter_cols:
    if col in df.columns:
        unique_vals = df[col].dropna().unique()
        if 1 < len(unique_vals) < 100:
            selected = st.sidebar.multiselect(f"{col}", sorted(unique_vals), key=col)
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]

# --- Free-text search ---
search_query = st.text_input("🔎 Search summaries or outcomes (keywords)", "")

if search_query:
    search_cols = ['Summary']
    if 'Outcome' in filtered_df.columns:
        search_cols.append('Outcome')

    mask = pd.Series(False, index=filtered_df.index)
    for col in search_cols:
        mask |= filtered_df[col].str.contains(search_query, case=False, na=False)

    filtered_df = filtered_df[mask]

# --- Results Count ---
st.write(f"### Showing {len(filtered_df)} result(s)")

# --- Display Results ---
for idx, row in filtered_df.iterrows():
    st.subheader(row.get("Title", "Untitled"))
    st.markdown(f"**Organism:** {row.get('Organism', 'N/A')} | **Condition:** {row.get('Space Condition', 'N/A')} | **Topic:** {row.get('Topic', 'N/A')}")
    st.markdown(f"**Summary:** {row.get('Summary', 'N/A')}")
    st.markdown(f"**Outcome:** {row.get('Outcome', 'N/A')}")
    st.markdown(f"[🔗 View NASA Record]({row.get('Record URL', '#')})")

    with st.expander("Show Full Study Details"):
        st.markdown(f"**Document Type:** {row.get('Document Type', 'N/A')}")
        st.markdown(f"**Author(s):** {row.get('Author Names', 'N/A')}")
        st.markdown(f"**Abstract:** {row.get('Abstract', 'N/A')}")
        st.markdown(f"**Funding Numbers:** {row.get('Funding Numbers', 'N/A')}")
        st.markdown(f"**Subject Categories:** {row.get('Subject Categories', 'N/A')}")

    st.markdown("---")

# --- Download Button ---
st.download_button(
    "📥 Download Filtered Data as CSV",
    filtered_df.to_csv(index=False).encode('utf-8'),
    "filtered_nasa_bioscience.csv",
    "text/csv"
)
from pubmed import pubmed_search, pubmed_fetch_abstracts, extract_abstracts, summarize_text

query = st.text_input("Search PubMed")

if query:
    with st.spinner("Searching PubMed..."):
        pmids = pubmed_search(query)
        xml = pubmed_fetch_abstracts(pmids)
        abstracts = extract_abstracts(xml)
        for abstract in abstracts:
            summary = summarize_text(abstract)
            st.markdown("**Summary:** " + summary)
            st.markdown("**Original Abstract:** " + abstract)
