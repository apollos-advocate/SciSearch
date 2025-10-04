import streamlit as st
import pandas as pd

st.set_page_config(page_title="SciSearch", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("structured_output.csv")

df = load_data()

st.title("🧬 SciSearch: NASA & PubMed Life Science Explorer")
st.markdown("""
Explore biological studies related to space conditions.
Filter, search summaries, and download data.
""")

# Sidebar filters
st.sidebar.header("🔍 Filter Studies")

filtered_df = df.copy()

filter_cols = ['Authors', 'Source']  # add more if you want to filter by other columns

for col in filter_cols:
    if col in df.columns:
        options = sorted(df[col].dropna().unique())
        selected = st.sidebar.multiselect(f"{col}", options)
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]

# Search box for free text search on Title and Summary
search_query = st.text_input("🔎 Search titles or summaries (keywords)")

if search_query:
    mask = (
        filtered_df['Title'].str.contains(search_query, case=False, na=False) |
        filtered_df['Summary'].str.contains(search_query, case=False, na=False)
    )
    filtered_df = filtered_df[mask]

st.write(f"### Showing {len(filtered_df)} result(s)")

for _, row in filtered_df.iterrows():
    st.subheader(row.get("Title", "Untitled"))
    st.markdown(f"**Authors:** {row.get('Authors', 'N/A')} | **Source:** {row.get('Source', 'N/A')}")
    st.markdown(f"**Summary:** {row.get('Summary', 'N/A')}")
    st.markdown("---")

# Download filtered data
st.download_button(
    "📥 Download filtered data as CSV",
    filtered_df.to_csv(index=False).encode('utf-8'),
    "filtered_studies.csv",
    "text/csv"
)
