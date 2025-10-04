import streamlit as st
import pandas as pd

st.set_page_config(page_title="SciSearch", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_csv("structured_output.csv")

df = load_data()

st.title("🧬 SciSearch: Space Life Science Explorer")
st.markdown("Search, explore, and download life science studies from PubMed related to space, microgravity, and biology.")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Studies")

filtered_df = df.copy()

# Filter by Source (optional if you later include multiple sources like NASA)
if "Source" in df.columns:
    sources = df["Source"].dropna().unique()
    selected_sources = st.sidebar.multiselect("Source", sources, default=sources)
    filtered_df = filtered_df[filtered_df["Source"].isin(selected_sources)]

# --- Keyword Search ---
search = st.text_input("🔎 Search summaries or titles", "")

if search:
    mask = (
        df["Title"].str.contains(search, case=False, na=False) |
        df["Summary"].str.contains(search, case=False, na=False)
    )
    filtered_df = filtered_df[mask]

# --- Show Results ---
st.write(f"### {len(filtered_df)} studies found")

for idx, row in filtered_df.iterrows():
    st.subheader(row.get("Title", "Untitled"))
    st.markdown(f"**Authors:** {row.get('Authors', 'N/A')}")
    st.markdown(f"**Summary:** {row.get('Summary', 'No summary available.')}")
    
    with st.expander("Show Full Abstract"):
        st.write(row.get("Abstract", "No abstract available."))

    st.markdown("---")

# --- Download Button ---
st.download_button(
    "📥 Download Filtered Results as CSV",
    filtered_df.to_csv(index=False).encode("utf-8"),
    "filtered_studies.csv",
    "text/csv"
)
