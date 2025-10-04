import streamlit as st
import pandas as pd

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("structured_output.csv")

df = load_data()

st.title("SciSearch")
st.markdown("Explore the effects of space conditions on biological systems in NASA studies.")


st.sidebar.header("Filter Studies")


filtered_df = df.copy()


filter_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in filter_cols:
    unique_vals = df[col].dropna().unique()
    if 1 < len(unique_vals) < 100:  
        selected = st.sidebar.multiselect(f"{col}", unique_vals, key=col)
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]


search_query = st.text_input("🔎 Search summaries or outcomes (keywords)", "")

if search_query:
    filtered_df = filtered_df[
        filtered_df['Summary'].str.contains(search_query, case=False, na=False) |
        filtered_df['Outcome'].str.contains(search_query, case=False, na=False)
    ]


st.write(f"### Showing {len(filtered_df)} result(s)")

for idx, row in filtered_df.iterrows():
    st.subheader(row["Title"])
    st.markdown(f"**Organism:** {row.get('Organism', 'N/A')} | **Condition:** {row.get('Space Condition', 'N/A')} | **Topic:** {row.get('Topic', 'N/A')}")
    st.markdown(f"**Summary:** {row.get('Summary', 'N/A')}")
    st.markdown(f"**Outcome:** {row.get('Outcome', 'N/A')}")
    st.markdown(f"[View NASA Record]({row.get('Record URL', '#')})")
    st.markdown("---")


st.download_button(
    "Download Filtered Data as CSV",
    filtered_df.to_csv(index=False).encode('utf-8'),
    "filtered_nasa_bioscience.csv",
    "text/csv"
)
