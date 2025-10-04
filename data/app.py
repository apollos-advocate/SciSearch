import streamlit as st
import pandas as pd

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("structured_output.csv")

df = load_data()

st.title("🚀 NASA Bioscience Dashboard")
st.markdown("Explore the effects of space conditions on biological systems in NASA studies.")

# --- Filters ---
st.sidebar.header("🔎 Filter Studies")

organism = st.sidebar.multiselect("Organism", df["Organism"].dropna().unique())
condition = st.sidebar.multiselect("Space Condition", df["Space Condition"].dropna().unique())
topic = st.sidebar.multiselect("Topic", df["Topic"].dropna().unique())

filtered_df = df.copy()

if organism:
    filtered_df = filtered_df[filtered_df["Organism"].isin(organism)]
if condition:
    filtered_df = filtered_df[filtered_df["Space Condition"].isin(condition)]
if topic:
    filtered_df = filtered_df[filtered_df["Topic"].isin(topic)]

st.write(f"### Showing {len(filtered_df)} filtered results")

# --- Table Display ---
for idx, row in filtered_df.iterrows():
    st.subheader(row["Title"])
    st.markdown(f"**Organism:** {row['Organism']} | **Condition:** {row['Space Condition']} | **Topic:** {row['Topic']}")
    st.markdown(f"**Summary:** {row['Summary']}")
    st.markdown(f"**Outcome:** {row.get('Outcome', 'N/A')}")
    st.markdown(f"[🔗 View NASA Record]({row['Record URL']})")
    st.markdown("---")

# --- Download button ---
st.download_button(
    "📥 Download Filtered Data as CSV",
    filtered_df.to_csv(index=False).encode('utf-8'),
    "filtered_nasa_bioscience.csv",
    "text/csv"
)
