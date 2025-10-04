import streamlit as st
from transformers import pipeline

st.title("Transformers Test")

try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    st.success("Transformers pipeline loaded successfully!")
except Exception as e:
    st.error(f"Failed to load transformers: {e}")

