import streamlit as st
import pdfplumber
from transformers import pipeline
import textwrap

# Page title
st.set_page_config(page_title="Smart Document Summarizer", layout="wide")
st.title("📄 Smart Document Summarizer")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    # Extract text from PDF
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    if text.strip() == "":
        st.error("⚠️ No extractable text found (maybe it's a scanned PDF?). Try another file.")
    else:
        # Summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Split text into chunks (transformers have token limits)
        max_chunk = 1000
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

        st.info("⏳ Summarizing... Please wait.")
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        final_summary = " ".join(summaries)

        # Split summary into paragraphs of ~4-5 lines each
        paragraphs = textwrap.wrap(final_summary, width=500)  # Wrap long text into chunks
        formatted_paragraphs = []
        current_para = ""
        line_count = 0

        for sentence in final_summary.split(". "):
            current_para += sentence.strip() + ". "
            line_count += 1
            if line_count >= 4:  # around 4 sentences per paragraph
                formatted_paragraphs.append(current_para.strip())
                current_para = ""
                line_count = 0

        if current_para:  # add leftover text
            formatted_paragraphs.append(current_para.strip())

        # Display results
        st.subheader("📌 Document Summary (Key Insights)")
        for i, para in enumerate(formatted_paragraphs, start=1):
            st.success(f"🔹 {para}")

        # Download button
        st.download_button("💾 Download Summary", final_summary, file_name="summary.txt")
