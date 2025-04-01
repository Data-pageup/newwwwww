import streamlit as st
from summarization import summarize_text
from main_topic import extract_main_topic_textrank
from main_branches import extract_main_branches_from_sentences
from sub_branches import extract_sub_branches
import fitz  # PyMuPDF for PDF text extraction
import pytesseracgit init
git add .
git commit -m "Initial commit"
t  # OCR for scanned PDFs
from pdf2image import convert_from_bytes  # Convert scanned PDFs to images
import re
import tempfile
import shutil

# Ensure Tesseract is installed correctly
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")  # Auto-detect path

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file, handling both selectable and scanned text."""
    text = ""

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Open PDF
    with fitz.open(tmp_file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"

    # If no text extracted, use OCR (for scanned PDFs)
    if not text.strip():
        uploaded_file.seek(0)  # ✅ Reset file pointer before re-reading
        images = convert_from_bytes(uploaded_file.read())
        text = "\n".join([pytesseract.image_to_string(img) for img in images])

    return text.strip()

def clean_text(text):
    """Perform text cleaning & preprocessing."""
    text = re.sub(r"\n+", " ", text)  # Remove excessive newlines
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    text = re.sub(r"-\s", "", text)  # Fix broken words at line breaks
    text = re.sub(r"[^\w\s.,!?]", "", text)  # Remove special characters except common punctuation
    text = re.sub(r"\.{3,}", "...", text)  # Normalize long sequences of dots
    return text.strip()

# Streamlit UI
st.title("📄 PDF Text Extractor & Cleaner")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    st.write("🔄 Extracting text... Please wait.")
    
    raw_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_text(raw_text)

    st.subheader("📜 Extracted Text")
    st.text_area("Raw Text", raw_text, height=300)

    st.subheader("🧹 Cleaned & Processed Text")
    st.text_area("Cleaned Text", cleaned_text, height=300)

    # ✅ Summarization
    summarized_text = summarize_text(cleaned_text)
    st.subheader("📄 Summarized Text")
    st.text_area("Summary", summarized_text, height=200)

    # ✅ Main Topic
    main_topic = extract_main_topic_textrank(summarized_text)
    st.subheader("Main Topic")
    st.write(main_topic)

    # ✅ Main Branches
    main_branches = extract_main_branches_from_sentences(summarized_text)
    st.subheader("Main Branches")
    st.write(main_branches)

    # ✅ Sub-Branches
    sub_branches = extract_sub_branches(summarized_text, main_branches)
    st.subheader("Sub Branches")
    st.write(sub_branches)

    # ✅ Download option
    st.download_button("📥 Download Summary", summarized_text, file_name="summary.txt")

from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    """Summarize extracted text using BART."""
    if len(text) < 50:
        return "Text is too short to summarize."

    # Limit text size for model input
    max_chunk_size = 1024
    text = text[:max_chunk_size]  # Truncate long text

    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']
from summa import keywords

def extract_main_topic_textrank(text):
    """Extract main topic using TextRank."""
    return keywords.keywords(text, words=1, split=False)
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_main_branches_from_sentences(text):
    """Extract main branches (noun phrases) from sentences."""
    doc = nlp(text)
    main_branches = []

    for sent in doc.sents:
        noun_phrases = [chunk.text for chunk in sent.noun_chunks]

        if noun_phrases:
            first_noun_phrase = noun_phrases[0]  # Get first noun phrase
            main_branches.append(first_noun_phrase)

    return main_branches

import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

def extract_sub_branches(text, main_branches):
    """Extract sub-branches related to main branches."""
    doc = nlp(text)
    sub_branches_map = defaultdict(set)

    for sent in doc.sents:
        sent_text = sent.text.lower()

        for branch in main_branches:
            branch_lower = branch.lower()
            if branch_lower in sent_text:
                words = sent_text.split()
                branch_idx = words.index(branch_lower.split()[0])

                for chunk in sent.noun_chunks:
                    if chunk.start > branch_idx:
                        phrase = chunk.text.strip().lower()

                        if phrase == branch_lower:  # Avoid duplicate branch
                            continue

                        if len(phrase.split()) > 1:  # Keep meaningful phrases
                            sub_branches_map[branch].add(phrase)

                # Extract verbs linked to the main branch
                for token in sent:
                    if token.head.text.lower() in branch_lower and token.pos_ in {"VERB", "NOUN"}:
                        sub_branches_map[branch].add(token.text.lower())

    return {branch: sorted(sub_branches_map[branch]) for branch in sub_branches_map}

