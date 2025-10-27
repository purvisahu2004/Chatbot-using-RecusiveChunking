import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="📄 Chat with PDF", page_icon="🤖")
st.title("🤖 ChatBot using Recursive Chunking")

# --- Step 1: Load your PDF properly using pdfplumber ---
pdf_path = "NFHS-5_Phase-II_0.pdf"
text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text += page.extract_text() or ""

if not text.strip():
    st.error("⚠️ No text found! Your PDF might be scanned or image-based.")
else:
    st.success("✅ PDF text loaded successfully!")

# --- Step 2: Recursive chunking (larger chunk size for more context) ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # increased size
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_text(text)

st.write(f"📚 Created **{len(chunks)}** chunks.")

# --- Step 3: Embedding setup ---
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# --- Step 4: Take user question ---
query = st.text_input("💬 Ask a question about your PDF:")

# --- Step 5: Find the top 3 most similar chunks ---
if query:
    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_idx = np.argsort(sims)[-3:][::-1]  # top 3 chunks

    st.subheader("🤖 Answer:")
    combined_answer = " ".join([chunks[i] for i in top_idx])
    st.write(combined_answer)

    st.caption(f"(Top chunk similarity scores: {[round(sims[i], 2) for i in top_idx]})")



