import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import numpy as np

# --- Streamlit setup ---
st.set_page_config(page_title="📄 Chat with PDF", page_icon="🤖")
st.title("🤖 Chat with Your PDF Document (Recursive Chunking)")

# --- Step 1: Load your existing PDF file ---
# ✅ Replace this with the path of your PDF file
pdf_path = "NFHS-5_Phase-II_0.pdf"

pdf_reader = PdfReader(pdf_path)
text = ""
for page in pdf_reader.pages:
    text += page.extract_text() or ""

st.success("✅ PDF loaded successfully!")

# --- Step 2: Recursive Chunking ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # each chunk ≈ 500 characters
    chunk_overlap=50,    # small overlap to preserve context
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_text(text)
st.write(f"📚 Created **{len(chunks)} semantic chunks** from your PDF.")

# --- Step 3: Create embeddings for chunks ---
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# --- Step 4: Take user question ---
query = st.text_input("💬 Ask a question about NFHW survey:")

# --- Step 5: Find the most similar chunk ---
if query:
    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, embeddings)
    best_idx = np.argmax(sims)
    best_score = sims[0][best_idx]

    if best_score < 0.3:
        st.info("🤖 Sorry, I couldn’t find relevant information in the document.")
    else:
        st.write("🤖 **Answer:**", chunks[best_idx])
        st.caption(f"(Similarity Score: {best_score:.2f})")
