import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# --- CONFIG ---
PDF_FILES = ["WOHG.pdf", "DAS.pdf"]
VECTORSTORE_PATH = "vectorstore_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Extract text from a single PDF ---
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(f"[✅] Extracted from {pdf_path}")
        return text
    except Exception as e:
        print(f"[❌] Failed to read {pdf_path}: {e}")
        return ""

# --- Chunk text for vector embedding ---
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        length_function=len
    )
    return splitter.create_documents([text])

# --- Combine all PDFs and create the vectorstore ---
def build_vectorstore(pdf_paths, output_path):
    print("[📄] Extracting text from all PDFs...")
    combined_text = ""
    for pdf in pdf_paths:
        combined_text += extract_text_from_pdf(pdf)
    
    print("[✂️] Chunking combined text...")
    documents = chunk_text(combined_text)
    print(f"[📚] Total chunks: {len(documents)}")

    print("[🔠] Creating embeddings and FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)

    print(f"[💾] Saving vectorstore to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    vectorstore.save_local(folder_path=output_path)
    print("[✅] Vectorstore created successfully!")

# --- MAIN ---
if __name__ == "__main__":
    build_vectorstore(PDF_FILES, VECTORSTORE_PATH)
