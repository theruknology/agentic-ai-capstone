import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv() 

# Configuration
DATA_PATH = "data/resumes/"
DB_PATH = "chroma_db"

def load_documents():
    """Loads all PDFs from the data folder."""
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")
    return documents

def chunk_documents(documents):
    """Splits documents into smaller chunks for the vector DB."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    # CHANGE 2: Use a local model (all-MiniLM-L6-v2 is standard, fast, and light)
    print("Initializing local embeddings (this runs on your laptop)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # This might take 10-20 seconds the first time to download the model
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=DB_PATH
    )
    print(f"Saved chunks to {DB_PATH}")

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    save_to_chroma(chunks)