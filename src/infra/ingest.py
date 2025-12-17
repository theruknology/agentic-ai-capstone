import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = "chroma_db"

# --- GLOBAL INITIALIZATION (The Fix) ---
# We initialize these ONCE when the module is imported.
# This prevents re-downloading the model 50 times for a batch of 50 files.
print("⚙️ Initializing Embedding Model (One-time setup)...")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Vector DB Client once
vector_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_function
)

def load_documents(data_dir="data/inbox"):
    """Loads all PDFs from the directory."""
    documents = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    return documents

def chunk_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def save_to_chroma(chunks):
    """Saves chunks to the global ChromaDB instance."""
    if not chunks:
        print("⚠️ No chunks to save.")
        return

    # Use the global 'vector_db' client we created at the top
    # This keeps the connection open and avoids locking issues
    vector_db.add_documents(chunks)
    print(f"✅ Saved {len(chunks)} chunks to ChromaDB")
    
    # Force a persist (optional in newer Chroma versions but good for safety)
    # vector_db.persist()