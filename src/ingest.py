import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv() # Ensure GOOGLE_API_KEY is in your .env file

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
    """Saves chunks to ChromaDB."""
    # Clear old DB if it exists (optional, for fresh start)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
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