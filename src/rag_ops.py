import os
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "chroma_db"

class AgenticRAG:
    def __init__(self):
        # 1. Local Embeddings (Must match ingest.py)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
        
        # 2. Groq LPU (Fast & Free)
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    def retrieve_candidates(self, job_description: str, k: int = 5):
        print(f"--- HOP 1: Retrieving top {k} chunks ---")
        results = self.db.similarity_search(job_description, k=10)
        return results

    def assess_relevance(self, job_description: str, retrieved_docs):
        print("--- HOP 2: Filtering Candidates (Agentic Filter) ---")
        relevant_candidates = []
        seen_sources = set()
        
        prompt = ChatPromptTemplate.from_template(
            """
            You are a strict technical recruiter. 
            Job Description: {job}
            
            Candidate Resume Excerpt: {resume_text}
            
            Does this candidate have RELEVANT skills/experience for this specific job? 
            Return strictly "YES" or "NO".
            """
        )
        chain = prompt | self.llm

        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown")
            
            if source in seen_sources:
                continue
            
            # Groq is fast, small sleep is enough
            time.sleep(1) 

            try:
                response = chain.invoke({"job": job_description, "resume_text": doc.page_content})
                decision = response.content.strip().upper()
                
                if "YES" in decision:
                    relevant_candidates.append(doc)
                    seen_sources.add(source)
                    print(f"✅ Kept: {source}")
                else:
                    print(f"❌ Dropped: {source}")
            except Exception as e:
                print(f"⚠️ Error processing {source}: {e}")
                
        return relevant_candidates

if __name__ == "__main__":
    # Quick Test
    rag = AgenticRAG()
    print("RAG Module Initialized.")