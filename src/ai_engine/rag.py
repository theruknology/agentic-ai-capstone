import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.infra.logger import logger, log_latency

load_dotenv()
DB_PATH = "chroma_db"

class AgenticRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    @log_latency
    def retrieve_candidates(self, job_description: str, k: int = 5):
        logger.info(f"🔍 HOP 1: Retrieving top {k} chunks...")
        # Fetch more initially to allow for filtering
        results = self.db.similarity_search(job_description, k=k*2)
        return results

    @log_latency
    def assess_relevance(self, job_description: str, retrieved_docs):
        logger.info("🔍 HOP 2: Agentic Relevance Filter...")
        relevant_candidates = []
        seen_sources = set()
        
        prompt = ChatPromptTemplate.from_template(
            """
            You are a strict recruiter.
            Job: {job}
            Resume Excerpt: {resume_text}
            
            Is this candidate relevant? Return ONLY "YES" or "NO".
            """
        )
        chain = prompt | self.llm

        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown")
            if source in seen_sources: continue
            
            try:
                # Fast check
                res = chain.invoke({"job": job_description, "resume_text": doc.page_content})
                if "YES" in res.content.upper():
                    relevant_candidates.append(doc)
                    seen_sources.add(source)
                    logger.info(f"   ✅ Kept Candidate: {source}")
                else:
                    logger.info(f"   ❌ Dropped Candidate: {source}")
            except Exception as e:
                logger.error(f"Filter error: {e}")
                
        return relevant_candidates

    @log_latency
    def verify_missing_skills(self, doc, missing_skills):
        """
        HOP 3: Gap Verification.
        If Agent says 'Missing Python', we re-query the DB specifically for 'Python' 
        in this resume to ensure it wasn't just missed in the chunking.
        """
        if not missing_skills:
            return []

        logger.info(f"🔍 HOP 3: Verifying {len(missing_skills)} missing skills...")
        verified_missing = []
        
        for skill in missing_skills:
            # Re-query specifically for this skill
            query = f"{skill} experience usage context"
            # In a real app, you would filter by source. 
            # Here we search globally but check if the result matches our current doc's content.
            results = self.db.similarity_search(query, k=3)
            
            found = False
            for r in results:
                # If we find the skill in a chunk that belongs to the same source
                if r.metadata.get("source") == doc.metadata.get("source"):
                    logger.warning(f"   ⚠️ Correction: Found '{skill}' in deeper search!")
                    found = True
                    break
            
            if not found:
                verified_missing.append(skill)
                
        return verified_missing