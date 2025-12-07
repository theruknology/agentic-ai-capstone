import os
import time  # <--- Added for rate limiting
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "chroma_db"

class AgenticRAG:
    def __init__(self):
        # 1. Local Embeddings (Free, Fast, Unlimited)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
        
        # 2. Gemini for Reasoning (Restricted to ~5 requests/min on Free Tier)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def retrieve_candidates(self, job_description: str, k: int = 5):
        print(f"--- HOP 1: Retrieving top {k} chunks ---")
        # We retrieve more chunks (k=10) knowing we will remove duplicates later
        results = self.db.similarity_search(job_description, k=10)
        return results

    def assess_relevance(self, job_description: str, retrieved_docs):
        print("--- HOP 2: Filtering Candidates (With Dedup & Throttling) ---")
        relevant_candidates = []
        seen_sources = set() # <--- Deduplication Tracker
        
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
            
            # --- FIX 1: DEDUPLICATION ---
            if source in seen_sources:
                continue # Skip this chunk if we already kept this candidate
            
            # --- FIX 2: RATE LIMIT THROTTLING ---
            # Sleep 15 seconds between calls to stay under the 5 RPM limit
            print(f"⏳ Sleeping 15s for API limits... (Processing {source})")
            time.sleep(15) 

            try:
                response = chain.invoke({"job": job_description, "resume_text": doc.page_content})
                decision = response.content.strip().upper()
                
                if "YES" in decision:
                    relevant_candidates.append(doc)
                    seen_sources.add(source) # Mark this candidate as "kept"
                    print(f"✅ Kept: {source}")
                else:
                    print(f"❌ Dropped: {source}")
                    # Note: We don't add to seen_sources here; 
                    # we might find a better chunk for this candidate later in the list.
            except Exception as e:
                print(f"⚠️ Error processing {source}: {e}")
                
        return relevant_candidates

    def analyze_skills(self, job_description: str, candidates):
        print("--- HOP 3: Analyzing Skill Gaps ---")
        analysis_results = []
        
        prompt = ChatPromptTemplate.from_template(
            """
            Compare this candidate to the job description.
            
            Job: {job}
            Resume: {resume}
            
            Output a concise JSON summary:
            {{
                "match_score": (0-100),
                "key_strengths": [list of 2-3 items],
                "missing_skills": [list of 2-3 items]
            }}
            """
        )
        chain = prompt | self.llm

        for doc in candidates:
            # --- FIX 2: RATE LIMIT THROTTLING ---
            print(f"⏳ Sleeping 15s for API limits... (Analyzing {doc.metadata.get('source')})")
            time.sleep(15)
            
            try:
                res = chain.invoke({"job": job_description, "resume": doc.page_content})
                analysis_results.append({
                    "source": doc.metadata.get("source"),
                    "analysis": res.content
                })
            except Exception as e:
                print(f"⚠️ Error analyzing candidate: {e}")
            
        return analysis_results

if __name__ == "__main__":
    test_job_path = "data/jobs/job_bioinformatics.txt"
    
    if os.path.exists(test_job_path):
        with open(test_job_path, "r") as f:
            job_desc = f.read()

        rag = AgenticRAG()
        
        # 1. Broad Search
        broad_docs = rag.retrieve_candidates(job_desc, k=5)
        
        if broad_docs:
            # 2. Filter
            refined_docs = rag.assess_relevance(job_desc, broad_docs)
            
            # 3. Analyze
            if refined_docs:
                final_output = rag.analyze_skills(job_desc, refined_docs)
                for item in final_output:
                    print(f"\nCandidate: {item['source']}")
                    print(item['analysis'])
            else:
                print("No suitable candidates found after filtering.")
    else:
        print("Job file not found.")