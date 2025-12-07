import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "chroma_db"

class AgenticRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)      
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def retrieve_candidates(self, job_description: str, k: int = 5):
        """Hop 1: Broad Retrieval via Vector Search"""
        print(f"--- HOP 1: Retrieving top {k} chunks ---")
        results = self.db.similarity_search(job_description, k=k)
        return results

    def assess_relevance(self, job_description: str, retrieved_docs):
        """Hop 2: LLM Relevance Filter"""
        print("--- HOP 2: Filtering Candidates ---")
        relevant_candidates = []
        
        # Simple prompt to filter noise
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
            response = chain.invoke({"job": job_description, "resume_text": doc.page_content})
            decision = response.content.strip().upper()
            
            if "YES" in decision:
                relevant_candidates.append(doc)
                print(f"✅ Kept: {source}")
            else:
                print(f"❌ Dropped: {source}")
                
        return relevant_candidates

    def analyze_skills(self, job_description: str, candidates):
        """Hop 3: Skill Gap Analysis"""
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
            res = chain.invoke({"job": job_description, "resume": doc.page_content})
            analysis_results.append({
                "source": doc.metadata.get("source"),
                "analysis": res.content
            })
            
        return analysis_results

# --- Simple Test Runner ---
if __name__ == "__main__":
    # Load one of the jobs we created
    with open("data/jobs/job_bioinformatics.txt", "r") as f:
        job_desc = f.read()

    rag = AgenticRAG()
    
    # 1. Broad Search
    broad_docs = rag.retrieve_candidates(job_desc, k=5)
    
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