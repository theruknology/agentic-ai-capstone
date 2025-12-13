import time
import json
import os
from typing import TypedDict
from dotenv import load_dotenv

from rag_ops import AgenticRAG
from pec_agents import PECAgents

load_dotenv()

METADATA_FILE = "data/candidates.json"

class HiringState(TypedDict):
    job_description: str
    candidate_id: str
    candidate_email: str
    candidate_source: str
    resume_text: str
    plan: dict
    screening: dict
    questions: dict
    assessment: dict
    critique: dict
    status: str

class HiringOrchestrator:
    def __init__(self):
        self.rag = AgenticRAG()
        self.agents = PECAgents()
        self.results_dir = "reports"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load Applicant Database
        self.metadata_db = {}
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                try:
                    self.metadata_db = json.load(f)
                except:
                    self.metadata_db = {}

    def run_workflow(self, job_file: str):
        print(f"📂 Loading Job: {job_file}")
        with open(job_file, "r") as f:
            job_desc = f.read()

        print("🔍 RAG: Retrieving & Filtering Candidates...")
        broad_matches = self.rag.retrieve_candidates(job_desc)
        refined_candidates = self.rag.assess_relevance(job_desc, broad_matches)

        if not refined_candidates:
            print("❌ No relevant candidates found. Exiting.")
            return

        print(f"✅ Found {len(refined_candidates)} qualified candidates. Starting PEC pipeline...")
        
        for i, doc in enumerate(refined_candidates):
            source_path = doc.metadata.get("source", "Unknown")
            filename = os.path.basename(source_path)
            
            # --- RESOLVE IDENTITY ---
            candidate_info = self.metadata_db.get(filename, {})
            real_name = candidate_info.get("name", f"Unknown ({filename})")
            email = candidate_info.get("email", "No Email Provided")
            
            print(f"\n🚀 Processing: {real_name} ({email})")
            
            state: HiringState = {
                "job_description": job_desc,
                "candidate_id": real_name,
                "candidate_email": email,
                "candidate_source": source_path,
                "resume_text": doc.page_content,
                "plan": {}, "screening": {}, "questions": {}, 
                "assessment": {}, "critique": {}, "status": "In Progress"
            }

            # Run Agents (Groq is fast, small sleep is fine)
            self._throttle("Planner") 
            state["plan"] = self.agents.plan_evaluation(job_desc, state["resume_text"])
            
            self._throttle("Screener")
            state["screening"] = self.agents.screen_resume(job_desc, state["resume_text"])
            
            self._throttle("Interviewer")
            state["questions"] = self.agents.generate_questions(job_desc, state["resume_text"])
            
            self._throttle("Assessor")
            state["assessment"] = self.agents.create_assessment(job_desc)

            self._throttle("Critic")
            state["critique"] = self.agents.critique_outputs(
                job_desc, state["screening"], state["questions"]
            )

            state["status"] = "Completed"
            self._save_report(state)

    def _throttle(self, agent_name):
        time.sleep(1) # Minimal sleep for Groq

    def _save_report(self, state: HiringState):
        # Create safe filename
        safe_id = "".join([c for c in state['candidate_id'] if c.isalnum() or c in (' ', '_')]).replace(" ", "_")
        filename = f"{self.results_dir}/{safe_id}_report.json"
        
        output_data = {
            "meta": {
                "id": state["candidate_id"],
                "email": state["candidate_email"],
                "source": state["candidate_source"],
                "timestamp": time.ctime()
            },
            "evaluation": {
                "plan": state["plan"],
                "screening": state["screening"],
                "interview_questions": state["questions"],
                "skill_assessment": state["assessment"],
                "critic_review": state["critique"]
            }
        }
        
        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"   💾 Saved report to {filename}")

if __name__ == "__main__":
    job_path = "data/jobs/current_job.txt"
    if os.path.exists(job_path):
        app = HiringOrchestrator()
        app.run_workflow(job_path)