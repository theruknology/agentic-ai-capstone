import time
import json
import os
from typing import TypedDict
from dotenv import load_dotenv

from rag_ops import AgenticRAG
from pec_agents import PECAgents

load_dotenv()

class HiringState(TypedDict):
    job_description: str
    candidate_id: str
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

    def run_workflow(self, job_file: str):
        print(f"📂 Loading Job: {job_file}")
        with open(job_file, "r") as f:
            job_desc = f.read()

        print("🔍 RAG: Retrieving & Filtering Candidates...")
        # Hop 1 & 2
        broad_matches = self.rag.retrieve_candidates(job_desc)
        refined_candidates = self.rag.assess_relevance(job_desc, broad_matches)

        if not refined_candidates:
            print("❌ No relevant candidates found. Exiting.")
            return

        print(f"✅ Found {len(refined_candidates)} qualified candidates. Starting PEC pipeline...")
        
        for i, doc in enumerate(refined_candidates):
            source = doc.metadata.get("source", "Unknown")
            print(f"\n🚀 Processing Candidate {i+1}/{len(refined_candidates)}: {source}")
            
            state: HiringState = {
                "job_description": job_desc,
                "candidate_id": f"CAND_{i+1:03d}",
                "candidate_source": source,
                "resume_text": doc.page_content,
                "plan": {}, "screening": {}, "questions": {}, 
                "assessment": {}, "critique": {}, "status": "In Progress"
            }

            # Run Agents with light throttling
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

        print(f"\n🎉 Evaluation Complete. Reports saved in '{self.results_dir}/'")

    def _throttle(self, agent_name):
        wait_time = 3 
        # print(f"   ⏳ Pausing {wait_time}s for Groq...") 
        time.sleep(wait_time)

    def _save_report(self, state: HiringState):
        filename = f"{self.results_dir}/{state['candidate_id']}_report.json"
        
        output_data = {
            "meta": {
                "id": state["candidate_id"],
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
    else:
        print(f"❌ Error: {job_path} not found.")