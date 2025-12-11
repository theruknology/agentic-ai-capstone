import time
import json
import os
from typing import List, Dict, TypedDict
from dotenv import load_dotenv

# Import our modules from previous phases
from rag_ops import AgenticRAG
from pec_agents import PECAgents

load_dotenv()

# --- 1. Define System State (The Memory) ---
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
        """
        The Master Loop:
        1. RAG retrieves candidates.
        2. PEC Agents evaluate each one.
        3. Saves JSON report.
        """
        # --- Step 1: Load Job & Retrieve Candidates (Phase 1) ---
        print(f"📂 Loading Job: {job_file}")
        with open(job_file, "r") as f:
            job_desc = f.read()

        print("🔍 RAG: Retrieving & Filtering Candidates...")
        # Note: rag.retrieve_candidates already includes some sleep/dedup logic
        broad_matches = self.rag.retrieve_candidates(job_desc)
        refined_candidates = self.rag.assess_relevance(job_desc, broad_matches)

        if not refined_candidates:
            print("❌ No relevant candidates found. Exiting.")
            return

        print(f"✅ Found {len(refined_candidates)} qualified candidates. Starting deep evaluation...")
        
        final_reports = []

        # --- Step 2: Loop Through Candidates (Phase 3 Orchestration) ---
        for i, doc in enumerate(refined_candidates):
            source = doc.metadata.get("source", "Unknown")
            print(f"\n🚀 Processing Candidate {i+1}/{len(refined_candidates)}: {source}")
            
            # Initialize State
            state: HiringState = {
                "job_description": job_desc,
                "candidate_id": f"CAND_{i+1:03d}",
                "candidate_source": source,
                "resume_text": doc.page_content,
                "plan": {}, "screening": {}, "questions": {}, 
                "assessment": {}, "critique": {}, "status": "In Progress"
            }

            # --- NODE A: PLANNER ---
            # We sleep BEFORE calling the agent to respect limits
            self._throttle("Planner") 
            state["plan"] = self.agents.plan_evaluation(job_desc, state["resume_text"])
            
            # --- NODE B: EXECUTORS ---
            # 1. Screen
            self._throttle("Screener")
            state["screening"] = self.agents.screen_resume(job_desc, state["resume_text"])
            
            # 2. Interview
            self._throttle("Interviewer")
            state["questions"] = self.agents.generate_questions(job_desc, state["resume_text"])
            
            # 3. Assess
            self._throttle("Assessor")
            state["assessment"] = self.agents.create_assessment(job_desc)

            # --- NODE C: CRITIC ---
            self._throttle("Critic")
            state["critique"] = self.agents.critique_outputs(
                job_desc, state["screening"], state["questions"]
            )

            # --- Step 3: Finalize & Save ---
            state["status"] = "Completed"
            final_reports.append(state)
            self._save_report(state)

        print(f"\n🎉 Evaluation Complete. Reports saved in '{self.results_dir}/'")

    def _throttle(self, agent_name):
        """Phase 3 Strict Throttling: Ensures we never hit 429 errors."""
        wait_time = 15
        print(f"   ⏳ Orchestrator: Pausing {wait_time}s before triggering {agent_name}...")
        time.sleep(wait_time)

    def _save_report(self, state: HiringState):
        """Generates the final JSON output file."""
        filename = f"{self.results_dir}/{state['candidate_id']}_report.json"
        
        # Clean up data for saving (remove large raw text blocks if needed)
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
    # Ensure you have the job file
    job_path = "data/jobs/job_bioinformatics.txt"
    
    if os.path.exists(job_path):
        app = HiringOrchestrator()
        app.run_workflow(job_path)
    else:
        print(f"❌ Error: {job_path} not found.")