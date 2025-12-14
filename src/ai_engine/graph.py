import time
import json
import os
import sys
from typing import TypedDict, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Fix imports to find the infra module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ai_engine.rag import AgenticRAG
from src.ai_engine.agents import PECAgents
from src.infra.db import get_redis_client  # <--- NEW IMPORT

load_dotenv()

# We no longer need METADATA_FILE
# METADATA_FILE = "data/candidates.json" 

class HiringState(TypedDict):
    job_description: str
    candidate_id: str
    candidate_email: str
    candidate_source: str
    resume_text: str
    
    # Agent Outputs
    plan: Dict
    screening: Dict
    questions: Dict
    assessment: Dict
    critique: Dict
    
    # Logic Control
    iteration_count: int 
    feedback: str

# --- Nodes (Keep logic same) ---
class GraphNodes:
    def __init__(self):
        self.agents = PECAgents()

    def planner_node(self, state: HiringState):
        print("   🔹 NODE: Planner")
        plan = self.agents.plan_evaluation(state["job_description"], state["resume_text"])
        return {"plan": plan}

    def screener_node(self, state: HiringState):
        count = state.get("iteration_count", 0) + 1
        print(f"   🔹 NODE: Screener (Attempt {count})")
        feedback = state.get("feedback", "")
        result = self.agents.screen_resume(state["job_description"], state["resume_text"], feedback)
        return {"screening": result, "iteration_count": count}

    def interviewer_node(self, state: HiringState):
        print("   🔹 NODE: Interviewer")
        questions = self.agents.generate_questions(state["job_description"], state["resume_text"])
        return {"questions": questions}

    def assessor_node(self, state: HiringState):
        print("   🔹 NODE: Assessor")
        assessment = self.agents.create_assessment(state["job_description"])
        return {"assessment": assessment}

    def critic_node(self, state: HiringState):
        print("   🔹 NODE: Critic")
        critique = self.agents.critique_outputs(
            state["job_description"], state["screening"], state["questions"]
        )
        return {"critique": critique}

def route_critique(state: HiringState):
    critique = state.get("critique", {})
    count = state.get("iteration_count", 0)
    if critique.get("critique_passed", True):
        return "end"
    if count >= 3:
        print("   ⚠️ Max Retries Reached.")
        return "end"
    return "refine"

# --- The Orchestrator (UPDATED) ---
class HiringOrchestrator:
    def __init__(self):
        self.rag = AgenticRAG()
        self.nodes = GraphNodes()
        self.results_dir = "reports"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # CONNECT TO REDIS
        self.redis = get_redis_client()

    def build_graph(self):
        workflow = StateGraph(HiringState)
        workflow.add_node("planner", self.nodes.planner_node)
        workflow.add_node("screener", self.nodes.screener_node)
        workflow.add_node("interviewer", self.nodes.interviewer_node)
        workflow.add_node("assessor", self.nodes.assessor_node)
        workflow.add_node("critic", self.nodes.critic_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "screener")
        workflow.add_edge("screener", "interviewer")
        workflow.add_edge("interviewer", "assessor")
        workflow.add_edge("assessor", "critic")

        workflow.add_conditional_edges(
            "critic",
            route_critique,
            {"end": END, "refine": "screener"}
        )
        return workflow.compile()

    def run_workflow(self, job_file: str):
        print(f"📂 Loading Job: {job_file}")
        with open(job_file, "r") as f:
            job_desc = f.read()

        print("🔍 RAG: Retrieving & Filtering Candidates...")
        broad_matches = self.rag.retrieve_candidates(job_desc, k=5)
        refined_candidates = self.rag.assess_relevance(job_desc, broad_matches)

        if not refined_candidates:
            print("❌ No relevant candidates found. Exiting.")
            return

        print(f"✅ Found {len(refined_candidates)} qualified candidates. Starting Graph Pipeline...")
        app = self.build_graph()

        for i, doc in enumerate(refined_candidates):
            source_path = doc.metadata.get("source", "Unknown")
            filename = os.path.basename(source_path)
            
            # --- FIX: LOOKUP REDIS INSTEAD OF JSON ---
            candidate_key = f"candidate:{filename}"
            # hgetall returns a dict like {'name': 'John', 'email': 'john@...'}
            candidate_info = self.redis.hgetall(candidate_key)
            
            real_name = candidate_info.get("name", f"Unknown ({filename})")
            email = candidate_info.get("email", "N/A")
            
            print(f"\n🚀 Processing: {real_name} ({email})")

            initial_state = {
                "job_description": job_desc,
                "candidate_id": real_name,
                "candidate_email": email,
                "candidate_source": source_path,
                "resume_text": doc.page_content,
                "iteration_count": 0
            }

            final_state = app.invoke(initial_state)
            self._save_report(final_state)

    def _save_report(self, state: HiringState):
        safe_id = "".join([c for c in state['candidate_id'] if c.isalnum() or c in (' ', '_')]).replace(" ", "_")
        filename = f"{self.results_dir}/{safe_id}_report.json"
        
        output_data = {
            "meta": {
                "id": state["candidate_id"],
                "email": state["candidate_email"],
                "source": state["candidate_source"],
                "timestamp": time.ctime(),
                "iterations": state.get("iteration_count", 1)
            },
            "evaluation": {
                "plan": state.get("plan"),
                "screening": state.get("screening"),
                "interview_questions": state.get("questions"),
                "skill_assessment": state.get("assessment"),
                "critic_review": state.get("critique")
            }
        }
        
        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"   💾 Saved report to {filename}")

if __name__ == "__main__":
    job_path = "data/jobs/current_job.txt"
    if os.path.exists(job_path):
        orchestrator = HiringOrchestrator()
        orchestrator.run_workflow(job_path)
    else:
        print(f"❌ Error: {job_path} not found.")