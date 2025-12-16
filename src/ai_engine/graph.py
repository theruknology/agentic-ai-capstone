import time
import json
import os
import sys
from typing import TypedDict, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ai_engine.rag import AgenticRAG
from src.ai_engine.agents import PECAgents
from src.infra.db import get_redis_client
from src.infra.logger import logger # New Logger

load_dotenv()

class HiringState(TypedDict):
    job_description: str
    candidate_id: str
    candidate_email: str
    candidate_source: str
    resume_text: str
    
    # Outputs
    plan: Dict
    screening: Dict
    questions: Dict
    assessment: Dict
    critique: Dict
    
    # Logic
    iteration_count: int 
    feedback: str

# --- Nodes ---
class GraphNodes:
    def __init__(self):
        self.agents = PECAgents()

    def planner_node(self, state: HiringState):
        logger.info("🔹 NODE: Planner")
        plan = self.agents.plan_evaluation(state["job_description"], state["resume_text"])
        return {"plan": plan}

    def screener_node(self, state: HiringState):
        count = state.get("iteration_count", 0) + 1
        logger.info(f"🔹 NODE: Screener (Attempt {count})")
        feedback = state.get("feedback", "")
        result = self.agents.screen_resume(state["job_description"], state["resume_text"], feedback)
        return {"screening": result, "iteration_count": count}

    def interviewer_node(self, state: HiringState):
        logger.info("🔹 NODE: Interviewer")
        questions = self.agents.generate_questions(state["job_description"], state["resume_text"])
        return {"questions": questions}

    def assessor_node(self, state: HiringState):
        logger.info("🔹 NODE: Assessor")
        assessment = self.agents.create_assessment(state["job_description"])
        return {"assessment": assessment}

    def critic_node(self, state: HiringState):
        logger.info("🔹 NODE: Critic")
        critique = self.agents.critique_outputs(
            state["job_description"], state["screening"], state["questions"]
        )
        return {"critique": critique, "feedback": critique.get("critic_feedback", "")}

def route_critique(state: HiringState):
    critique = state.get("critique", {})
    count = state.get("iteration_count", 0)
    
    if critique.get("critique_passed", True):
        return "end"
    if count >= 3:
        logger.warning("⚠️ Max Retries Reached. Force Finishing.")
        return "end"
    return "refine"

# --- Orchestrator ---
class HiringOrchestrator:
    def __init__(self):
        self.rag = AgenticRAG()
        self.nodes = GraphNodes()
        self.results_dir = "reports"
        os.makedirs(self.results_dir, exist_ok=True)
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
        logger.info(f"📂 Loading Job: {job_file}")
        with open(job_file, "r") as f:
            job_desc = f.read()

        # Hop 1 & 2
        broad_matches = self.rag.retrieve_candidates(job_desc, k=5)
        refined_candidates = self.rag.assess_relevance(job_desc, broad_matches)

        if not refined_candidates:
            logger.warning("❌ No relevant candidates found.")
            return

        logger.info(f"✅ Starting Graph for {len(refined_candidates)} candidates...")
        app = self.build_graph()

        for doc in refined_candidates:
            source_path = doc.metadata.get("source", "Unknown")
            filename = os.path.basename(source_path)
            
            # Redis Lookup
            candidate_key = f"candidate:{filename}"
            candidate_info = self.redis.hgetall(candidate_key)
            real_name = candidate_info.get("name", f"Unknown ({filename})")
            
            logger.info(f"🚀 Processing: {real_name}")

            initial_state = {
                "job_description": job_desc,
                "candidate_id": real_name,
                "candidate_email": candidate_info.get("email", "N/A"),
                "candidate_source": source_path,
                "resume_text": doc.page_content,
                "iteration_count": 0
            }

            final_state = app.invoke(initial_state)

            # --- HOP 3 INTEGRATION: GAP VERIFICATION ---
            missing = final_state['screening'].get('missing_skills', [])
            if missing:
                # We verify against the original doc
                verified = self.rag.verify_missing_skills(doc, missing)
                final_state['screening']['missing_skills'] = verified
                
            self._save_report(final_state)

    def _save_report(self, state: HiringState):
        safe_id = "".join([c for c in state['candidate_id'] if c.isalnum() or c in (' ', '_')]).replace(" ", "_")
        filename = f"{self.results_dir}/{safe_id}_report.json"
        
        # SCHEMA ALIGNMENT WITH RUBRIC
        output_data = {
            "evaluation_metadata": {
                "candidate_id": state["candidate_id"],
                "candidate_email": state["candidate_email"],
                "source": state["candidate_source"],
                "timestamp": time.ctime(),
                "model": "Llama-3-70B-Groq"
            },
            "match_score": state["screening"].get("match_score", 0),
            "ranked_candidates": [state["candidate_id"]],
            "critic_feedback": state.get("critique", {}).get("critic_feedback", "None"),
            "full_details": {
                "plan": state.get("plan"),
                "screening": state.get("screening"),
                "interview_questions": state.get("questions"),
                "skill_assessment": state.get("assessment"),
                "critic_review": state.get("critique")
            }
        }
        
        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"💾 Report saved: {filename}")