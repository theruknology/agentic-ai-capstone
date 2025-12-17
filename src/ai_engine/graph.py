import time
import json
import os
import sys
from typing import TypedDict, Dict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ai_engine.rag import AgenticRAG
from src.ai_engine.agents import PECAgents
from src.ai_engine.tools import lookup_salary_range, search_skill_framework
from src.infra.db import get_redis_client
from src.infra.logger import logger

load_dotenv()

# --- State Definition (Same as before) ---
class HiringState(TypedDict):
    job_description: str
    candidate_id: str
    candidate_email: str
    candidate_source: str
    resume_text: str
    planner_messages: List[BaseMessage]
    plan: Dict
    screening: Dict
    questions: Dict
    assessment: Dict
    critique: Dict
    iteration_count: int 
    feedback: str

# --- Nodes (Keep exact same logic) ---
class GraphNodes:
    def __init__(self):
        self.agents = PECAgents()
        self.tools = [lookup_salary_range, search_skill_framework]
        self.tool_node = ToolNode(self.tools)

    def planner_node(self, state: HiringState):
        logger.info("🔹 NODE: Planner (ReAct)")
        messages = state.get("planner_messages", [])
        response = self.agents.plan_evaluation(state["job_description"], state["resume_text"], messages)
        return {"planner_messages": messages + [response]}

    def tool_execution_node(self, state: HiringState):
        logger.info("🛠️ NODE: Tool Executor")
        last_message = state["planner_messages"][-1]
        tool_outputs = self.tool_node.invoke({"messages": [last_message]})
        return {"planner_messages": state["planner_messages"] + tool_outputs["messages"]}

    def planner_parser_node(self, state: HiringState):
        logger.info("📄 NODE: Parsing Final Plan")
        last_message = state["planner_messages"][-1]
        content = last_message.content
        try:
            json_str = content[content.find("{"):content.rfind("}")+1]
            plan = json.loads(json_str)
        except:
            plan = {"steps": ["Manual Review"], "logic": "Agent failed to output valid JSON."}
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

# --- Routing ---
def route_planner(state: HiringState):
    if state["planner_messages"][-1].tool_calls: return "tools"
    return "parser"

def route_critique(state: HiringState):
    critique = state.get("critique", {})
    count = state.get("iteration_count", 0)
    if critique.get("critique_passed", True): return "end"
    if count >= 3: return "end"
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
        workflow.add_node("tools", self.nodes.tool_execution_node)
        workflow.add_node("planner_parser", self.nodes.planner_parser_node)
        workflow.add_node("screener", self.nodes.screener_node)
        workflow.add_node("interviewer", self.nodes.interviewer_node)
        workflow.add_node("assessor", self.nodes.assessor_node)
        workflow.add_node("critic", self.nodes.critic_node)

        workflow.set_entry_point("planner")
        workflow.add_conditional_edges("planner", route_planner, {"tools": "tools", "parser": "planner_parser"})
        workflow.add_edge("tools", "planner")
        workflow.add_edge("planner_parser", "screener")
        workflow.add_edge("screener", "interviewer")
        workflow.add_edge("interviewer", "assessor")
        workflow.add_edge("assessor", "critic")
        workflow.add_conditional_edges("critic", route_critique, {"end": END, "refine": "screener"})
        return workflow.compile()

    def run_workflow(self, job_file: str):
        logger.info(f"📂 Loading Job: {job_file}")
        with open(job_file, "r") as f: job_desc = f.read()

        # Hop 1: Retrieve All
        # We retrieve, but we need to track WHO we retrieved to check who got dropped
        broad_matches = self.rag.retrieve_candidates(job_desc, k=5)
        
        # Hop 2: Filter
        refined_candidates = self.rag.assess_relevance(job_desc, broad_matches)
        
        # --- NEW LOGIC: Handle Dropped Candidates ---
        # 1. Get IDs of passed candidates
        passed_sources = {doc.metadata.get("source") for doc in refined_candidates}
        
        # 2. Identify dropped
        dropped_docs = [doc for doc in broad_matches if doc.metadata.get("source") not in passed_sources]
        
        # 3. Generate Rejection Reports for Dropped Docs
        # Deduplicate by source first
        processed_sources = set()
        
        for doc in dropped_docs:
            source = doc.metadata.get("source", "Unknown")
            if source in processed_sources or source in passed_sources:
                continue
                
            logger.info(f"🚫 Saving Rejection Report for: {source}")
            self._save_rejection_report(doc)
            processed_sources.add(source)

        if not refined_candidates:
            logger.warning("❌ No qualified candidates to process.")
            return

        # --- Normal Graph Processing (Passed Candidates) ---
        logger.info(f"✅ Starting Graph for {len(refined_candidates)} candidates...")
        app = self.build_graph()

        for doc in refined_candidates:
            source = doc.metadata.get("source")
            if source in processed_sources: continue # Skip if already handled (safety)
            
            # (Run the graph logic)
            filename = os.path.basename(source)
            candidate_info = self.redis.hgetall(f"candidate:{filename}")
            real_name = candidate_info.get("name", f"Unknown ({filename})")
            
            logger.info(f"🚀 Processing: {real_name}")
            
            initial_state = {
                "job_description": job_desc,
                "candidate_id": real_name,
                "candidate_email": candidate_info.get("email", "N/A"),
                "candidate_source": source,
                "resume_text": doc.page_content,
                "iteration_count": 0,
                "planner_messages": []
            }

            final_state = app.invoke(initial_state)

            missing = final_state['screening'].get('missing_skills', [])
            if missing:
                verified = self.rag.verify_missing_skills(doc, missing)
                final_state['screening']['missing_skills'] = verified
                
            self._save_report(final_state)
            processed_sources.add(source)

    # --- NEW FUNCTION ---
    def _save_rejection_report(self, doc):
        """Creates a simplified report for candidates dropped at Hop 2."""
        source = doc.metadata.get("source", "Unknown")
        filename = os.path.basename(source)
        
        # Try to get Name from Redis
        candidate_info = self.redis.hgetall(f"candidate:{filename}")
        real_name = candidate_info.get("name", f"Unknown ({filename})")
        email = candidate_info.get("email", "N/A")
        
        safe_id = "".join([c for c in real_name if c.isalnum() or c in (' ', '_')]).replace(" ", "_")
        report_path = f"{self.results_dir}/{safe_id}_report.json"
        
        output_data = {
            "evaluation_metadata": {
                "candidate_id": real_name,
                "candidate_email": email,
                "source": source,
                "timestamp": time.ctime(),
                "model": "RAG-Filter (Rejected)"
            },
            "match_score": 0, # Automatic 0
            "ranked_candidates": [],
            "full_details": {
                "screening": {
                    "reasoning": "REJECTED AT INITIAL FILTER: Resume did not match basic keyword relevance thresholds for this job.",
                    "missing_skills": ["Fundamental Job Alignment"],
                    "matching_skills": []
                }
            }
        }
        
        with open(report_path, "w") as f:
            json.dump(output_data, f, indent=2)

    def _save_report(self, state: HiringState):
        # (Same as before)
        safe_id = "".join([c for c in state['candidate_id'] if c.isalnum() or c in (' ', '_')]).replace(" ", "_")
        filename = f"{self.results_dir}/{safe_id}_report.json"
        
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