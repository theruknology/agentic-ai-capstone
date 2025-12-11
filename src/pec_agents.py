import time
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define Structured Outputs (Pydantic Models) ---
# These force the LLM to give us clean JSON we can use in code.

class Plan(BaseModel):
    steps: List[str] = Field(description="List of steps to evaluate the candidate.")
    logic: str = Field(description="Brief reasoning for this plan.")

class ScreeningResult(BaseModel):
    fit_score: float = Field(description="A score from 0.0 to 1.0 representing fit.")
    matching_skills: List[str] = Field(description="Skills from JD found in resume.")
    missing_skills: List[str] = Field(description="Skills in JD but missing in resume.")
    reasoning: str = Field(description="Short explanation of the score.")

class InterviewQuestions(BaseModel):
    questions: List[str] = Field(description="5-10 technical interview questions.")
    difficulty: str = Field(description="Overall difficulty level (Junior/Mid/Senior).")

class SkillAssessment(BaseModel):
    tasks: List[str] = Field(description="2-3 practical coding tasks or case studies.")
    evaluation_criteria: str = Field(description="What to look for in the solution.")

class Critique(BaseModel):
    critique_passed: bool = Field(description="True if the evaluation seems accurate/consistent.")
    issues: List[str] = Field(description="List of potential hallucinations or inconsistencies.")
    feedback: str = Field(description="Suggestions for the human recruiter.")

# --- 2. The PEC Agent Classes ---

class PECAgents:
    def __init__(self):
        # We use a lower temperature for strict JSON generation
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    def _safe_invoke(self, chain, inputs, agent_name):
        """Helper to handle Rate Limits automatically."""
        print(f"⏳ {agent_name} is thinking... (Sleeping 15s for Rate Limit)")
        time.sleep(15) # Safety buffer for Free Tier
        try:
            return chain.invoke(inputs)
        except Exception as e:
            print(f"⚠️ Error in {agent_name}: {e}")
            return None

    # --- AGENT 1: PLANNER ---
    def plan_evaluation(self, job_desc: str, candidate_summary: str):
        parser = JsonOutputParser(pydantic_object=Plan)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Lead Recruiter Planning Agent.
            Job: {job}
            Candidate Summary: {candidate}
            
            Create a plan to evaluate this candidate. 
            Standard steps usually include: ["Screening", "Interview Generation", "Skill Assessment"].
            
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "candidate": candidate_summary}, "PlannerAgent")

    # --- AGENT 2: EXECUTOR (Screener) ---
    def screen_resume(self, job_desc: str, resume_text: str):
        parser = JsonOutputParser(pydantic_object=ScreeningResult)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Technical Screener. Compare the resume to the job description.
            
            Job: {job}
            Resume: {resume}
            
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "resume": resume_text}, "ScreenerAgent")

    # --- AGENT 3: EXECUTOR (Interviewer) ---
    def generate_questions(self, job_desc: str, resume_text: str):
        parser = JsonOutputParser(pydantic_object=InterviewQuestions)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Senior Interviewer. Generate technical questions tailored to this candidate's gaps and strengths.
            
            Job: {job}
            Resume: {resume}
            
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "resume": resume_text}, "InterviewerAgent")

    # --- AGENT 4: EXECUTOR (Assessor) ---
    def create_assessment(self, job_desc: str):
        parser = JsonOutputParser(pydantic_object=SkillAssessment)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Technical Lead. Design a practical take-home assignment for this role.
            
            Job: {job}
            
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc}, "AssessorAgent")

    # --- AGENT 5: CRITIC ---
    def critique_outputs(self, job_desc: str, screening: dict, questions: dict):
        parser = JsonOutputParser(pydantic_object=Critique)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Quality Assurance Critic. Review the work of the other agents.
            
            Job: {job}
            Screening Result: {screening}
            Proposed Questions: {questions}
            
            Check for:
            1. Are the questions actually relevant to the Job Description?
            2. Did the screener miss any obvious red flags?
            3. Is the tone professional?
            
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {
            "job": job_desc, 
            "screening": str(screening), 
            "questions": str(questions)
        }, "CriticAgent")

# --- Manual Test Runner (Simulating Orchestration) ---
if __name__ == "__main__":
    agents = PECAgents()
    
    # Load Dummy Data
    try:
        with open("data/jobs/job_bioinformatics.txt", "r") as f:
            job_desc = f.read()
        # We simulate the "Resume Text" that RAG would have retrieved
        resume_excerpt = "Experienced in Python and R. Worked on RNA-seq data analysis. PhD in Computational Biology."
    except FileNotFoundError:
        print("❌ Please ensure data/jobs/job_bioinformatics.txt exists.")
        exit()

    print("\n--- 🏁 STARTING PEC PIPELINE TEST ---")
    
    # 1. PLAN
    plan = agents.plan_evaluation(job_desc, resume_excerpt)
    print(f"\n📋 PLAN: {plan}")
    
    # 2. EXECUTE (Screen)
    if plan:
        screen_result = agents.screen_resume(job_desc, resume_excerpt)
        print(f"\n🔍 SCREENING: {json.dumps(screen_result, indent=2)}")
        
        # 3. EXECUTE (Interview)
        interview_q = agents.generate_questions(job_desc, resume_excerpt)
        print(f"\n🎤 QUESTIONS: {json.dumps(interview_q, indent=2)}")
        
        # 4. EXECUTE (Assess)
        assessment = agents.create_assessment(job_desc)
        print(f"\n💻 ASSESSMENT: {json.dumps(assessment, indent=2)}")
        
        # 5. CRITIQUE
        if screen_result and interview_q:
            critique = agents.critique_outputs(job_desc, screen_result, interview_q)
            print(f"\n⚖️ CRITIC REPORT: {json.dumps(critique, indent=2)}")

    print("\n--- ✅ TEST COMPLETE ---")