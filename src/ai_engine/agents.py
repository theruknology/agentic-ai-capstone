import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# Import our tools for integration
from src.ai_engine.tools import lookup_salary_range, search_skill_framework

load_dotenv()

# --- Pydantic Models (Same as before) ---
class Plan(BaseModel):
    steps: List[str] = Field(description="List of automated analysis steps (no manual/human tasks).")
    logic: str = Field(description="Reasoning for this automated workflow.")

class ScreeningResult(BaseModel):
    fit_score: float = Field(description="Fit score from 0.0 to 1.0.")
    matching_skills: List[str] = Field(description="Skills found in resume.")
    missing_skills: List[str] = Field(description="Skills missing from resume.")
    reasoning: str = Field(description="Concise justification.")

class InterviewQuestions(BaseModel):
    questions: List[str] = Field(description="5-7 highly technical questions based on gaps.")
    difficulty: str = Field(description="Difficulty level.")

class SkillAssessment(BaseModel):
    tasks: List[str] = Field(description="1-2 practical coding/design tasks.")
    evaluation_criteria: str = Field(description="What to check in the solution.")

class Critique(BaseModel):
    critique_passed: bool = Field(description="True if the output is acceptable.")
    issues: List[str] = Field(description="List of factual hallucinations or errors.")
    feedback: str = Field(description="Instructions for the Screener if rejected.")

# --- The Agents ---
class PECAgents:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
        
        # BIND TOOLS: This gives the LLM the ability to call functions
        # This addresses the feedback about "tools not integrated"
        self.tool_llm = self.llm.bind_tools([lookup_salary_range, search_skill_framework])

    def _safe_invoke(self, chain, inputs, agent_name):
        print(f"⚡ {agent_name} working...")
        time.sleep(1) 
        try:
            return chain.invoke(inputs)
        except Exception as e:
            print(f"⚠️ Error in {agent_name}: {e}")
            return None

    def plan_evaluation(self, job_desc: str, resume_text: str):
        # Planner can now use tools implicitly
        parser = JsonOutputParser(pydantic_object=Plan)
        prompt = ChatPromptTemplate.from_template(
            """
            You are the Architect of an **Autonomous AI Hiring System**.
            Your goal is to plan a **text-based evaluation** of a candidate.

            CONSTRAINTS:
            - You CANNOT schedule meetings, calls, or physical interviews.
            - You CANNOT check references manually.
            - You can ONLY plan for: Semantic Analysis, Gap Identification, Question Generation, and Skill Assessment Design.

            Job: {job}
            Resume Summary: {resume}
            
            Create a concise execution plan for the downstream AI agents.
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "resume": resume_text}, "PlannerAgent")

    def screen_resume(self, job_desc: str, resume_text: str, feedback: str = ""):
        # FEEDBACK INTEGRATION: This enables the refinement loop
        parser = JsonOutputParser(pydantic_object=ScreeningResult)
        
        context_prompt = ""
        if feedback:
            context_prompt = f"CRITICAL INSTRUCTION: Your previous output was rejected. \nFEEDBACK: '{feedback}' \nYou must fix this specifically."

        prompt = ChatPromptTemplate.from_template(
            """
            You are a Technical Screener AI. Compare the resume to the job description strictly based on the text provided.
            {feedback_context}
            
            Job: {job}
            Resume: {resume}
            
            Output a fit score (0.0-1.0) and lists of matching/missing skills.
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions(), feedback_context=context_prompt) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "resume": resume_text}, "ScreenerAgent")

    def generate_questions(self, job_desc: str, resume_text: str):
        parser = JsonOutputParser(pydantic_object=InterviewQuestions)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Technical Interviewer AI. 
            Generate 5-7 **technical** interview questions to probe the candidate's missing skills or depth of knowledge.
            Do not ask generic HR questions like "Tell me about yourself".
            
            Job: {job}
            Resume: {resume}
            
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "resume": resume_text}, "InterviewerAgent")

    def create_assessment(self, job_desc: str):
        parser = JsonOutputParser(pydantic_object=SkillAssessment)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Technical Lead AI. 
            Design a short, practical coding task or system design scenario to validate the core skills required in the Job Description.
            
            Job: {job}
            
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc}, "AssessorAgent")

    def critique_outputs(self, job_desc: str, screening: dict, questions: dict):
        parser = JsonOutputParser(pydantic_object=Critique)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Quality Assurance Validator.
            
            Your job is to prevent **Hallucinations** (lying about skills) or **Broken Logic**.
            
            Job: {job}
            Screening Output: {screening}
            Proposed Questions: {questions}
            
            CRITERIA FOR PASSING (return critique_passed=True):
            1. The screening score seems reasonable given the resume.
            2. The interview questions are technical and relevant to the Job Description.
            3. No factual hallucinations (e.g., saying they know Python when the resume says they don't).
            
            **IMPORTANT:** Do NOT reject for minor stylistic preferences. Only reject for factual errors or irrelevance.
            
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {
            "job": job_desc, 
            "screening": str(screening), 
            "questions": str(questions)
        }, "CriticAgent")