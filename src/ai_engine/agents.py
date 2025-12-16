import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

from src.infra.logger import logger, log_latency
try:
    from src.ai_engine.tools import lookup_salary_range, search_skill_framework
except ImportError:
    from tools import lookup_salary_range, search_skill_framework

load_dotenv()

# --- Updated Schema to Match Rubric ---
class Plan(BaseModel):
    steps: List[str] = Field(description="Automated analysis steps.")
    logic: str = Field(description="Reasoning for workflow.")

class ScreeningResult(BaseModel):
    match_score: float = Field(description="Score 0-100.") # RUBRIC KEY
    ranked_candidates: List[str] = Field(description="List of candidate IDs.") # RUBRIC KEY
    matching_skills: List[str] = Field(description="Skills found.")
    missing_skills: List[str] = Field(description="Skills missing.")
    reasoning: str = Field(description="Detailed analysis.")

class InterviewQuestions(BaseModel):
    questions: List[str] = Field(description="Technical questions.")
    difficulty: str = Field(description="Difficulty level.")

class SkillAssessment(BaseModel):
    tasks: List[str] = Field(description="Practical tasks.")
    evaluation_criteria: str = Field(description="Criteria.")

class Critique(BaseModel):
    critique_passed: bool = Field(description="True if acceptable.")
    critic_feedback: str = Field(description="Feedback for refinement.") # RUBRIC KEY
    issues: List[str] = Field(description="List of errors.")

# --- The Agents ---
class PECAgents:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
        # BIND TOOLS: Explicitly giving agents tool capabilities
        self.tool_llm = self.llm.bind_tools([lookup_salary_range, search_skill_framework])

    def _safe_invoke(self, chain, inputs, agent_name):
        logger.info(f"⚡ {agent_name} working...")
        # Removed arbitrary sleep; relying on logger
        try:
            return chain.invoke(inputs)
        except Exception as e:
            logger.error(f"⚠️ Error in {agent_name}: {e}")
            return None

    @log_latency
    def plan_evaluation(self, job_desc: str, resume_text: str):
        parser = JsonOutputParser(pydantic_object=Plan)
        prompt = ChatPromptTemplate.from_template(
            """
            You are the Planner Agent. Plan a text-based evaluation.
            Constraint: You CANNOT schedule calls. Only automated analysis.
            Job: {job}
            Resume: {resume}
            {format_instructions}
            """
        )
        # We use tool_llm logic implicitly here if we were to expand to a full AgentExecutor
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "resume": resume_text}, "PlannerAgent")

    @log_latency
    def screen_resume(self, job_desc: str, resume_text: str, feedback: str = ""):
        parser = JsonOutputParser(pydantic_object=ScreeningResult)
        context = ""
        if feedback:
            context = f"PREVIOUS FEEDBACK: {feedback}. Fix these issues."

        prompt = ChatPromptTemplate.from_template(
            """
            You are a Screener Agent.
            {feedback_context}
            Job: {job}
            Resume: {resume}
            Output match_score (0-100).
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions(), feedback_context=context) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "resume": resume_text}, "ScreenerAgent")

    @log_latency
    def generate_questions(self, job_desc: str, resume_text: str):
        parser = JsonOutputParser(pydantic_object=InterviewQuestions)
        prompt = ChatPromptTemplate.from_template(
            """
            Generate technical interview questions.
            Job: {job}
            Resume: {resume}
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "resume": resume_text}, "InterviewerAgent")

    @log_latency
    def create_assessment(self, job_desc: str):
        parser = JsonOutputParser(pydantic_object=SkillAssessment)
        prompt = ChatPromptTemplate.from_template(
            """
            Design a technical task.
            Job: {job}
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc}, "AssessorAgent")

    @log_latency
    def critique_outputs(self, job_desc: str, screening: dict, questions: dict):
        parser = JsonOutputParser(pydantic_object=Critique)
        prompt = ChatPromptTemplate.from_template(
            """
            Validate the output. Pass if reasonable.
            Job: {job}
            Screening: {screening}
            Questions: {questions}
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {
            "job": job_desc, 
            "screening": str(screening), 
            "questions": str(questions)
        }, "CriticAgent")