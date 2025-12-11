import time
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# --- Pydantic Models (Structured Output) ---
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
    difficulty: str = Field(description="Overall difficulty level.")

class SkillAssessment(BaseModel):
    tasks: List[str] = Field(description="2-3 practical coding tasks or case studies.")
    evaluation_criteria: str = Field(description="What to look for in the solution.")

class Critique(BaseModel):
    critique_passed: bool = Field(description="True if the evaluation seems accurate.")
    issues: List[str] = Field(description="List of potential hallucinations.")
    feedback: str = Field(description="Suggestions for improvement.")

# --- The Agents ---
class PECAgents:
    def __init__(self):
        # Llama 3.3 is excellent at following JSON instructions
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    def _safe_invoke(self, chain, inputs, agent_name):
        print(f"⚡ {agent_name} working...")
        time.sleep(1) # Minimal sleep for Groq
        try:
            return chain.invoke(inputs)
        except Exception as e:
            print(f"⚠️ Error in {agent_name}: {e}")
            return None

    def plan_evaluation(self, job_desc: str, candidate_summary: str):
        parser = JsonOutputParser(pydantic_object=Plan)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Lead Recruiter Planning Agent.
            Job: {job}
            Candidate Summary: {candidate}
            Create a plan to evaluate this candidate.
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "candidate": candidate_summary}, "PlannerAgent")

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

    def generate_questions(self, job_desc: str, resume_text: str):
        parser = JsonOutputParser(pydantic_object=InterviewQuestions)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Senior Interviewer. Generate technical questions tailored to this candidate.
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
            You are a Technical Lead. Design a practical take-home assignment.
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
            You are a Quality Assurance Critic. Review the work of the other agents.
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