import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

from src.infra.logger import logger, log_latency
try:
    from src.ai_engine.tools import lookup_salary_range, search_skill_framework
except ImportError:
    from tools import lookup_salary_range, search_skill_framework

load_dotenv()

# --- Pydantic Models (Strict Schema) ---
class Plan(BaseModel):
    steps: List[str] = Field(description="List of automated analysis steps (no manual/human tasks).")
    logic: str = Field(description="Reasoning for this automated workflow.")

class ScreeningResult(BaseModel):
    match_score: float = Field(description="Score 0-100.")
    ranked_candidates: List[str] = Field(description="List of candidate IDs.")
    matching_skills: List[str] = Field(description="Skills found in resume.")
    missing_skills: List[str] = Field(description="Skills missing from resume.")
    reasoning: str = Field(description="Detailed analysis.")

class InterviewQuestions(BaseModel):
    questions: List[str] = Field(description="5-7 highly technical questions based on gaps.")
    difficulty: str = Field(description="Difficulty level.")

class SkillAssessment(BaseModel):
    tasks: List[str] = Field(description="1-2 practical coding/design tasks.")
    evaluation_criteria: str = Field(description="What to check in the solution.")

class Critique(BaseModel):
    critique_passed: bool = Field(description="True if the output is acceptable.")
    critic_feedback: str = Field(description="Feedback for refinement.")
    issues: List[str] = Field(description="List of factual hallucinations or errors.")

# --- The Agents ---
class PECAgents:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
        # BIND TOOLS: The model now knows these functions exist
        self.tool_llm = self.llm.bind_tools([lookup_salary_range, search_skill_framework])

    def _safe_invoke(self, chain, inputs, agent_name):
        logger.info(f"⚡ {agent_name} working...")
        try:
            return chain.invoke(inputs)
        except Exception as e:
            logger.error(f"⚠️ Error in {agent_name}: {e}")
            return None

    @log_latency
    def plan_evaluation(self, job_desc: str, resume_text: str, messages=None):
        """
        PLANNER AGENT (ReAct Mode):
        Returns a raw AIMessage. If it contains tool_calls, the Graph will execute them.
        """
        if messages is None:
            messages = []
        
        # 1. DETAILED SYSTEM PROMPT (Restored Constraints)
        system_msg = """
        You are the Architect of an **Autonomous AI Hiring System**.
        Your goal is to plan a **text-based evaluation** of a candidate.

        CONSTRAINTS:
        - You CANNOT schedule meetings, calls, or physical interviews.
        - You CANNOT check references manually.
        - You can ONLY plan for: Semantic Analysis, Gap Identification, Question Generation, and Skill Assessment Design.

        TOOLS AVAILABLE:
        1. 'lookup_salary_range': Use this IF the resume mentions salary expectations.
        2. 'search_skill_framework': Use this IF you need to verify if a skill is relevant.

        INSTRUCTIONS:
        - Check the resume. Does it mention salary? If yes, CALL THE TOOL.
        - Does it list obscure skills? If yes, CALL THE TOOL.
        - Once you have enough info, output the FINAL PLAN in this JSON format:
        {
            "steps": ["step1", "step2"],
            "logic": "explanation"
        }
        """
        
        # 2. Construct Message History
        # If this is the first turn, we add the Initial Prompt.
        # If it's a loop (messages exist), we just append to the history.
        input_messages = [SystemMessage(content=system_msg)] + messages
        if not messages:
            # First run: Add the user input
            input_messages.append(HumanMessage(content=f"Job: {job_desc}\nResume: {resume_text}"))

        # 3. Invoke with Tools enabled
        return self.tool_llm.invoke(input_messages)

    @log_latency
    def screen_resume(self, job_desc: str, resume_text: str, feedback: str = ""):
        parser = JsonOutputParser(pydantic_object=ScreeningResult)
        
        # Detailed Context Prompt
        context_prompt = ""
        if feedback:
            context_prompt = f"CRITICAL INSTRUCTION: Your previous output was rejected. \nFEEDBACK: '{feedback}' \nYou must fix this specifically."

        prompt = ChatPromptTemplate.from_template(
            """
            You are a Technical Screener AI. Compare the resume to the job description.
            {feedback_context}
            
            Job: {job}
            Resume: {resume}
            
            Output a match_score (0-100) and strict lists of matching/missing skills.
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions(), feedback_context=context_prompt) | self.llm | parser
        return self._safe_invoke(chain, {"job": job_desc, "resume": resume_text}, "ScreenerAgent")

    @log_latency
    def generate_questions(self, job_desc: str, resume_text: str):
        parser = JsonOutputParser(pydantic_object=InterviewQuestions)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Technical Interviewer AI. 
            Generate 5-7 **technical** interview questions to probe the candidate's missing skills.
            Do not ask generic HR questions like "Tell me about yourself".
            
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
            You are a Technical Lead AI. 
            Design a short, practical coding task or system design scenario.
            
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
            You are a Quality Assurance Validator (Pragmatic Critic).
            
            Your job is to prevent **Hallucinations** or **Broken Logic**.
            
            Job: {job}
            Screening Output: {screening}
            Proposed Questions: {questions}
            
            CRITERIA:
            1. The match_score seems reasonable.
            2. Questions are technical and relevant.
            3. No factual hallucinations.
            
            **IMPORTANT:** Do NOT reject for minor stylistic preferences. Only reject for factual errors.
            
            {format_instructions}
            """
        )
        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser
        return self._safe_invoke(chain, {
            "job": job_desc, 
            "screening": str(screening), 
            "questions": str(questions)
        }, "CriticAgent")