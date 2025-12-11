import os
import time
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define the Tools ---
# The docstring inside the function is CRITICAL. 
# It tells the LLM *when* and *how* to use the tool.

@tool
def lookup_salary_range(role: str, location: str) -> dict:
    """
    Useful for finding the market salary range for a specific job role and location.
    Use this when evaluating if a candidate is too expensive or within budget.
    """
    # In a real app, this would query an API (like Glassdoor or Levels.fyi).
    # For the capstone, we return deterministic mock data.
    print(f"🛠️ TOOL CALL: Looking up salary for '{role}' in '{location}'...")
    
    data = {
        "Data Scientist": {"NY": "120k-160k", "SF": "140k-180k", "Remote": "110k-150k"},
        "Software Engineer": {"NY": "130k-170k", "SF": "150k-200k", "Remote": "120k-160k"},
        "Bioinformatics Scientist": {"Boston": "100k-140k", "SF": "120k-160k", "Remote": "95k-135k"}
    }
    
    # Fuzzy matching logic for the demo
    key = "Software Engineer"
    if "data" in role.lower(): key = "Data Scientist"
    if "bio" in role.lower(): key = "Bioinformatics Scientist"
    
    loc_key = "Remote"
    if "ny" in location.lower() or "new york" in location.lower(): loc_key = "NY"
    if "sf" in location.lower() or "francisco" in location.lower(): loc_key = "SF"
    if "boston" in location.lower(): loc_key = "Boston"
    
    result = data.get(key, {}).get(loc_key, "80k-120k")
    return {"role": key, "location": loc_key, "market_range": result}

@tool
def search_skill_framework(skill: str) -> list:
    """
    Useful for finding related technical skills or expanding on a specific acronym.
    Use this to understand if a candidate's skill (e.g., 'React') is relevant to a requirement (e.g., 'Frontend').
    """
    print(f"🛠️ TOOL CALL: Searching framework for skill '{skill}'...")
    
    # Mock taxonomy
    taxonomy = {
        "python": ["Django", "Flask", "Pandas", "NumPy", "Scripting"],
        "machine learning": ["TensorFlow", "PyTorch", "Scikit-learn", "Deep Learning"],
        "react": ["JavaScript", "Frontend", "Redux", "Hooks", "Web Development"],
        "ngs": ["Bioinformatics", "Genomics", "DNA Sequencing", "Illumina"]
    }
    
    return taxonomy.get(skill.lower(), ["No specific related skills found in database."])

# --- 2. Setup the Agent ---

def run_agentic_demo():
    # We use a tool-aware model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    tools = [lookup_salary_range, search_skill_framework]
    
    # Define the "ReAct" Prompt
    # We explicitly tell the agent it has access to tools.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Hiring Assistant. You have access to tools to look up salary data and skill definitions. "
                   "If you need information, use the tools. If you have the answer, just reply."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # This is where the tool outputs get injected back into the chat
    ])

    # Construct the Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # The Executor handles the "Loop": Think -> Act -> Observe -> Think
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # --- 3. Run Scenarios ---
    
    print("\n🔹 SCENARIO 1: Salary Check")
    query1 = "I am hiring a Bioinformatics Scientist in Boston. What is the fair salary range?"
    agent_executor.invoke({"input": query1})
    
    time.sleep(2) # Pause for readability
    
    print("\n🔹 SCENARIO 2: Skill Expansion")
    query2 = "The candidate lists 'NGS' as a skill. What other concepts is this related to?"
    agent_executor.invoke({"input": query2})

if __name__ == "__main__":
    run_agentic_demo()