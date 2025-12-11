import time
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

@tool
def lookup_salary_range(role: str, location: str) -> dict:
    """Useful for finding market salary range for a role/location."""
    print(f"🛠️ TOOL CALL: Salary lookup for '{role}' in '{location}'")
    return {"role": role, "location": location, "range": "100k-140k (Mock Data)"}

@tool
def search_skill_framework(skill: str) -> list:
    """Useful for finding related technical skills."""
    print(f"🛠️ TOOL CALL: Skill search for '{skill}'")
    return ["Python", "Pandas", "Bioinformatics"] if "bio" in skill.lower() else ["General Skill"]

def run_agentic_demo():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    tools = [lookup_salary_range, search_skill_framework]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Hiring Assistant with tool access."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("\n🔹 SCENARIO: Salary Check")
    agent_executor.invoke({"input": "What is the salary for a Bioinformatics Scientist in Boston?"})

if __name__ == "__main__":
    run_agentic_demo()