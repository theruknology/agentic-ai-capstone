from langchain_core.tools import tool

@tool
def lookup_salary_range(role: str, location: str) -> dict:
    """
    Useful for finding the market salary range for a specific job role and location.
    Use this when evaluating if a candidate is too expensive or within budget.
    """
    # Mock data for deterministic results in the capstone
    print(f"🛠️ TOOL CALL: Salary lookup for '{role}' in '{location}'")
    return {"role": role, "location": location, "market_range": "100k-140k (Mock Data)"}

@tool
def search_skill_framework(skill: str) -> list:
    """
    Useful for finding related technical skills. 
    Use this to verify if a candidate's skill is relevant (e.g., 'React' is related to 'Frontend').
    """
    print(f"🛠️ TOOL CALL: Skill search for '{skill}'")
    # Mock taxonomy
    taxonomy = {
        "python": ["Django", "Flask", "Pandas", "NumPy"],
        "react": ["Frontend", "JavaScript", "Redux"],
        "ngs": ["Bioinformatics", "Genomics"]
    }
    for key, related in taxonomy.items():
        if key in skill.lower():
            return related
    return ["General Technical Skill"]