# Multi-Agent Hiring Assistant

**Students:** Mohammed Ruknuddin, Aakar Mathur, Mustafa Fatehi, Palak Sharma, Parina Parekh

**Progress:** Phase 4/5 (Agentic Prompting + Tools)

## Overview
An **Agentic RAG** system that screens job candidates using local embeddings (HuggingFace) for privacy and LLM reasoning. Uses 3-hop retrieval: broad vector search → agentic filtering → gap analysis & scoring.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_actual_api_key_here
```

### 3. Prepare Data
- **Resumes:** Place PDF files in `data/resumes/`
- **Jobs:** Place text files in `data/jobs/`

## How to Run

### Option 1: Terminal (CLI)
```bash
# Step 1: Ingest and vectorize resumes
python src/ingest.py

# Step 2: Run evaluation on all candidates
python src/main.py
```

### Option 2: Web GUI (Streamlit)
```bash
streamlit run src/app.py
```
Then interact with the application in your browser. Upload resumes and job descriptions, and view analysis reports directly in the GUI.
