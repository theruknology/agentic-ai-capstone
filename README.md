# Multi-Agent Hiring Assistant

**Students:** Mohammed Ruknuddin, Aakar Mathur, Mustafa Fatehi, Palak Sharma, Parina Parekh

**Progress:** Phase 1 (Data Ingestion & Agentic RAG)

## 📖 Overview
This project implements an **Agentic RAG** system to screen job candidates. It uses **Local Embeddings** (HuggingFace) for privacy and cost-efficiency, and **Google Gemini** for reasoning.

## 🚀 Key Features
* **Hybrid Architecture:** Uses local CPU for vector embeddings (Unlimited/Free) and Gemini API for reasoning.
* **3-Hop Agentic Retrieval:**
    1.  **Broad Search:** Retrieves top candidate chunks via Vector Search.
    2.  **Agentic Filter:** LLM evaluates relevance and **deduplicates** results.
    3.  **Gap Analysis:** LLM identifies missing skills and scores the candidate.
* **Robustness:** Includes **Automatic Throttling** (sleep timers) to respect Gemini Free Tier rate limits (5 RPM).

## 🛠️ Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
````

### 2\. Configure Environment

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

### 3\. Prepare Data

  * **Resumes:** Place PDF files in `data/resumes/` (Split merged PDFs into individual files).
  * **Jobs:** Place text files in `data/jobs/`.

## 🏃‍♂️ How to Run

### Step 1: Ingest Data (Build "Memory")

Parses PDFs and saves vectors locally using `HuggingFace all-MiniLM-L6-v2`.

```bash
python src/ingest.py
```

### Step 2: Run Agentic RAG (The "Brain")

Retrieves and analyzes candidates against the job description.

```bash
python src/rag_ops.py
```

*\> **Note:** The script pauses for 15 seconds between candidates to prevent "429 Rate Limit" errors on the Gemini Free Tier.*
