# AI Readiness Capstone: Multi-Agent Hiring Assistant

**Students:** Mohammed Ruknuddin, Aakar Mathur, Mustafa Fatehi, Palak Sharma, Parina  

**Progress:** Phase 1 (Data Ingestion & Agentic RAG)

## 📖 Project Overview
This project implements an **Agentic RAG (Retrieval-Augmented Generation)** system designed to automate the initial screening of candidates for job roles. Unlike standard RAG, which simply retrieves text, this system uses a **3-Hop Agentic Logic** to retrieve, filter, and analyze candidate resumes against a specific Job Description (JD).

The system is built using **Python**, **LangChain**, **Google Gemini**, and **ChromaDB**.

---

## 🚀 Key Features (Phase 1)
* **Vector Data Ingestion:** Automatically parses PDF resumes and Text JDs, chunks them, and embeds them using Google's embedding models.
* **ChromaDB Integration:** Local vector storage for fast semantic retrieval.
* **3-Hop Agentic Retrieval:**
    1.  **Hop 1 (Broad Search):** Semantic similarity search to find top candidate chunks.
    2.  **Hop 2 (Refinement):** LLM-based filtering to discard irrelevant matches (Agentic Filter).
    3.  **Hop 3 (Gap Analysis):** LLM-based synthesis to score candidates and identify missing skills.

---

## 🛠️ Prerequisites
* **Python 3.10+** (Developed on 3.12)
* **Google Gemini API Key** (Required for Embeddings and LLM inference)

---

## ⚙️ Installation & Setup

### 1. Clone/Unzip the Repository
Ensure you are in the root directory of the project.

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
````

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

*(See `requirements.txt` content below if file is missing)*

### 4\. Configure Environment Variables

Create a file named `.env` in the root directory and add your Google API key:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

-----

## 📂 Project Structure

Ensure your data is placed in the correct folders before running.

```text
├── data/
│   ├── resumes/       <-- Place PDF resumes here (Split into individual files)
│   └── jobs/          <-- Place Job Descriptions (.txt) here
├── src/
│   ├── ingest.py      <-- Script to load PDF/Text data into ChromaDB
│   └── rag_ops.py     <-- Script containing the 3-Hop Agentic Logic
├── chroma_db/         <-- (Auto-generated) Local Vector Store
├── .env               <-- API Keys (Do not commit to version control)
└── README.md
```

-----

## 🏃‍♂️ How to Run

### Step 1: Data Preparation

1.  Place candidate resumes (PDF format) inside `data/resumes/`.
      * *Note: If using the course sample PDF, please split it into individual PDF files (e.g., `resume_1.pdf`, `resume_2.pdf`).*
2.  Place job descriptions (TXT format) inside `data/jobs/`.

### Step 2: Ingest Data (Build the Brain)

Run the ingestion script to parse documents and populate the Vector Database.

```bash
python src/ingest.py
```

*Expected Output:*

> `Loaded X pages.`
> `Split into Y chunks.`
> `Saved chunks to chroma_db`

### Step 3: Run Agentic RAG (Test the System)

Run the retrieval operation script. This will pick a job description and evaluate the resumes against it using the Multi-Hop logic.

```bash
python src/rag_ops.py
```

*Expected Output:*

> `--- HOP 1: Retrieving top 5 chunks ---`
> `--- HOP 2: Filtering Candidates ---`
> `✅ Kept: data/resumes/resume_2.pdf`
> `❌ Dropped: data/resumes/resume_4.pdf`
> `--- HOP 3: Analyzing Skill Gaps ---`
> `Candidate: ... JSON Analysis ...`

-----

## 📦 Dependencies (`requirements.txt`)

If you need to generate the requirements file, it contains:

```text
langchain
langchain-community
langchain-google-genai
chromadb
pypdf
python-dotenv
langgraph
```
