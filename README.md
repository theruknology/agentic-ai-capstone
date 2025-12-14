# J*BLESS: Agentic AI Recruitment Architecture

![Version](https://img.shields.io/badge/version-2.0.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11-green?style=flat-square)
![Docker](https://img.shields.io/badge/docker-compose-blue?style=flat-square&logo=docker)
![Status](https://img.shields.io/badge/status-production_ready-success?style=flat-square)

This is an autonomous, event-driven recruitment platform designed to automate the technical evaluation lifecycle. 

Unlike linear automation scripts, this system utilizes a **Cyclic State Graph** (implemented via LangGraph) to model the recursive reasoning process of human recruiters. It employs a distributed architecture where the frontend (Ingestion) is decoupled from the cognitive engine (Agents) via a high-throughput Redis event bus.

---

## System Architecture

The system follows a **Domain-Driven Design (DDD)** approach, separating the user interface, infrastructure, and cognitive logic.

### 1. Event-Driven Core
The system rejects tight coupling in favor of an asynchronous **Producer-Consumer** model:
* **Producer (Portal):** Ingests PDF resumes, writes atomic metadata to a Redis Hash, and pushes a job identifier to a Redis List (`resume_queue`).
* **Consumer (Worker):** A blocking-pop worker service that consumes jobs instantly. This ensures 0ms latency between submission and processing start.

### 2. Cognitive State Machine (PEC Framework)
We implement the **Planner-Executor-Critic** framework to distribute cognitive load:
* **Planner Agent:** Deconstructs the Job Description into discrete analysis steps.
* **Executor Agents (Screener/Interviewer):** Perform semantic analysis and gap identification using Llama-3-70B.
* **Critic Agent:** Validates output for hallucinations. 
    * *Self-Correction Loop:* If the Critic rejects an output, the graph routes execution **back** to the Screener with specific feedback, enforcing an iterative refinement loop before finalization.

---

## Directory Structure

```text
agentic-ai-capstone/
├── src/
│   ├── ai_engine/          # Cognitive Layer
│   │   ├── agents.py       # Llama-3 Agent Definitions
│   │   ├── graph.py        # LangGraph State Machine & Routing Logic
│   │   ├── rag.py          # Vector Retrieval (ChromaDB + HuggingFace)
│   │   └── tools.py        # Deterministic Tools (Salary/Skill Lookup)
│   ├── infra/              # Infrastructure Layer
│   │   ├── db.py           # Redis Connection Factory
│   │   ├── ingest.py       # PDF Parsing & Chunking Strategies
│   │   └── notifier.py     # Discord Webhook Integration
│   └── ui/                 # Presentation Layer
│       ├── candidate_portal.py # Public Application Interface
│       └── recruiter_dash.py   # Analytics & Reporting Dashboard
├── worker.py               # Background Consumer Service
├── evaluation.py               
└── requirements.txt        # Dependencies
````

-----

## Quick Start (Docker)

The recommended way to run the system is via Docker Compose, which handles the Redis dependency and networking automatically.

### 1\. Configuration

Create a `.env` file in the root directory:

```ini
# Inference Engine
GROQ_API_KEY=gsk_...

# Infrastructure
REDIS_HOST=redis
REDIS_PORT=6379

# Notifications
DISCORD_WEBHOOK_URL=[https://discord.com/api/webhooks/](https://discord.com/api/webhooks/)...
```

### 2\. Access Points

  * **Candidate Portal:** [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)
  * **Recruiter Dashboard:** [http://localhost:8502](https://www.google.com/search?q=http://localhost:8502)
  * **Redis Insight (Database GUI):** [http://localhost:8001](https://www.google.com/search?q=http://localhost:8001)


## Development Setup (Manual)

If you need to debug individual components locally:

**Prerequisites:**

  * Python 3.10+
  * Local Redis instance running on port 6379

**Installation:**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Running Components:**
Run each service in a separate terminal to simulate the distributed environment.

```bash
# Terminal 1: The Worker (Cognitive Engine)
python src/worker.py

# Terminal 2: The Portal (Ingestion)
streamlit run src/ui/candidate_portal.py

# Terminal 3: The Dashboard (Analytics)
streamlit run src/ui/recruiter_dash.py
```

-----

## Technical Highlights

  * **Self-Correction:** The LangGraph implementation features a conditional edge at the `Critic` node. It routes to `END` only upon validation success; otherwise, it triggers a `REFINE` edge, passing error context back to the upstream agent.
  * **Privacy-First RAG:** Candidate resumes are embedded locally using `sentence-transformers/all-MiniLM-L6-v2`. No sensitive biometric data is sent to external embedding endpoints.
  * **Tool Binding:** Agents are not restricted to text generation; they are bound to Python functions (`lookup_salary`, `search_skills`), allowing for deterministic data retrieval during the reasoning phase.

