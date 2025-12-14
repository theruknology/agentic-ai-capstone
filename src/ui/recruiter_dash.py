import sys
import streamlit as st
import os
import shutil
import pandas as pd
import json
import time

# Import our backend modules
# We import the functions directly to run them from the UI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.infra.ingest import load_documents, chunk_documents, save_to_chroma
from src.ai_engine.graph import HiringOrchestrator

# Configuration
DATA_DIR = "data"
RESUME_DIR = os.path.join(DATA_DIR, "resumes")
JOB_DIR = os.path.join(DATA_DIR, "jobs")
REPORTS_DIR = "reports"

# Ensure directories exist
os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs(JOB_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- UI Layout ---
st.set_page_config(page_title="AI Hiring Agent", page_icon="🤖", layout="wide")

st.title("🤖 Agentic AI Hiring Assistant")
st.markdown("### Powered by Groq (Llama 3), LangChain & Streamlit")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Configuration")
    api_status = "✅ Connected" if os.getenv("GROQ_API_KEY") else "❌ Missing Key"
    st.write(f"API Status: {api_status}")
    
    st.divider()
    st.write("**System Status**")
    if os.path.exists("chroma_db"):
        st.success("🧠 Memory (Vector DB): Ready")
    else:
        st.warning("🧠 Memory: Empty")

# --- TAB STRUCTURE ---
tab1, tab2, tab3 = st.tabs(["📂 1. Upload Data", "🚀 2. Run Agents", "📊 3. Analysis Report"])

# --- TAB 1: DATA UPLOAD ---
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Resumes (PDF)")
        uploaded_files = st.file_uploader("Drop candidate resumes here", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Process Resumes (Ingest)"):
                with st.status("Processing Resumes..."):
                    # 1. Clear old data
                    if os.path.exists(RESUME_DIR):
                        shutil.rmtree(RESUME_DIR)
                    os.makedirs(RESUME_DIR)
                    
                    # 2. Save new files
                    for file in uploaded_files:
                        with open(os.path.join(RESUME_DIR, file.name), "wb") as f:
                            f.write(file.getbuffer())
                    st.write(f"✅ Saved {len(uploaded_files)} files.")
                    
                    # 3. Run Ingestion (Calls ingest.py logic)
                    st.write("⚙️ Parsing PDFs...")
                    docs = load_documents()
                    st.write("✂️ Chunking text...")
                    chunks = chunk_documents(docs)
                    st.write("💾 Creating Vector Embeddings...")
                    save_to_chroma(chunks)
                    st.success("Ingestion Complete!")
                    time.sleep(1)
                    st.rerun()

    with col2:
        st.subheader("Job Description")
        job_text = st.text_area("Paste the Job Description here", height=300)
        
        if st.button("Save Job Description"):
            with open(os.path.join(JOB_DIR, "current_job.txt"), "w") as f:
                f.write(job_text)
            st.success("Job Description Saved!")

# --- TAB 2: ORCHESTRATION ---
with tab2:
    st.subheader("Agentic Orchestration")
    st.info("This will trigger the multi-agent pipeline: RAG -> Planner -> Screener -> Interviewer -> Critic.")
    
    if st.button("🚀 Start Hiring Pipeline", type="primary"):
        job_path = os.path.join(JOB_DIR, "current_job.txt")
        
        if not os.path.exists(job_path):
            st.error("Please save a Job Description first!")
        elif not os.path.exists("chroma_db"):
            st.error("Please ingest resumes first!")
        else:
            # Run the Orchestrator
            orchestrator = HiringOrchestrator()
            
            # We redirect stdout to UI logs (simple version)
            status_placeholder = st.empty()
            with st.status("🤖 Agents are working...", expanded=True) as status:
                st.write("🔍 RAG Agent: Retrieving candidates...")
                # We run the actual backend logic
                orchestrator.run_workflow(job_path)
                st.write("✅ Pipeline Finished!")
                status.update(label="Evaluation Complete!", state="complete", expanded=False)
            
            st.success("All candidates processed! Go to 'Analysis Report' tab.")

# --- TAB 3: REPORTING ---
with tab3:
    st.subheader("Candidate Rankings")
    
    # Load all reports
    report_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith("_report.json")]
    
    if not report_files:
        st.warning("No reports found. Run the pipeline first.")
    else:
        results = []
        for rf in report_files:
            with open(os.path.join(REPORTS_DIR, rf)) as f:
                data = json.load(f)
                
                # Extract key metrics for the table
                screening = data["evaluation"]["screening"]
                results.append({
                    "Candidate ID": data["meta"]["id"],
                    "Source": data["meta"]["source"],
                    "Match Score": f"{screening.get('fit_score', 0) * 100:.1f}%",
                    "Decision": screening.get("reasoning", "N/A")[:100] + "...", # Truncate
                    "Missing Skills": ", ".join(screening.get("missing_skills", [])[:3]),
                    "Filename": rf
                })
        
        df = pd.DataFrame(results)
        
        # Display Interactive Table
        st.dataframe(df, width="stretch")
        
        st.divider()
        
        # Detailed View
        selected_candidate = st.selectbox("Select Candidate for Full Details", df["Candidate ID"])
        
        if selected_candidate:
            # Find the full JSON for this candidate
            row = df[df["Candidate ID"] == selected_candidate].iloc[0]
            with open(os.path.join(REPORTS_DIR, row["Filename"])) as f:
                full_data = json.load(f)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 📋 Screening Plan")
                st.json(full_data["evaluation"]["plan"])
                
                st.markdown("### 🎤 Interview Questions")
                for q in full_data["evaluation"]["interview_questions"].get("questions", []):
                    st.write(f"- {q}")

            with c2:
                st.markdown("### ⚖️ Critic's Review")
                critique = full_data["evaluation"]["critic_review"]
                if critique.get("critique_passed"):
                    st.success("Critique Passed: Consistent Evaluation")
                else:
                    st.error("Critique Flagged Issues")
                st.write(critique.get("feedback"))
                
                st.markdown("### 💻 Skill Assessment")
                st.json(full_data["evaluation"]["skill_assessment"])